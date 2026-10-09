import torch
from torch import nn
from .layers import CausalInterestBlock, _mlp

class LongTermInterestEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        d = config.embedding_size
        self.projection = nn.Linear(d, d, bias=False)
        self.attention = _mlp(4 * d, config.mlp_hidden_size, 1, config.dropout_prob)

    def forward(self, user_embeddings, item_embeddings, valid_mask):
        # [B,d]、[B,L,d]、[B,L]
        query = user_embeddings[:, None, :].expand_as(item_embeddings)
        mapped = self.projection(item_embeddings)
        features = torch.cat((mapped, query, mapped - query, mapped * query), dim=-1)
        weights = self.attention(features).squeeze(-1)
        weights = weights.masked_fill(~valid_mask, -torch.inf).softmax(-1)
        return (weights.unsqueeze(-1) * item_embeddings).sum(1)


class ShortTermInterestEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.max_length = config.short_seq_length
        d = config.embedding_size
        self.position_embedding = nn.Embedding(self.max_length, d)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.blocks = nn.ModuleList([
            CausalInterestBlock(d, config.num_heads, config.dropout_prob)
            for _ in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(d)

    def forward(self, item_seq, item_seq_len, item_embedding):
        width = min(item_seq.size(1), self.max_length)
        short_lengths = item_seq_len.clamp_max(width)
        local_position = torch.arange(width, device=item_seq.device)
        index = (item_seq_len - short_lengths)[:, None] + local_position[None, :]
        recent = item_seq.gather(1, index.clamp_max(item_seq.size(1) - 1))
        padding = local_position[None, :] >= short_lengths[:, None]
        recent = recent.masked_fill(padding, 0)
        z = self.dropout(item_embedding(recent) + self.position_embedding(local_position)[None])
        z = z.masked_fill(padding.unsqueeze(-1), 0)
        for block in self.blocks:
            z = block(z, padding)
        z = self.norm(z)
        return z[torch.arange(z.size(0), device=z.device), short_lengths - 1]