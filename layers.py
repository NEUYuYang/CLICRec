import torch
from torch import nn
import torch.nn.functional as F

def _mlp(input_dim, hidden_dims, output_dim, dropout):
    layers = []
    for width in hidden_dims:
        layers.extend((nn.Linear(input_dim, width), nn.ReLU(), nn.Dropout(dropout)))
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class GraphAttention(nn.Module):

    def __init__(self, in_dim, out_dim, heads=1, concat=False, dropout=0.0):
        super().__init__()
        self.heads, self.out_dim, self.concat, self.dropout = heads, out_dim, concat, dropout
        self.projection = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(heads, out_dim))
        self.att_dst = nn.Parameter(torch.empty(heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(heads * out_dim if concat else out_dim))
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x, edge_index):
        n = x.size(0)
        src, dst = edge_index
        nonself = src != dst
        loop = torch.arange(n, device=x.device)
        src = torch.cat((src[nonself], loop))
        dst = torch.cat((dst[nonself], loop))
        z = self.projection(x).view(n, self.heads, self.out_dim)
        score = F.leaky_relu((z[src] * self.att_src).sum(-1) +
                            (z[dst] * self.att_dst).sum(-1), negative_slope=0.2)
        index = dst[:, None].expand(-1, self.heads)
        maxima = score.new_full((n, self.heads), -torch.inf)
        maxima.scatter_reduce_(0, index, score.detach(), reduce='amax', include_self=True)
        exp_score = (score - maxima[dst]).exp()
        denom = score.new_zeros((n, self.heads)).scatter_add_(0, index, exp_score)
        attention = F.dropout(exp_score / denom[dst].clamp_min(1e-12),
                              p=self.dropout, training=self.training)
        output = z.new_zeros(z.shape)
        output.index_add_(0, dst, attention.unsqueeze(-1) * z[src])
        output = output.flatten(1) if self.concat else output.mean(1)
        return output + self.bias


class CausalInterestBlock(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn = _mlp(dim, (4 * dim,), dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask):
        causal = torch.ones(x.size(1), x.size(1), dtype=torch.bool, device=x.device).triu(1)
        z = self.norm1(x)
        attended = self.attention(z, z, z, attn_mask=causal,
                                  key_padding_mask=padding_mask, need_weights=False)[0]
        x = x + self.dropout(attended)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x.masked_fill(padding_mask.unsqueeze(-1), 0)