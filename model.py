from typing import Optional
import torch
from torch import nn
from .config import CLICRecConfig
from .inputs import GraphInputs
from .encoders import LongTermInterestEncoder, ShortTermInterestEncoder
from .fusion import ContextAwareFusion
from .graphs import StaticGraphEncoder, DynamicGraphEncoder
from .layers import _mlp
from .losses import joint_loss
from .prototypes import refresh_prototypes


class CLICRecModel(nn.Module):

    def __init__(self, config: CLICRecConfig):
        super().__init__()
        self.config = config
        d = config.embedding_size
        self.user_embedding = nn.Embedding(config.num_users, d, padding_idx=0)
        self.item_embedding = nn.Embedding(config.num_items, d, padding_idx=0)
        self.long_encoder = LongTermInterestEncoder(config)
        self.short_encoder = ShortTermInterestEncoder(config)
        self.fusion = ContextAwareFusion(config)
        self.long_graph = StaticGraphEncoder(config)
        self.short_graph = DynamicGraphEncoder(config)
        self.long_link_mlp = _mlp(2 * d, (d,), 1, config.dropout_prob)
        self.short_link_mlp = _mlp(2 * d, (d,), 1, config.dropout_prob)
        for name, count in (('long', config.long_clusters), ('short', config.short_clusters)):
            self.register_buffer(f'{name}_centers', torch.zeros(count, d))
            self.register_buffer(f'{name}_prototypes', torch.zeros(count, d))
            self.register_buffer(f'{name}_labels', torch.full((config.num_users,), -1, dtype=torch.long))
        self.register_buffer('prototypes_ready', torch.tensor(False))
        self.apply(self._init_weights)
        with torch.no_grad():
            self.user_embedding.weight[0].zero_()
            self.item_embedding.weight[0].zero_()

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(module.weight)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode_interests(self, user, item_seq, item_seq_len):
        if item_seq.ndim != 2 or item_seq.size(0) == 0 or item_seq.size(1) == 0:
            raise ValueError('item_seq must be nonempty [B,L]')
        if user.shape != item_seq_len.shape or user.shape != (item_seq.size(0),):
            raise ValueError('user and item_seq_len must be [B]')
        if (item_seq_len < 1).any() or (item_seq_len > item_seq.size(1)).any():
            raise ValueError('each history length must be in [1,L]')
        positions = torch.arange(item_seq.size(1), device=item_seq.device)
        valid = positions[None, :] < item_seq_len[:, None]
        if (item_seq[valid] == 0).any() or (item_seq[~valid] != 0).any():
            raise ValueError('histories must contain nonzero IDs followed by right padding 0')
        values = self.item_embedding(item_seq)
        long_interest = self.long_encoder(self.user_embedding(user), values, valid)
        short_interest = self.short_encoder(item_seq, item_seq_len, self.item_embedding)
        context = self.fusion.encode_context(values, item_seq_len)
        return long_interest, short_interest, context

    def forward(self, user, item_seq, item_seq_len, return_details=False):
        long_interest, short_interest, context = self.encode_interests(user, item_seq, item_seq_len)
        fused, alpha = self.fusion(context, long_interest, short_interest)
        if return_details:
            return {'long_interest': long_interest, 'short_interest': short_interest,
                    'context': context, 'alpha': alpha, 'fused': fused}
        return fused

    def encode_graphs(self, graphs: GraphInputs):
        x = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        return self.long_graph(x, graphs.long_edge_index), self.short_graph(x, graphs.sessions)

    def refresh_prototypes(self, history_batches, graphs: GraphInputs):
        return refresh_prototypes(self, history_batches, graphs)

    def calculate_loss(self, batch, graphs: Optional[GraphInputs] = None, return_components=False):
        return joint_loss(self, batch, graphs, return_components)

    def predict(self, batch, candidate_ids=None):
        if candidate_ids is None:
            candidate_ids = batch['item_id']
        fused = self(batch['user_id'], batch['item_seq'], batch['item_seq_len'])
        if candidate_ids.ndim not in (1, 2) or candidate_ids.size(0) != fused.size(0):
            raise ValueError('candidate_ids must be [B] or [B,K]')
        items = self.item_embedding(candidate_ids)
        score = (fused * items).sum(-1) if items.ndim == 2 else (fused[:, None] * items).sum(-1)
        return score.masked_fill(candidate_ids == 0, -torch.inf)

    def full_sort_predict(self, batch):
        fused = self(batch['user_id'], batch['item_seq'], batch['item_seq_len'])
        scores = fused @ self.item_embedding.weight.T
        return scores.masked_fill(torch.arange(self.config.num_items, device=scores.device)[None] == 0, -torch.inf)