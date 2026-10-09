import torch
from torch import nn
import torch.nn.functional as F
from .layers import GraphAttention

class GALSTM(nn.Module):

    def __init__(self, dim, heads, dropout, time_epsilon):
        super().__init__()
        self.time_epsilon = time_epsilon
        self.current = nn.ModuleDict({g: GraphAttention(dim, dim, heads, dropout=dropout)
                                      for g in ('i', 'f', 'c', 'o', 'interval', 'span')})
        self.previous = nn.ModuleDict({g: GraphAttention(dim, dim, heads, dropout=dropout)
                                       for g in ('i', 'f', 'c', 'o')})
        self.interval_embedding = nn.Linear(1, dim)
        self.span_embedding = nn.Linear(1, dim)
        self.interval_gate = nn.Linear(dim, dim)
        self.span_gate = nn.Linear(dim, dim)
        self.interval_output = nn.Linear(dim, dim, bias=False)
        self.span_output = nn.Linear(dim, dim, bias=False)

    def _time_feature(self, value, x, layer):
        value = torch.as_tensor(value, device=x.device, dtype=x.dtype)
        if value.numel() not in (1, x.size(0)):
            raise ValueError('time input must be scalar or one value per graph node')
        if not torch.isfinite(value).all() or (value < 0).any():
            raise ValueError('time differences must be finite and nonnegative')
        value = value.reshape(-1, 1)
        return torch.tanh(layer(value.clamp_min(self.time_epsilon).log()))

    def forward(self, x, session, h=None, c=None, previous_edge_index=None):
        if h is None:
            h = torch.zeros_like(x)
        if c is None:
            c = torch.zeros_like(x)
        gates = {g: self.current[g](x, session.edge_index) for g in ('i', 'f', 'c', 'o')}
        if previous_edge_index is not None:
            for g in gates:
                gates[g] = gates[g] + self.previous[g](h, previous_edge_index)
        delta = self._time_feature(session.interval, x, self.interval_embedding)
        span = self._time_feature(session.span, x, self.span_embedding)
        t_delta = torch.sigmoid(self.current['interval'](x, session.edge_index) +
                                self.interval_gate(delta))
        t_span = torch.sigmoid(self.current['span'](x, session.edge_index) + self.span_gate(span))
        new_c = (torch.sigmoid(gates['f']) * t_delta * c +
            torch.sigmoid(gates['i']) * t_span * torch.tanh(gates['c']))

        out = torch.sigmoid(gates['o'] + self.interval_output(delta) + self.span_output(span))
        new_h = out * torch.tanh(new_c)
        if session.active_mask is not None:
            if session.active_mask.shape != (x.size(0),) or session.active_mask.dtype != torch.bool:
                raise ValueError('active_mask must be bool [num_users+num_items]')
            mask = session.active_mask[:, None]
            new_h, new_c = torch.where(mask, new_h, h), torch.where(mask, new_c, c)
        return new_h, new_c


class StaticGraphEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.gat1 = GraphAttention(
            config.embedding_size, config.gat_hidden_dim, config.gat_num_heads,
            concat=True, dropout=config.gat_dropout,
        )
        self.gat2 = GraphAttention(
            config.gat_hidden_dim * config.gat_num_heads, config.embedding_size,
            dropout=config.gat_dropout,
        )

    def forward(self, node_embeddings, edge_index):
        return self.gat2(F.elu(self.gat1(node_embeddings, edge_index)), edge_index)


class DynamicGraphEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.cell = GALSTM(
            config.embedding_size, config.gat_num_heads,
            config.gat_dropout, config.time_epsilon,
        )

    def forward(self, node_embeddings, sessions):
        if not sessions:
            raise ValueError('at least one preprocessed session graph is required')
        h = c = previous = None
        for session in sessions:
            h, c = self.cell(node_embeddings, session, h, c, previous)
            previous = session.edge_index
        return h