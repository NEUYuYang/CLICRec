import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from .layers import _mlp

class ContextAwareFusion(nn.Module):

    def __init__(self, config):
        super().__init__()
        d = config.embedding_size
        self.context_gru = nn.GRU(d, d, batch_first=True)
        self.gate = _mlp(3 * d, config.mlp_hidden_size, 1, config.dropout_prob)

    def encode_context(self, item_embeddings, item_seq_len):
        packed = pack_padded_sequence(
            item_embeddings, item_seq_len.detach().cpu(),
            batch_first=True, enforce_sorted=False,
        )
        _, hidden = self.context_gru(packed)
        return hidden[-1]

    def forward(self, context, long_interest, short_interest):
        alpha = torch.sigmoid(self.gate(torch.cat((context, long_interest, short_interest), -1)))
        fused = alpha * long_interest + (1 - alpha) * short_interest
        return fused, alpha