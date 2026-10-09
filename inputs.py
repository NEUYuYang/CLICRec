from dataclasses import dataclass
from typing import Optional, Sequence
from torch import Tensor

@dataclass
class SessionGraph:  
    edge_index: Tensor  # [2,E]
    interval: Tensor
    span: Tensor
    active_mask: Optional[Tensor] = None


@dataclass
class GraphInputs:
    long_edge_index: Tensor
    sessions: Sequence[SessionGraph]
    long_pos_edges: Optional[Tensor] = None  # [2,E_pos]
    long_neg_edges: Optional[Tensor] = None  # [2,E_neg]
    short_pos_edges: Optional[Tensor] = None  # [2,E_pos]
    short_neg_edges: Optional[Tensor] = None  # [2,E_neg]