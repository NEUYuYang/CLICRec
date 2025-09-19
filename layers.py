import torch
from torch.nn import Parameter
from torch_geometric.nn import GATConv
from torch_geometric.nn.inits import glorot, zeros
class GALSTM(torch.nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            head: int,
    ):
        super(GALSTM, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.head = head
        self._create_parameters_and_layers()
        self._set_parameters()
        self.last_edge_index = None

    def _create_time_interval(self):
        self.W_ti = Parameter(torch.Tensor(1, self.out_channels))
        self.b_ti = Parameter(torch.Tensor(1, self.out_channels))
        '''More details will be made public after the paper is published.'''
        self.W_ti2 = Parameter(torch.Tensor(self.out_channels, self.out_channels))

    def _create_time_span(self):
        self.W_ts = Parameter(torch.Tensor(1, self.out_channels))
        self.b_ts = Parameter(torch.Tensor(1, self.out_channels))
        self.conv_ts = GATConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            heads=self.head,
            concat=False
        )
        self.W_ts1 = Parameter(torch.Tensor(self.out_channels, self.out_channels))
        self.b_ts1 = Parameter(torch.Tensor(1, self.out_channels))
        self.W_ts2 = Parameter(torch.Tensor(self.out_channels, self.out_channels))

    def _create_input_gate_parameters_and_layers(self):

        self.conv_i = GATConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            heads=self.head,
            concat=False
        )

        '''More details will be made public after the paper is published.'''

    def _create_forget_gate_parameters_and_layers(self):

        self.conv_f = GATConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            heads=self.head,
            concat=False
        )

        self.conv_fh = GATConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            heads=self.head,
            concat=False
        )
        self.b_f = Parameter(torch.Tensor(1, self.out_channels))

    def _create_cell_state_parameters_and_layers(self):

        self.conv_c = GATConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            heads=self.head,
            concat=False
        )

        self.conv_ch = GATConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            heads=self.head,
            concat=False
        )
        self.b_c = Parameter(torch.Tensor(1, self.out_channels))

    def _create_output_gate_parameters_and_layers(self):

        self.conv_o = GATConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            heads=self.head,
            concat=False
        )

        self.conv_oh = GATConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            heads=self.head,
            concat=False
        )
        self.b_o = Parameter(torch.Tensor(1, self.out_channels))

    def _create_parameters_and_layers(self):
        self._create_time_interval()
        self._create_time_span()
        self._create_input_gate_parameters_and_layers()
        '''More details will be made public after the paper is published.'''
        self._create_output_gate_parameters_and_layers()

    def _set_parameters(self):
        glorot(self.W_ti)
        glorot(self.W_ti1)
        glorot(self.W_ti2)
        glorot(self.W_ts)
        glorot(self.W_ts1)
        glorot(self.W_ts2)
        zeros(self.b_ti)
        zeros(self.b_ti1)
        zeros(self.b_ts)
        zeros(self.b_ts1)
        zeros(self.b_i)
        zeros(self.b_f)
        zeros(self.b_c)
        zeros(self.b_o)

    def _set_time_interval(self, X, edge_index, interval):

        self.tif = torch.tanh(torch.matmul(interval.unsqueeze(1), self.W_ti.T) + self.b_ti)
        ti = torch.sigmoid(
            self.conv_ti(X, edge_index) + torch.matmul(torch.mean(self.tif, dim=0), self.W_ti1) + self.b_ti1)
        return ti

    def _set_time_span(self, X, edge_index, span):
        self.tsf = torch.tanh(torch.matmul(span.unsqueeze(1), self.W_ts.T) + self.b_ts)
        '''More details will be made public after the paper is published.'''
        return ts

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _set_cell_state(self, X, C):
        if C is None:
            C = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return C

    def _calculate_input_gate(self, X, edge_index, edge_weight, H, C):
        I = self.conv_i(X, edge_index, edge_weight)
        if self.last_edge_index != None:
            I = I + self.conv_ih(H, self.last_edge_index, edge_weight)
        I = I + self.b_i
        I = torch.sigmoid(I)
        return I

    def _calculate_forget_gate(self, X, edge_index, edge_weight, H, C):
        F = self.conv_f(X, edge_index, edge_weight)
        '''More details will be made public after the paper is published.'''
        F = torch.sigmoid(F)
        return F

    def _calculate_cell_state(self, X, edge_index, edge_weight, H, C, I, F, intevral, span):
        T = self.conv_c(X, edge_index, edge_weight)
        if self.last_edge_index != None:
            T = T + self.conv_ch(H, self.last_edge_index, edge_weight)
        T = T + self.b_c
        T = torch.tanh(T)
        C = F * intevral * C + I * span * T
        return C

    def _calculate_output_gate(self, X, edge_index, edge_weight, H, C):
        O = self.conv_o(X, edge_index, edge_weight)
        if self.last_edge_index != None:
            O = O + self.conv_oh(H, self.last_edge_index, edge_weight)
        O = O + torch.matmul(torch.mean(self.tif, dim=0), self.W_ti2) + torch.matmul(torch.mean(self.tif, dim=0),
                                                                                     self.W_ti2) + self.b_o
        O = torch.sigmoid(O)
        return O

    def _calculate_hidden_state(self, O, C):
        H = O * torch.tanh(C)
        return H

    def forward(
            self,
            X: torch.FloatTensor,
            edge_index: torch.LongTensor,
            intevrals,
            spans,
            edge_weight: torch.FloatTensor = None,
            H: torch.FloatTensor = None,
            C: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        H = self._set_hidden_state(X, H)
        C = self._set_cell_state(X, C)
        '''More details will be made public after the paper is published.'''
        C = self._calculate_cell_state(X, edge_index, edge_weight, H, C, I, F, ti, si)
        O = self._calculate_output_gate(X, edge_index, edge_weight, H, C)
        H = self._calculate_hidden_state(O, C)
        self.last_edge_index = edge_index
        return H, C