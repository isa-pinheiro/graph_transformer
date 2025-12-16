import math
import torch
import torch.nn as nn
import dgl.sparse as dglsp

class MultiHeadedSparseAttentionModule(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dh = hidden_size // num_heads

        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.linear_e = nn.Linear(hidden_size, hidden_size)

        self.linear_oh = nn.Linear(hidden_size, hidden_size)
        self.linear_oe = nn.Linear(hidden_size, hidden_size)

    def forward(self, A: dglsp.SparseMatrix, h_x: torch.Tensor, h_e: torch.Tensor):
        N = h_x.shape[0]
        E = h_e.shape[0]
        nh = self.num_heads
        dh = self.dh

        q = self.linear_q(h_x).view(N, dh, nh)
        k = self.linear_k(h_x).view(N, dh, nh)
        v = self.linear_v(h_x).view(N, dh, nh)
        e = self.linear_e(h_e).view(E, dh, nh)

        scores = dglsp.sddmm(A, q, k.transpose(0,1))
        scores = scores / math.sqrt(dh)

        e_gate = e.sum(dim=1)
        scores = dglsp.val_like(scores, scores.val * e_gate)

        oe_feat = scores.val.unsqueeze(1) * e
        oe = self.linear_oe(oe_feat.reshape(E, self.hidden_size))

        scores = dglsp.softmax(scores, dim=1)

        h_out = dglsp.bspmm(scores, v)

        h_out = h_out.reshape(N, self.hidden_size)
        oh = self.linear_oh(h_out)

        return oh, oe
