import math
import torch
import torch.nn as nn
import dgl.sparse as dglsp

class MultiHeadedSparseAttentionModule(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dh = hidden_size//num_heads

        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.linear_o = nn.Linear(hidden_size, hidden_size)

    def forward(self, A: dglsp.SparseMatrix, h: torch.Tensor):
        N = h.shape[0]
        nh = self.num_heads
        dh = self.dh

        q = self.linear_q(h).view(N, dh, nh)
        k = self.linear_k(h).view(N, dh, nh)
        v = self.linear_v(h).view(N, dh, nh)

        scores = dglsp.sddmm(A, q, k.transpose(0, 1))

        scores = scores / math.sqrt(dh)

        scores = dglsp.softmax(scores, dim=1)

        out = dglsp.bspmm(scores, v)

        out = out.reshape(N, self.hidden_size)

        return self.linear_o(out)
