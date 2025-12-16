import torch
import torch.nn as nn
from components.multiheaded_sparse_attention import MultiHeadedSparseAttentionModule


class GraphTransformerLayer(nn.Module):
    def __init__(self, hidden_size:int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.attention = MultiHeadedSparseAttentionModule(hidden_size,num_heads)

        self.bnh_1 = nn.BatchNorm1d(hidden_size)
        self.bnh_2 = nn.BatchNorm1d(hidden_size)

        self.bne_1 = nn.BatchNorm1d(hidden_size)
        self.bne_2 = nn.BatchNorm1d(hidden_size)

        self.ffn_h = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.ReLU(),
            nn.Linear(2 * hidden_size, hidden_size),
        )

        self.ffn_e = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.ReLU(),
            nn.Linear(2 * hidden_size, hidden_size),
        )

    def forward(self, A: torch.Tensor, h_x: torch.Tensor, h_e: torch.Tensor):
        h_att, e_att = self.attention(A, h_x, h_e)

        h = self.bnh_1(h_x + h_att)
        h_ffn = self.ffn_h(h)
        h = self.bnh_2(h + h_ffn)

        e = self.bne_1(h_e + e_att)
        e_ffn = self.ffn_e(e)
        e = self.bne_2(e + e_ffn)

        return h, e