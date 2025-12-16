import torch
import torch.nn as nn
from components.multiheaded_sparse_attention import MultiHeadedSparseAttentionModule

class GraphTransformerLayer(nn.Module):
    def __init__(self, hidden_size:int, num_heads: int):
        super().__init__()
        self.mhsa = MultiHeadedSparseAttentionModule(hidden_size, num_heads)

        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 2*hidden_size),
            nn.ReLU(),
            nn.Linear(2*hidden_size, hidden_size),
        )

    def forward(self, A:torch.Tensor, h:torch.Tensor):
        # A: representa matriz de adjacencia do grafo. tamanho (n, n)
        # h: representa o vetor de hidden representations do nós. tamanho (n, hidden_size)
        
        h1 = self.mhsa(A, h)
        h = self.bn1(h+h1)
        h2 = self.ffn(h)
        h = self.bn2(h+h2)
        
        return h
