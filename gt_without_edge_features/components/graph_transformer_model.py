import dgl
import torch
import torch.nn as nn
import dgl.sparse as dglsp
from components.graph_transformer_layer import GraphTransformerLayer

class GraphTransformerModel(nn.Module):
    def __init__(self, 
                 num_atom_type: int, 
                 pos_enc_dim: int,
                 out_size: int, 
                 hidden_size: int=16, 
                 num_heads: int=8, 
                 num_layers: int=10):
        super().__init__()
        
        self.embedding = nn.Embedding(num_atom_type, hidden_size)

        self.pos_linear = nn.Linear(pos_enc_dim, hidden_size)

        self.layers = nn.ModuleList(
            [GraphTransformerLayer(hidden_size, num_heads) for _ in range(num_layers-1)]
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.ReLU(),
            nn.Linear(hidden_size//4, out_size),
        )

    def forward(self, g: dgl.DGLGraph, h: torch.Tensor, pos_enc: torch.Tensor):
        indices = torch.stack(g.edges())    
        N = g.num_nodes()
        A = dglsp.spmatrix(indices, shape=(N, N))


        h = self.embedding(h)
        h_pos_enc = self.pos_linear(pos_enc.float())
        h = h + h_pos_enc

        for layer in self.layers:
            h = layer(A, h)
            
        g.ndata['h'] = h
        h = dgl.mean_nodes(g, 'h')

        h = self.classifier(h)
        return h