import dgl
import torch
import torch.nn as nn
import dgl.sparse as dglsp
from ogb.graphproppred.mol_encoder import AtomEncoder
from components.graph_transformer_layer import GraphTransformerLayer

class GraphTransformerModel(nn.Module):
    def __init__(self,
                 num_atom_type: int,
                 num_bond_type: int,
                 pos_enc_dim: int,
                 out_size: int, 
                 hidden_size: int=16,
                 num_heads: int=8, 
                 num_layers: int=10):
        super().__init__()

        self.embedding_h = nn.Embedding(num_atom_type, hidden_size)
        self.embedding_e = nn.Embedding(num_bond_type, hidden_size)

        self.pos_linear = nn.Linear(pos_enc_dim, hidden_size)

        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_size, num_heads) for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.ReLU(),
            nn.Linear(hidden_size//4, out_size),
        )

    def forward(self, g: dgl.DGLGraph, h: torch.Tensor, pos_enc: torch.Tensor, e: torch.Tensor):
        indices = torch.stack(g.edges())
        N = g.num_nodes()
        A = dglsp.spmatrix(indices, shape=(N, N))

        h = self.embedding_h(h)
        h_pos_enc = self.pos_linear(pos_enc.float())
        h = h + h_pos_enc

        e = self.embedding_e(e)

        for layer in self.layers:
            h, e = layer(A, h, e)

        g.ndata['h'] = h
        h = dgl.mean_nodes(g, 'h')
        
        h = self.classifier(h)

        return h