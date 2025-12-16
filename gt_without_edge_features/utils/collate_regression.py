import torch
import dgl
import numpy as np
def collate(samples):
    graphs, labels = map(list, zip(*samples))
    labels = torch.tensor(np.array(labels)).unsqueeze(1)
    batched_graph = dgl.batch(graphs)       
        
    return batched_graph, labels