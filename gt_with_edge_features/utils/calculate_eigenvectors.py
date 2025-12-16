import torch
from tqdm import tqdm
from dgl import lap_pe

def k_eigenvectors(dataset, k):
    for idx in tqdm(range(len(dataset)), desc="Computing Laplacian PE"):
        g, _ = dataset[idx]
        g.ndata["PE"] = lap_pe(
            g,
            k=k,
            padding=True
        )
