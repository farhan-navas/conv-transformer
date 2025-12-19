import torch
from torch.utils.data import Dataset
import numpy as np


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings_npy_path: str):
        self.embs = np.load(embeddings_npy_path).astype(np.float32, copy=False)
        if self.embs.ndim != 2:
            raise ValueError(f"Expected embeddings.npy to be [N, D], got {self.embs.shape}")

    def __len__(self) -> int:
        return self.embs.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = self.embs[idx]  # [D]
        return torch.from_numpy(x)
