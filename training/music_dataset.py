import torch
from torch.utils.data import Dataset


class MusicDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.seq_len + 1]

        chunk_tensor = torch.tensor(chunk, dtype=torch.long)

        x = chunk_tensor[:-1]
        y = chunk_tensor[1:]

        return x, y
