import torch
import torch.nn as nn
import math
from constants import DEFAULT_RECURRENCE_LENGTH
class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, recurrence_length=DEFAULT_RECURRENCE_LENGTH):
        super().__init__()
        self.d_model = d_model

        pe = torch.zeros(recurrence_length, d_model)

        position = torch.arange(0, recurrence_length, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        print(self.pe[:, :x.size(1)].detach().shape)
        return self.pe[:, :x.size(1)].detach()


if __name__ == "__main__":
     x=PositionalEmbedding(d_model=10)
     x.forward(torch.zeros(1,20,10))
