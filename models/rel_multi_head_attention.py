import torch
import torch.nn as nn


class RelMultiHeadAttention:
    def __init__(self, num_heads, d_model, dropout=0.1):
        if d_model % num_heads != 0:
            raise ValueError(f"d_model {d_model} is not dividable by {num_heads}")

        self.d_model = d_model
        self.dropout = dropout
        self.num_heads = num_heads
        self.d_heads = d_model // num_heads

        self.q_net = nn.Linear(d_model, d_model, bias=False)
        self.k_net = nn.Linear(d_model, d_model, bias=False)
        self.v_net = nn.Linear(d_model, d_model, bias=False)

        self.pos_net = nn.Linear(d_model, d_model, bias=False)

        # Biases
        self.u = nn.Parameter(torch.Tensor(num_heads, self.d_head))
        self.v = nn.Parameter(torch.Tensor(num_heads, self.d_head))

        nn.init.xavier_uniform_(self.u)
        nn.init.xavier_uniform_(self.v)

        self.dropout = nn.Dropout(dropout)

        self.o_net = nn.Linear(d_model, d_model, bias=False)

    def forward(self, query, key, value, rel_positional_encoding, mask=None):
        self.q_net = query
        self.k_net = key
        self.v_net = value
        self.pos_net = rel_positional_encoding

        split_heads = lambda x: x.view(x.size(0), x.size(1), self.num_heads, self.d_heads).transpose(1, 2)

        # to be continued...

        pass

    def backward(self, grad_output):
        pass
