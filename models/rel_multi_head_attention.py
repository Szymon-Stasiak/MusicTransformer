import torch
import torch.nn as nn


class RelMultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        super().__init__()
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
        self.u = nn.Parameter(torch.Tensor(num_heads, self.d_heads))
        self.v = nn.Parameter(torch.Tensor(num_heads, self.d_heads))

        nn.init.xavier_uniform_(self.u)
        nn.init.xavier_uniform_(self.v)

        self.dropout = nn.Dropout(dropout)

        self.o_net = nn.Linear(d_model, d_model, bias=False)

    def rel_shift(self, x):
        batch_size, num_heads, q_len, k_len = x.size()
        zero_pad = torch.zeros((batch_size, num_heads, q_len, 1), device=x.device, dtype=x.dtype)
        x = torch.cat([zero_pad, x], dim=-1)

        x = x.view(batch_size, num_heads, k_len + 1, q_len)
        x = x[:, :, 1:].view(batch_size, num_heads, q_len, k_len)
        return x

    def forward(self, x, pos_emb, mem=None, mask=None):
        if mem is not None:
            cat = torch.cat([mem, x], dim=1)
        else:
            cat = x

        q = self.q_net(x)
        k = self.k_net(cat)
        v = self.v_net(cat)
        p = self.pos_net(pos_emb)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        p = p.view(-1, self.num_heads, self.d_heads).transpose(0, 1)

        attention_content = torch.einsum('bhqd,bhkd->bhqk', q + self.u.unsqueeze(0).unsqueeze(2), k)
        attention_position = torch.einsum('bhqd,hkd->bhqk', q + self.v.unsqueeze(0).unsqueeze(2), p)
        attention_position = self.rel_shift(attention_position)

        attn_score = (attention_content + attention_position) / (self.d_heads ** 0.5)

        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, float('-inf'))

        attn_prob = torch.softmax(attn_score, dim=-1)
        attn_prob = self.dropout(attn_prob)

        attn_vec = torch.einsum('bhqk,bhkd->bhqd', attn_prob, v)

        attn_vec = attn_vec.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.d_model)

        output = self.o_net(attn_vec)
        return output
    def split_heads(self, t):
        B, T, D = t.size()
        return t.view(B, T, self.num_heads, self.d_heads).transpose(1, 2)
