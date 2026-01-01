import torch.nn as nn
from models.rel_multi_head_attention import RelMultiHeadAttention


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = RelMultiHeadAttention(num_heads, d_model, dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos_emb, mem=None, mask=None):

        residual = x
        x_norm = self.norm1(x)

        attn_output = self.self_attn(x_norm, pos_emb, mem, mask)
        x = residual + self.dropout(attn_output)

        residual = x
        x_norm = self.norm2(x)

        ff_output = self.feed_forward(x_norm)
        x = residual + self.dropout(ff_output)

        return x
