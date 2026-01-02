import torch
import torch.nn as nn

from models.positional_embedding import PositionalEmbedding
from models.decoder_layer import DecoderLayer


class MusicTransformerXL(nn.Module):
    def __init__(self, vocab_size, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.pos_emb = PositionalEmbedding(d_model)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, x, mems=None):

        if mems is None:
            mems = [None] * len(self.layers)

        word_emb = self.embedding(x)
        word_emb = self.dropout(word_emb)

        pos_emb = self.pos_emb(word_emb)

        q_len = x.size(1)
        k_len = x.size(1) + (mems[0].size(0) if mems[0] is not None else 0)

        attn_mask = torch.triu(torch.ones(q_len, k_len, device=x.device), diagonal=1 + (k_len - q_len)).bool()

        hidden_state = word_emb
        new_mems = []

        for i, layer in enumerate(self.layers):
            new_mems.append(hidden_state.detach())

            m_i = mems[i]

            valid_mask = ~attn_mask

            hidden_state = layer(hidden_state, pos_emb, mem=m_i, mask=valid_mask)

        hidden_state = self.norm(hidden_state)
        logits = self.output_head(hidden_state)

        return logits, new_mems
