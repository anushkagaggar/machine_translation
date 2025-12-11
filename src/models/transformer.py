import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # odd d_model -> last column stays zero in cos
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to tensor x: shape [B, S, D]"""
        length = x.size(1)
        return x + self.pe[:, :length, :x.size(2)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor):
        # x: [B, S, D] -> [B, nhead, S, d_k]
        B, S, D = x.size()
        x = x.view(B, S, self.nhead, self.d_k)
        return x.permute(0, 2, 1, 3)

    def _combine_heads(self, x: torch.Tensor):
        # x: [B, nhead, S, d_k] -> [B, S, D]
        x = x.permute(0, 2, 1, 3).contiguous()
        B, S, _, _ = x.size()
        return x.view(B, S, self.d_model)

    def forward(self, query, key, value, attn_mask: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None):
        # query,key,value: [B, S, D]
        B = query.size(0)

        q = self._split_heads(self.w_q(query))  # [B, nhead, S_q, d_k]
        k = self._split_heads(self.w_k(key))
        v = self._split_heads(self.w_v(value))

        # scaled dot-product
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, nhead, S_q, S_k]

        if attn_mask is not None:
            # attn_mask expected shape: [S_q, S_k] or [B*nhead, S_q, S_k]
            scores = scores + attn_mask.unsqueeze(0)

        if key_padding_mask is not None:
            # key_padding_mask: [B, S_k] where True indicates PAD
            # we want to set large negative where padded
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,S_k]
            scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # [B, nhead, S_q, d_k]

        out = self._combine_heads(out)
        out = self.w_o(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff, dropout=dropout)

        # Pre-Norm style
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        # Pre-Norm
        x2 = self.norm1(x)
        sa = self.self_attn(x2, x2, x2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        x = x + sa

        x2 = self.norm2(x)
        ff = self.ff(x2)
        x = x + ff
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Self-attn (with causal mask)
        x2 = self.norm1(x)
        sa = self.self_attn(x2, x2, x2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        x = x + sa

        # Cross-attn
        x2 = self.norm2(x)
        ca = self.cross_attn(x2, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        x = x + ca

        # FF
        x2 = self.norm3(x)
        ff = self.ff(x2)
        x = x + ff
        return x


class Encoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, nhead: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, d_ff, dropout=dropout) for _ in range(num_layers)
        ])

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, nhead: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, d_ff, dropout=dropout) for _ in range(num_layers)
        ])

    def forward(self, x, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        return x


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        d_model: int,
        nhead: int,
        d_ff: int,
        vocab_size: int,
        max_len: int = 512,
        dropout: float = 0.1,
        tie_embeddings: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.src_tok_emb = nn.Embedding(vocab_size, d_model)
        self.tgt_tok_emb = nn.Embedding(vocab_size, d_model)

        if tie_embeddings:
            # tie src and tgt embeddings
            self.tgt_tok_emb.weight = self.src_tok_emb.weight

        self.pos_encoder = PositionalEncoding(d_model, max_len)

        self.encoder = Encoder(num_encoder_layers, d_model, nhead, d_ff, dropout=dropout)
        self.decoder = Decoder(num_decoder_layers, d_model, nhead, d_ff, dropout=dropout)

        self.final_linear = nn.Linear(d_model, vocab_size)

    def forward(self, src_ids, tgt_input_ids, src_key_padding_mask=None, tgt_key_padding_mask=None, tgt_mask=None):
        # src_ids: [B, S_src], tgt_input_ids: [B, S_tgt]
        src_emb = self.src_tok_emb(src_ids) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_tok_emb(tgt_input_ids) * math.sqrt(self.d_model)

        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_encoder(tgt_emb)

        # Encoder: memory [B, S_src, D]
        memory = self.encoder(src_emb, src_mask=None, src_key_padding_mask=src_key_padding_mask)

        # Decoder: outputs [B, S_tgt, D]
        out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_mask=None, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=src_key_padding_mask)

        logits = self.final_linear(out)  # [B, S_tgt, vocab]
        return logits


# ----------------- Mask helpers -----------------

def generate_square_subsequent_mask(sz: int, device=None) -> torch.Tensor:
    """Generate causal mask for tgt: shape [sz, sz] with 0 for allowed, -inf for masked"""
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def make_padding_mask(pad_mask: torch.Tensor) -> torch.Tensor:
    """Convert pad bool mask [B, S] (True where PAD) -> key_padding_mask expected by attention: [B, S] True where PAD"""
    return pad_mask


# ----------------- Quick smoke-test -----------------
if __name__ == '__main__':
    # quick forward pass sanity check
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    B = 8
    S_src = 20
    S_tgt = 22
    vocab_size = 32000
    d_model = 256
    nhead = 8
    d_ff = 1024

    model = Seq2SeqTransformer(
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_model=d_model,
        nhead=nhead,
        d_ff=d_ff,
        vocab_size=vocab_size,
        max_len=512,
        tie_embeddings=True,
    ).to(device)

    src = torch.randint(3, vocab_size, (B, S_src), dtype=torch.long, device=device)
    tgt_in = torch.randint(3, vocab_size, (B, S_tgt), dtype=torch.long, device=device)

    # Simulate padding masks (False where real token)
    src_pad_mask = torch.zeros((B, S_src), dtype=torch.bool, device=device)
    tgt_pad_mask = torch.zeros((B, S_tgt), dtype=torch.bool, device=device)

    tgt_mask = generate_square_subsequent_mask(S_tgt, device=device)

    logits = model(src, tgt_in, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask, tgt_mask=tgt_mask)
    print('Logits shape:', logits.shape)  # expect [B, S_tgt, vocab_size]
