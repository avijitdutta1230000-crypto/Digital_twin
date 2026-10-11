"""Forecasting and segmentation models.

Baselines : Persistence, LSTM, CNN-LSTM (DeepCrowd-style conv + recurrent)
Proposed  : MM-STFormer — graph spatial mixing + Transformer temporal attention
            + multimodal exogenous fusion (weather / traffic / calendar).
Flood     : small U-Net for SAR water segmentation (+ Grad-CAM in xai.py).
"""
import math
import torch
import torch.nn as nn


class Persistence(nn.Module):
    def __init__(self, n_nodes, n_out):
        super().__init__(); self.n, self.h = n_nodes, n_out
        self.dummy = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        return x[:, -1:, :self.n].repeat(1, self.h, 1)


class LSTMForecaster(nn.Module):
    def __init__(self, n_feat, n_nodes, n_out, hidden=128):
        super().__init__(); self.n, self.h = n_nodes, n_out
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, n_nodes * n_out)
    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(o[:, -1]).view(-1, self.h, self.n)


class CNNLSTM(nn.Module):
    """1-D temporal convolutions extract local patterns, LSTM models sequence."""
    def __init__(self, n_feat, n_nodes, n_out, hidden=128, ch=64):
        super().__init__(); self.n, self.h = n_nodes, n_out
        self.conv = nn.Sequential(nn.Conv1d(n_feat, ch, 3, padding=1), nn.ReLU(),
                                  nn.Conv1d(ch, ch, 3, padding=1), nn.ReLU())
        self.lstm = nn.LSTM(ch, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_nodes * n_out)
    def forward(self, x):
        z = self.conv(x.transpose(1, 2)).transpose(1, 2)
        o, _ = self.lstm(z)
        return self.head(o[:, -1]).view(-1, self.h, self.n)


class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MMSTFormer(nn.Module):
    """Multimodal Spatio-Temporal Transformer (proposed).

    x: (B, T, N + E) — first N columns are the target nodes (road sensors / zones),
    last E are exogenous signals (weather, cross-domain series, calendar).

    1. Temporal embedding: each variable's whole history (length T) is embedded
       into one d-dim token (captures long-range temporal shape in one step).
    2. Graph mixing: node tokens are enriched with their neighbours' history via
       the normalised adjacency A (spatial prior from the road network).
    3. Multimodal attention: node tokens and exogenous tokens (each weather /
       traffic / calendar series is its own token, tagged with a learned
       modality embedding) attend to each other — so the model learns *which*
       external signal matters for *which* node; a learned gate scales the
       cross-modal contribution.
    4. Per-node head predicts the n_out-step horizon, residual on the last value.
    """
    def __init__(self, n_nodes, n_exo, n_out, A=None, n_in=12, d=64, heads=4, layers=2):
        super().__init__(); self.n, self.h, self.e = n_nodes, n_out, n_exo
        self.register_buffer("A", torch.as_tensor(A) if A is not None else torch.eye(n_nodes))
        self.node_emb = nn.Linear(2 * n_in, d)
        self.exo_emb = nn.Linear(n_in, d)
        self.node_id = nn.Parameter(torch.randn(n_nodes, d) * 0.02)
        self.exo_id = nn.Parameter(torch.randn(max(n_exo, 1), d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.gate = nn.Parameter(torch.zeros(1))
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_out))

    def forward(self, x):
        xn = x[..., :self.n]                                     # (B, T, N)
        neigh = (xn @ self.A.T)                                   # graph neighbourhood mixing
        tok = self.node_emb(torch.cat([xn, neigh], 1).transpose(1, 2)) + self.node_id   # (B, N, d)
        if self.e:
            ex = self.exo_emb(x[..., self.n:].transpose(1, 2)) + self.exo_id           # (B, E, d)
            tok = torch.cat([tok, torch.sigmoid(self.gate + 2) * ex], 1)
        z = self.encoder(tok)[:, :self.n]
        return self.head(z).transpose(1, 2) + xn[:, -1:]         # (B, n_out, N)


class FloodUNet(nn.Module):
    def __init__(self, cin=2, c=16):
        super().__init__()
        blk = lambda a, b: nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(),
                                         nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU())
        self.e1, self.e2, self.b = blk(cin, c), blk(c, 2 * c), blk(2 * c, 4 * c)
        self.pool = nn.MaxPool2d(2)
        self.u2, self.d2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, 2), blk(4 * c, 2 * c)
        self.u1, self.d1 = nn.ConvTranspose2d(2 * c, c, 2, 2), blk(2 * c, c)
        self.out = nn.Conv2d(c, 1, 1)
    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.pool(e1)); b = self.b(self.pool(e2))
        d2 = self.d2(torch.cat([self.u2(b), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)
