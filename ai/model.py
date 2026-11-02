"""Temporal sign-classification networks (PyTorch). Input: (B, SEQ_LEN, FEAT_DIM) landmark features.

Three interchangeable architectures, selected by name, all exportable to ONNX:
  transformer : small Transformer encoder with learned positional embedding (default)
  gru         : 2-layer bidirectional GRU
  tcn         : stack of dilated 1-D convolutions (temporal convolutional network)
"""
from __future__ import annotations

import torch
from torch import nn

from .features import FEAT_DIM, SEQ_LEN

ARCHS = ("transformer", "gru", "tcn")


class TransformerNet(nn.Module):
    def __init__(self, num_classes: int, in_dim: int = FEAT_DIM, d_model: int = 128, heads: int = 4,
                 layers: int = 3, dropout: float = 0.2, seq_len: int = SEQ_LEN):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d_model), nn.GELU(), nn.Dropout(dropout))
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=d_model * 2, dropout=dropout,
                                           batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model * 2, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(self.proj(x) + self.pos)
        h = self.norm(h)
        pooled = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=-1)
        return self.head(pooled)


class GRUNet(nn.Module):
    def __init__(self, num_classes: int, in_dim: int = FEAT_DIM, hidden: int = 128, dropout: float = 0.3, **_):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU())
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 4, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(self.proj(x))
        return self.head(torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=-1))


class _TCNBlock(nn.Module):
    def __init__(self, ch: int, dilation: int, dropout: float):
        super().__init__()
        pad = dilation
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=pad, dilation=dilation), nn.BatchNorm1d(ch), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(ch, ch, 3, padding=pad, dilation=dilation), nn.BatchNorm1d(ch))
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class TCNNet(nn.Module):
    def __init__(self, num_classes: int, in_dim: int = FEAT_DIM, ch: int = 128, dropout: float = 0.2, **_):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.inp = nn.Conv1d(in_dim, ch, 1)
        self.blocks = nn.Sequential(*[_TCNBlock(ch, d, dropout) for d in (1, 2, 4, 8)])
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(ch * 2, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.inp(self.norm(x).transpose(1, 2)))
        return self.head(torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=-1))


def build_model(arch: str, num_classes: int, **kw) -> nn.Module:
    if arch == "transformer":
        return TransformerNet(num_classes, **kw)
    if arch == "gru":
        return GRUNet(num_classes, **kw)
    if arch == "tcn":
        return TCNNet(num_classes, **kw)
    raise ValueError(f"unknown architecture '{arch}' (choose from {ARCHS})")


def export_onnx(model: nn.Module, path: str, seq_len: int = SEQ_LEN, in_dim: int = FEAT_DIM) -> None:
    model.eval()
    dummy = torch.zeros(1, seq_len, in_dim)
    torch.onnx.export(model, dummy, path, input_names=["sequence"], output_names=["logits"],
                      dynamic_axes={"sequence": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)
