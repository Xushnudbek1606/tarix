"""Dataset loading, speaker-aware splitting and augmentation.

Layout (see docs/DATASET.md):
    dataset/<sign_folder>/<speaker>_<id>.npz    landmark sequence, key 'sequence' (T, FRAME_DIM)   <- training input
    dataset/<sign_folder>/<speaker>_<id>.mp4    optional source video (convert with training/extract_landmarks.py)
    dataset/signs.json                          vocabulary registry
Folder name = sign name lower-cased (e.g. dataset/pasport/). Adding a sign = new folder + signs.json entry.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ai.features import FRAME_DIM, mirror_sequence, sequence_features

SYNTHETIC_DIR = "_synthetic"


@dataclass
class Sample:
    path: Path
    label: int
    sign: str
    speaker: str


def load_vocab(dataset_dir: Path) -> list[dict]:
    data = json.loads((dataset_dir / "signs.json").read_text(encoding="utf-8"))["signs"]
    return sorted(data, key=lambda s: s["id"])


def scan(dataset_dir: Path, vocab: list[dict], synthetic: bool = False) -> tuple[list[Sample], list[str]]:
    """Return (samples, class_names). Classes without any sample are dropped (they cannot be learned)."""
    root = dataset_dir / SYNTHETIC_DIR if synthetic else dataset_dir
    names, samples = [], []
    for s in vocab:
        folder = root / s["name"].lower()
        files = sorted(folder.glob("*.npz")) if folder.exists() else []
        if not files:
            continue
        label = len(names)
        names.append(s["name"])
        for f in files:
            speaker = f.stem.rsplit("_", 1)[0] or "unknown"
            samples.append(Sample(f, label, s["name"], speaker))
    return samples, names


def split(samples: list[Sample], seed: int = 42, val: float = 0.15, test: float = 0.15) -> dict[str, list[Sample]]:
    """Speaker-independent split when >= 4 speakers exist (honest accuracy), else stratified per class."""
    rng = np.random.default_rng(seed)
    speakers = sorted({s.speaker for s in samples})
    out = {"train": [], "val": [], "test": []}
    if len(speakers) >= 4:
        order = list(rng.permutation(speakers))
        n_test = max(1, round(len(order) * test))
        n_val = max(1, round(len(order) * val))
        test_sp, val_sp = set(order[:n_test]), set(order[n_test:n_test + n_val])
        for s in samples:
            out["test" if s.speaker in test_sp else "val" if s.speaker in val_sp else "train"].append(s)
        out["mode"] = "speaker-independent"  # type: ignore[assignment]
    else:
        by_label: dict[int, list[Sample]] = {}
        for s in samples:
            by_label.setdefault(s.label, []).append(s)
        for items in by_label.values():
            idx = rng.permutation(len(items))
            n = len(items)
            n_test = max(1, int(round(n * test))) if n >= 5 else (1 if n >= 3 else 0)
            n_val = max(1, int(round(n * val))) if n >= 5 else (1 if n >= 4 else 0)
            for k, i in enumerate(idx):
                bucket = "test" if k < n_test else "val" if k < n_test + n_val else "train"
                out[bucket].append(items[i])
        out["mode"] = "random-stratified (fewer than 4 speakers: accuracy is optimistic)"  # type: ignore[assignment]
    return out


def load_sequence(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        seq = z["sequence"].astype(np.float32)
    if seq.ndim != 2 or seq.shape[1] != FRAME_DIM:
        raise ValueError(f"{path}: expected (T,{FRAME_DIM}), got {seq.shape}")
    return seq


def trim_idle(seq: np.ndarray) -> np.ndarray:
    """Drop leading/trailing frames without any visible hand, matching what the live segmenter keeps."""
    vis = (seq[:, -3] > 0.5) | (seq[:, -2] > 0.5)         # left / right hand flags
    idx = np.flatnonzero(vis)
    return seq[idx[0]: idx[-1] + 1] if len(idx) >= 4 else seq


def augment(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    s = seq.copy()
    if rng.random() < 0.2:                                  # left-handed signer
        s = mirror_sequence(s)
    # speed change: 0.8x - 1.25x
    factor = rng.uniform(0.8, 1.25)
    n = max(8, int(len(s) * factor))
    src, dst = np.linspace(0, 1, len(s)), np.linspace(0, 1, n)
    s = np.stack([np.interp(dst, src, s[:, c]) for c in range(s.shape[1])], axis=1).astype(np.float32)
    s[:, -4:] = (s[:, -4:] > 0.5)
    body = slice(0, FRAME_DIM - 4)
    s[:, body] *= rng.uniform(0.92, 1.08)                   # distance to camera
    s[:, body] += rng.normal(0, 0.006, size=s[:, body].shape).astype(np.float32)   # landmark jitter
    return s


class SignDataset(Dataset):
    def __init__(self, samples: list[Sample], train: bool, seed: int = 0) -> None:
        self.samples, self.train = samples, train
        self.rng = np.random.default_rng(seed)
        self.cache = [trim_idle(load_sequence(s.path)) for s in samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        seq = self.cache[i]
        if self.train:
            seq = augment(seq, self.rng)
        return torch.from_numpy(sequence_features(seq)), self.samples[i].label
