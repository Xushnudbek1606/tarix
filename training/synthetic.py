"""Synthetic landmark sequences - ONLY for testing the pipeline end to end.

Each sign gets a deterministic pseudo-trajectory + hand shape. This is NOT sign language:
a model trained on it proves that data -> training -> ONNX -> recognition -> UI works, nothing more.
Models trained on it are flagged `synthetic: true` and shown as such in the admin panel.

    python -m training.synthetic --per-sign 40 --speakers 6
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from ai.features import F_FACE, F_LH, F_POSE, F_RH, FRAME_DIM, SLICES
from .dataset import SYNTHETIC_DIR, load_vocab

ROOT = Path(__file__).resolve().parents[1]


def _templates(sign_id: int) -> dict:
    r = np.random.default_rng(10_000 + sign_id)
    return {"shape": r.normal(0, 0.6, (21, 3)).astype(np.float32),
            "start": r.uniform(-1.2, 1.2, 3).astype(np.float32), "end": r.uniform(-1.2, 1.2, 3).astype(np.float32),
            "loop": r.uniform(0, 0.5), "two": r.random() < 0.5, "freq": r.uniform(1, 3)}


def make_sequence(sign_id: int, speaker: int, rng: np.random.Generator) -> np.ndarray:
    t = _templates(sign_id)
    n = int(rng.integers(22, 46))
    seq = np.zeros((n, FRAME_DIM), np.float32)
    spk = np.random.default_rng(777 + speaker)
    offset, hand_scale = spk.normal(0, 0.08, 3), spk.uniform(0.9, 1.1)
    for i in range(n):
        u = i / (n - 1)
        centre = t["start"] * (1 - u) + t["end"] * u + offset
        centre = centre + t["loop"] * np.array([np.sin(2 * np.pi * t["freq"] * u), np.cos(2 * np.pi * t["freq"] * u), 0])
        hand = (t["shape"] * hand_scale * (0.8 + 0.4 * u) + centre) + rng.normal(0, 0.01, (21, 3))
        shape = (t["shape"] * (0.8 + 0.4 * u)) + rng.normal(0, 0.02, (21, 3))
        seq[i, SLICES["rh"]] = hand.reshape(-1)
        seq[i, SLICES["rh_shape"]] = shape.reshape(-1)
        seq[i, F_RH] = 1.0
        if t["two"]:
            seq[i, SLICES["lh"]] = (hand * [-1, 1, 1]).reshape(-1)
            seq[i, SLICES["lh_shape"]] = (shape * [-1, 1, 1]).reshape(-1)
            seq[i, F_LH] = 1.0
        seq[i, SLICES["pose"]] = rng.normal(0, 0.02, 27)
        seq[i, F_POSE] = seq[i, F_FACE] = 1.0
    return seq


def generate(dataset_dir: Path, per_sign: int = 40, speakers: int = 6, seed: int = 1) -> Path:
    out = dataset_dir / SYNTHETIC_DIR
    if out.exists():
        shutil.rmtree(out)
    rng = np.random.default_rng(seed)
    for s in load_vocab(dataset_dir):
        folder = out / s["name"].lower()
        folder.mkdir(parents=True)
        for k in range(per_sign):
            spk = k % speakers
            np.savez_compressed(folder / f"spk{spk}_{k:03d}.npz", sequence=make_sequence(s["id"], spk, rng))
    (out / "README.txt").write_text("SYNTHETIC test data - not Uzbek Sign Language. Safe to delete.\n")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-sign", type=int, default=40)
    ap.add_argument("--speakers", type=int, default=6)
    a = ap.parse_args()
    print("wrote", generate(ROOT / "dataset", a.per_sign, a.speakers))
