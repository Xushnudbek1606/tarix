"""Train / retrain the sign model.

    python -m training.train --epochs 60 --arch transformer
    python -m training.train --allow-synthetic --epochs 30      # pipeline test only

Produces models/<version>/{model.onnx, model.pt, meta.json, confusion_matrix.png} and, if the
validation accuracy is acceptable, makes it the active model (models/current.json).
Versions are immutable; roll back by activating an older version in the admin panel.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.features import FEAT_DIM, SEQ_LEN            # noqa: E402
from ai.model import ARCHS, build_model, export_onnx  # noqa: E402
from training.dataset import SignDataset, load_vocab, scan, split  # noqa: E402

MIN_REAL_PER_CLASS = 5


def evaluate(model: nn.Module, loader: DataLoader, num_classes: int, device: str) -> dict:
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu().numpy()
            for t, p in zip(y.numpy(), pred):
                cm[t, p] += 1
    total = int(cm.sum())
    acc = float(np.trace(cm) / total) if total else 0.0
    per_class = {}
    for i in range(num_classes):
        tp, fp, fn = cm[i, i], cm[:, i].sum() - cm[i, i], cm[i, :].sum() - cm[i, i]
        per_class[i] = {"precision": float(tp / (tp + fp)) if tp + fp else 0.0,
                        "recall": float(tp / (tp + fn)) if tp + fn else 0.0, "support": int(cm[i].sum())}
    return {"accuracy": acc, "confusion": cm, "per_class": per_class, "n": total}


class Calibrated(nn.Module):
    """Wraps a model and divides the logits by a fitted temperature (confidence calibration)."""
    def __init__(self, model: nn.Module, temperature: float) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("t", torch.tensor(float(temperature)))

    def forward(self, x):
        return self.model(x) / self.t


def fit_temperature(model: nn.Module, loader: DataLoader, device: str) -> float:
    """Temperature scaling on the validation set so that reported confidence matches real accuracy."""
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits.append(model(x.to(device)).cpu()); labels.append(y)
    z, y = torch.cat(logits), torch.cat(labels)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=80)

    def closure():
        opt.zero_grad()
        loss = nn.functional.cross_entropy(z / log_t.exp().clamp(0.05, 5.0), y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(log_t.detach().exp().clamp(0.05, 5.0))


def save_confusion_png(cm: np.ndarray, names: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.55), max(5, len(names) * 0.5)))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=70, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Confusion matrix (test set, row-normalised)")
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def next_version(models_dir: Path, synthetic: bool) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{'synthetic' if synthetic else 'v'}-{stamp}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(ROOT / "dataset"))
    ap.add_argument("--models", default=str(ROOT / "models"))
    ap.add_argument("--arch", default="transformer", choices=ARCHS)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-synthetic", action="store_true", help="train on dataset/_synthetic (pipeline test)")
    ap.add_argument("--version", default=None)
    ap.add_argument("--min-val-acc", type=float, default=0.85, help="auto-activate only above this")
    ap.add_argument("--no-activate", action="store_true")
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dataset_dir, models_dir = Path(a.dataset), Path(a.models)
    vocab = load_vocab(dataset_dir)

    samples, names = scan(dataset_dir, vocab, synthetic=False)
    synthetic = False
    counts = {n: sum(1 for s in samples if s.sign == n) for n in names}
    enough = [n for n, c in counts.items() if c >= MIN_REAL_PER_CLASS]
    if len(enough) < 2:
        if not a.allow_synthetic:
            print("ERROR: no real Uzbek Sign Language data found.\n"
                  f"  Need >= {MIN_REAL_PER_CLASS} recorded samples for at least 2 signs in {dataset_dir}.\n"
                  "  Record them in the admin panel (Training tab) or place videos and run\n"
                  "  python -m training.extract_landmarks. To only test the pipeline, add --allow-synthetic.")
            return 2
        samples, names = scan(dataset_dir, vocab, synthetic=True)
        synthetic = True
        if not samples:
            print("ERROR: run `python -m training.synthetic` first")
            return 2
        print("WARNING: training on SYNTHETIC data. This model is not Uzbek Sign Language.")
    else:                                              # drop signs with too few samples to learn
        keep = [n for n in names if n in enough]
        remap = {names.index(n): i for i, n in enumerate(keep)}
        samples = [type(s)(s.path, remap[s.label], s.sign, s.speaker) for s in samples if s.label in remap]
        dropped = [n for n in names if n not in keep]
        names = keep
        if dropped:
            print(f"note: skipping signs with < {MIN_REAL_PER_CLASS} samples: {', '.join(dropped)}")

    parts = split(samples, seed=a.seed)
    split_mode = parts.pop("mode")                     # type: ignore[arg-type]
    print(f"{len(samples)} samples, {len(names)} classes, split={split_mode}: " +
          ", ".join(f"{k}={len(v)}" for k, v in parts.items()))
    if not parts["train"] or not parts["val"] or not parts["test"]:
        print("ERROR: not enough samples to build train/val/test splits")
        return 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = SignDataset(parts["train"], train=True, seed=a.seed)
    val_ds, test_ds = SignDataset(parts["val"], False), SignDataset(parts["test"], False)
    mk = lambda ds, sh: DataLoader(ds, batch_size=a.batch, shuffle=sh, num_workers=0)
    train_dl, val_dl, test_dl = mk(train_ds, True), mk(val_ds, False), mk(test_ds, False)

    model = build_model(a.arch, len(names)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(train_dl))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    best, best_state, bad = -1.0, None, 0
    history = []
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        run = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item() * len(y)
        val = evaluate(model, val_dl, len(names), device)
        history.append({"epoch": ep, "loss": run / len(train_ds), "val_acc": val["accuracy"]})
        if val["accuracy"] > best:
            best, bad = val["accuracy"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ep == 1 or ep % 5 == 0 or ep == a.epochs:
            print(f"epoch {ep:3d}/{a.epochs}  loss {run / len(train_ds):.4f}  val_acc {val['accuracy']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        if bad >= 25:
            print("early stopping")
            break

    model.load_state_dict(best_state)
    model.to(device)
    val = evaluate(model, val_dl, len(names), device)
    test = evaluate(model, test_dl, len(names), device)
    print(f"best val acc {val['accuracy']:.3f} | TEST acc {test['accuracy']:.3f} (n={test['n']})")

    version = a.version or next_version(models_dir, synthetic)
    out = models_dir / version
    out.mkdir(parents=True, exist_ok=False)
    temperature = fit_temperature(model, val_dl, device)
    print(f"calibration temperature {temperature:.3f}")
    model.cpu()
    export_onnx(Calibrated(model, temperature), str(out / "model.onnx"))
    torch.save(best_state, out / "model.pt")
    save_confusion_png(test["confusion"], names, out / "confusion_matrix.png")

    meta = {
        "version": version, "arch": a.arch, "synthetic": synthetic, "classes": names, "num_classes": len(names),
        "temperature": temperature, "seq_len": SEQ_LEN, "feat_dim": FEAT_DIM, "created_at": datetime.now(timezone.utc).isoformat(),
        "split": {"mode": split_mode, **{k: len(v) for k, v in parts.items()}},
        "train_args": {k: v for k, v in vars(a).items() if k not in ("dataset", "models")},
        "metrics": {"val_accuracy": val["accuracy"], "test_accuracy": test["accuracy"], "test_samples": test["n"],
                    "per_class": {names[i]: m for i, m in test["per_class"].items()},
                    "confusion_matrix": test["confusion"].tolist()},
        "history": history,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if not a.no_activate and val["accuracy"] >= a.min_val_acc:
        (models_dir / "current.json").write_text(json.dumps({"version": version}), encoding="utf-8")
        print(f"activated model {version}")
    else:
        print(f"model {version} saved but NOT activated (val acc {val['accuracy']:.3f} < {a.min_val_acc} or --no-activate)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
