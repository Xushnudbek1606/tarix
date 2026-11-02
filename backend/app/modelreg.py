"""Model registry on disk: models/<version>/{model.onnx,meta.json} + models/current.json."""
from __future__ import annotations

import json
from pathlib import Path


def list_models(models_dir: Path) -> list[dict]:
    out = []
    for meta in sorted(models_dir.glob("*/meta.json")):
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        data["version"] = data.get("version", meta.parent.name)
        out.append(data)
    return out


def current_version(models_dir: Path) -> str | None:
    f = models_dir / "current.json"
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8")).get("version")
        except (OSError, json.JSONDecodeError):
            return None
    return None


def current_meta(models_dir: Path) -> dict | None:
    v = current_version(models_dir)
    if not v:
        return None
    meta = models_dir / v / "meta.json"
    if not meta.exists():
        return None
    return json.loads(meta.read_text(encoding="utf-8"))


def activate(models_dir: Path, version: str) -> None:
    if not (models_dir / version / "meta.json").exists():
        raise FileNotFoundError(version)
    (models_dir / "current.json").write_text(json.dumps({"version": version}), encoding="utf-8")


def public_model_info(meta: dict | None) -> dict | None:
    if not meta:
        return None
    m = meta.get("metrics", {})
    return {"version": meta.get("version"), "arch": meta.get("arch"), "synthetic": bool(meta.get("synthetic")),
            "num_classes": meta.get("num_classes"), "val_accuracy": m.get("val_accuracy"),
            "test_accuracy": m.get("test_accuracy"), "created_at": meta.get("created_at")}
