import json

import numpy as np
import pytest

from tests.conftest import ROOT


@pytest.mark.slow
def test_training_pipeline_end_to_end(tmp_path):
    """synthetic data -> train -> ONNX -> metrics -> activation -> inference (pipeline test only)."""
    from training.synthetic import generate
    from training.train import main
    ds = tmp_path / "dataset"; ds.mkdir()
    (ds / "signs.json").write_text((ROOT / "dataset" / "signs.json").read_text(encoding="utf-8"), encoding="utf-8")
    generate(ds, per_sign=10, speakers=5)
    models = tmp_path / "models"
    assert main(["--dataset", str(ds), "--models", str(models), "--allow-synthetic", "--epochs", "6", "--version", "t1"]) == 0
    meta = json.loads((models / "t1" / "meta.json").read_text())
    assert meta["synthetic"] is True and meta["num_classes"] == 17 and meta["split"]["mode"] == "speaker-independent"
    assert (models / "t1" / "model.onnx").exists() and (models / "t1" / "confusion_matrix.png").exists()
    assert 0 <= meta["metrics"]["test_accuracy"] <= 1 and np.array(meta["metrics"]["confusion_matrix"]).shape == (17, 17)

    from ai.features import FEAT_DIM, SEQ_LEN
    from ai.recognizer import SignClassifier
    if (models / "current.json").exists():
        c = SignClassifier()
        assert c.load(models)
        sign, conf, top = c.predict(np.zeros((SEQ_LEN, FEAT_DIM), np.float32))
        assert sign in c.classes and 0 <= conf <= 1 and len(top) == 3


def test_training_refuses_without_real_data(tmp_path, capsys):
    from training.train import main
    ds = tmp_path / "dataset"; ds.mkdir()
    (ds / "signs.json").write_text((ROOT / "dataset" / "signs.json").read_text(encoding="utf-8"), encoding="utf-8")
    assert main(["--dataset", str(ds), "--models", str(tmp_path / "m")]) == 2
    assert "no real Uzbek Sign Language data" in capsys.readouterr().out


def test_split_is_speaker_independent(tmp_path):
    from pathlib import Path
    from training.dataset import Sample, split
    samples = [Sample(Path(f"{s}_{i}.npz"), i % 3, "X", f"spk{s}") for s in range(8) for i in range(6)]
    parts = split(samples)
    assert parts.pop("mode") == "speaker-independent"
    spk = {k: {s.speaker for s in v} for k, v in parts.items()}
    assert not (spk["train"] & spk["test"]) and not (spk["train"] & spk["val"]) and not (spk["val"] & spk["test"])
