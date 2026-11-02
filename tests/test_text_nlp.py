import json
from pathlib import Path

import numpy as np

from ai.features import FEAT_DIM, FRAME_DIM, SEQ_LEN, mirror_sequence, resample, sequence_features
from ai.nlp import SentenceBuilder
from common.uzbek_text import (canonical_text, cyrillic_to_latin, detect_script, is_probably_uzbek,
                               latin_to_cyrillic, normalize_latin, to_script)

ROOT = Path(__file__).resolve().parents[1]
SIGNS = json.loads((ROOT / "dataset" / "signs.json").read_text(encoding="utf-8"))["signs"]


def test_every_sign_cyrillic_matches_converter():
    for s in SIGNS:
        assert latin_to_cyrillic(s["uz_latin"]) == s["uz_cyrillic"], s["name"]
        assert cyrillic_to_latin(s["uz_cyrillic"]) == normalize_latin(s["uz_latin"]), s["name"]


def test_sentence_transliteration():
    assert latin_to_cyrillic("Men pasport olish uchun keldim.") == "Мен паспорт олиш учун келдим."
    assert latin_to_cyrillic("Pasportingizni kameraga ko'rsating.") == "Паспортингизни камерага кўрсатинг."
    assert to_script("Рахмат", "latin").lower() == "raxmat"


def test_apostrophes_normalised():
    assert normalize_latin("ko'rsating") == normalize_latin("ko‘rsating") == normalize_latin("koʻrsating")
    assert canonical_text("tug'ilganlik") == "tug‘ilganlik"


def test_script_detection_and_russian_passthrough():
    assert detect_script("салом") == "cyrillic" and detect_script("salom") == "latin"
    assert not is_probably_uzbek("Ожидайте, ещё немного")        # contains щ, a Russian-only letter


def test_sentence_builder_templates():
    b = SentenceBuilder(ROOT / "dataset" / "signs.json")
    assert b.build(["MEN", "PASPORT", "OLISH", "KELDIM"]) == "Men pasport olish uchun keldim."
    assert b.build(["MEN", "MEN", "PASPORT", "OLISH", "KELDIM"]) == "Men pasport olish uchun keldim."   # duplicates collapsed
    assert b.build(["TUGILGANLIK", "GUVOHNOMA", "OLISH", "KELDIM"]) == "Men tug‘ilganlik guvohnomasini olish uchun keldim."
    assert b.build(["SALOM", "YORDAM"]).startswith("Salom, ")
    assert b.build([]) == ""
    assert b.build(["ARIZA", "HUJJAT"]) == "Ariza, hujjat."          # literal fallback, nothing invented


def test_feature_shapes_and_mirror():
    seq = np.random.default_rng(0).normal(size=(17, FRAME_DIM)).astype("float32")
    seq[:, -4:] = 1.0
    assert resample(seq).shape == (SEQ_LEN, FRAME_DIM)
    assert sequence_features(seq).shape == (SEQ_LEN, FEAT_DIM)
    assert np.allclose(mirror_sequence(mirror_sequence(seq)), seq, atol=1e-6)
