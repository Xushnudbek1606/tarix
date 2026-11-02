from pathlib import Path

import numpy as np

from ai.features import F_RH, FRAME_DIM
from ai.nlp import SentenceBuilder
from ai.recognizer import Recognizer, RecognizerConfig

ROOT = Path(__file__).resolve().parents[1]


class Stub:
    """Classifier stand-in: returns a scripted sequence of (sign, confidence)."""
    loaded = True
    version = "stub"

    def __init__(self, script):
        self.script, self.calls = list(script), 0

    def predict(self, feats):
        self.calls += 1
        s, c = self.script.pop(0)
        return s, c, [(s, c)]


def hand_frame():
    v = np.zeros(FRAME_DIM, np.float32)
    v[F_RH] = 1.0
    return v


def run(rec, signs, gap_frames=12, tail=90):
    events, t = [], 0.0
    for _ in signs:
        for _ in range(20):
            events += rec.push(hand_frame(), t); t += 0.033
        for _ in range(gap_frames):
            events += rec.push(np.zeros(FRAME_DIM, np.float32), t); t += 0.033
    for _ in range(tail):
        events += rec.push(np.zeros(FRAME_DIM, np.float32), t); t += 0.033
    return events


def make(script, threshold=0.8):
    return Recognizer(Stub(script), SentenceBuilder(ROOT / "dataset" / "signs.json"), RecognizerConfig(threshold=threshold))


def test_sentence_from_signs():
    ev = run(make([("MEN", .95), ("PASPORT", .93), ("OLISH", .9), ("KELDIM", .97)]), range(4))
    sent = [e for e in ev if e["type"] == "sentence"]
    assert len(sent) == 1 and sent[0]["text"] == "Men pasport olish uchun keldim."
    assert abs(sent[0]["confidence"] - .9) < 1e-9


def test_low_confidence_is_never_turned_into_a_statement():
    ev = run(make([("MEN", .95), ("PASPORT", .40)]), range(2))
    assert not [e for e in ev if e["type"] == "sentence"]
    assert [e for e in ev if e["type"] == "uncertain"]


def test_pause_inside_a_sign_does_not_split_it():
    rec = make([("PASPORT", .9)])
    ev = run(rec, range(1), gap_frames=3)               # shorter than end_gap
    assert rec.classifier.calls == 1


def test_no_hands_no_events():
    rec = make([])
    assert run(rec, [], tail=200) == []
