"""Real-time sign recognizer: streaming frames -> segmented signs -> sentence.

Flow per frame (push):  hands visible & moving  -> collect frames into a segment
                        hands gone / resting     -> close the segment, classify it once
                        long pause after signs   -> build the sentence from collected glosses

The ONNX model is loaded once and kept in memory. Frames are never classified one by one:
each segment is a whole temporal sequence (resampled to SEQ_LEN) so movement is understood.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .features import F_LH, F_RH, SLICES, sequence_features
from .nlp import SentenceBuilder

log = logging.getLogger("govsign.recognizer")


class SignClassifier:
    """ONNX Runtime wrapper. Hot-swappable: `load()` can be called again for a new model version."""

    def __init__(self) -> None:
        self.session = None
        self.meta: dict = {}
        self.classes: list[str] = []
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self.session is not None

    @property
    def version(self) -> str:
        return self.meta.get("version", "")

    def load(self, models_dir: Path) -> bool:
        import onnxruntime as ort
        cur = models_dir / "current.json"
        if not cur.exists():
            return False
        version = json.loads(cur.read_text(encoding="utf-8"))["version"]
        folder = models_dir / version
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        providers = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        session = ort.InferenceSession(str(folder / "model.onnx"), sess_options=opts, providers=providers)
        with self._lock:
            self.session, self.meta, self.classes = session, meta, list(meta["classes"])
        log.info("Loaded model %s (%d classes, %s)", version, len(self.classes), providers[0])
        return True

    def predict(self, feats: np.ndarray) -> tuple[str, float, list[tuple[str, float]]]:
        with self._lock:
            session, classes = self.session, self.classes
        logits = session.run(None, {"sequence": feats[None].astype(np.float32)})[0][0]
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        order = np.argsort(probs)[::-1][:3]
        top = [(classes[i], float(probs[i])) for i in order]
        return top[0][0], top[0][1], top


@dataclass
class RecognizerConfig:
    threshold: float = 0.80            # minimum confidence to accept a sign
    min_segment: int = 8               # frames of signing needed to count as a sign
    max_segment: int = 120             # safety cap (~4 s at 30 fps)
    end_gap: int = 9                   # consecutive hand-free frames that close a sign (~0.3 s)
    sentence_gap_s: float = 2.0        # pause that closes the sentence
    motion_eps: float = 0.004          # mean per-frame hand displacement considered "moving"


@dataclass
class Recognizer:
    classifier: SignClassifier
    builder: SentenceBuilder
    config: RecognizerConfig = field(default_factory=RecognizerConfig)

    def __post_init__(self) -> None:
        self._seg: list[np.ndarray] = []
        self._gap = 0
        self._items: list[dict] = []            # accepted glosses in the current sentence
        self._rejected = 0
        self._last_sign_t = 0.0
        self.state = "idle"                     # idle | signing | processing

    def reset(self) -> None:
        self._seg, self._gap, self._items, self._rejected, self.state = [], 0, [], 0, "idle"

    @staticmethod
    def _hands_visible(vec: np.ndarray) -> bool:
        return bool(vec[F_LH] > 0.5 or vec[F_RH] > 0.5)

    def push(self, vec: np.ndarray, now: float | None = None) -> list[dict]:
        """Feed one frame vector; returns a (usually empty) list of events."""
        now = time.monotonic() if now is None else now
        events: list[dict] = []
        cfg = self.config
        visible = self._hands_visible(vec)

        if visible:
            self._gap = 0
            self._seg.append(vec)
            if len(self._seg) == cfg.min_segment:
                self.state = "signing"
                events.append({"type": "state", "state": "signing"})
            if len(self._seg) >= cfg.max_segment:
                events += self._close_segment(now)
        else:
            if self._seg:
                self._gap += 1
                if self._gap >= cfg.end_gap:
                    events += self._close_segment(now)
            elif self._items or self._rejected:
                if now - self._last_sign_t >= cfg.sentence_gap_s:
                    events += self._finish_sentence()
        return events

    # ------------------------------------------------------------------ internals
    def _close_segment(self, now: float) -> list[dict]:
        seg = np.stack(self._seg)
        self._seg, self._gap = [], 0
        self.state = "idle"
        if len(seg) < self.config.min_segment or not self.classifier.loaded:
            return []
        t0 = time.perf_counter()
        sign, conf, top = self.classifier.predict(sequence_features(seg))
        ms = (time.perf_counter() - t0) * 1000
        accepted = conf >= self.config.threshold
        self._last_sign_t = now
        ev = {"type": "recognition", "sign": sign, "confidence": conf, "top": top, "accepted": accepted,
              "latency_ms": round(ms, 1)}
        if accepted:
            self._items.append({"sign": sign, "confidence": conf})
        else:
            self._rejected += 1
        return [ev]

    def _finish_sentence(self) -> list[dict]:
        items, rejected = self._items, self._rejected
        self._items, self._rejected = [], 0
        if rejected and not items:
            return [{"type": "uncertain", "confidence": 0.0}]
        if not items:
            return []
        if rejected:                         # part of the sentence was uncertain: do not guess the rest
            return [{"type": "uncertain", "confidence": min(i["confidence"] for i in items)}]
        text = self.builder.build([i["sign"] for i in items])
        return [{"type": "sentence", "text": text, "items": items,
                 "confidence": min(i["confidence"] for i in items)}]

    def flush(self) -> list[dict]:
        """Force-close whatever is pending (used by the 'send now' button and tests)."""
        ev = self._close_segment(time.monotonic()) if self._seg else []
        return ev + self._finish_sentence()
