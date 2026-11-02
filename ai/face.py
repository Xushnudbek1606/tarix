"""Face ID: embedding extraction + liveness. Everything runs locally; no image ever leaves the desk.

Embedders (pluggable, chosen automatically):
  onnx-face    : ArcFace / MobileFaceNet style ONNX model at models/face/face_embedder.onnx
                 (112x112 aligned RGB input, 128/512-d output). RECOMMENDED for production.
  geometry-v1  : fallback built from MediaPipe face-mesh geometry. It works for a demo with
                 a few enrolled people but is NOT a production-grade biometric.
Only the embedding (never the image) is sent to the backend, which stores it encrypted.

Liveness (active challenge): the person must blink and turn the head slightly while the
face is tracked. A static photo cannot do either. A pre-recorded video replay is harder to
rule out with a plain webcam; for high-security desks add an IR/depth camera (see docs/SECURITY.md).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("govsign.face")

# MediaPipe face-mesh indices
_L_EYE, _R_EYE = 468, 473                   # iris centres (face landmarker has 478 points)
_L_EYE_C, _R_EYE_C = (33, 133), (362, 263)  # fallback eye corners
_NOSE, _MOUTH_L, _MOUTH_R = 1, 61, 291
_GEOM_IDX = sorted(set(range(0, 468, 4)) | {1, 33, 133, 362, 263, 61, 291, 152, 10, 234, 454})


def _eye_centres(face: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(face) > 473:
        return face[_L_EYE, :2], face[_R_EYE, :2]
    return face[list(_L_EYE_C), :2].mean(0), face[list(_R_EYE_C), :2].mean(0)


class GeometryEmbedder:
    name = "geometry-v1"
    dim = len(_GEOM_IDX) * 3

    def embed(self, frame_bgr: np.ndarray, face: np.ndarray) -> np.ndarray:
        f = face[:, :3].astype(np.float32).copy()
        a, b = _eye_centres(face)
        centre = (a + b) / 2
        iod = float(np.linalg.norm(b - a)) + 1e-6
        ang = np.arctan2(b[1] - a[1], b[0] - a[0])
        c, s = np.cos(-ang), np.sin(-ang)
        xy = f[:, :2] - centre
        rot = np.stack([xy[:, 0] * c - xy[:, 1] * s, xy[:, 0] * s + xy[:, 1] * c], axis=1) / iod
        pts = np.concatenate([rot, f[:, 2:3] / iod], axis=1)[_GEOM_IDX]
        v = (pts - pts.mean(0)).reshape(-1)
        return v / (np.linalg.norm(v) + 1e-8)


class OnnxEmbedder:
    name = "onnx-face"

    def __init__(self, path: Path) -> None:
        import onnxruntime as ort
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        self.dim = int(self.session.get_outputs()[0].shape[-1])
        # canonical ArcFace 5-point template (112x112)
        self._dst = np.array([[38.29, 51.69], [73.53, 51.50], [56.03, 71.74],
                              [41.55, 92.37], [70.73, 92.20]], dtype=np.float32)

    def embed(self, frame_bgr: np.ndarray, face: np.ndarray) -> np.ndarray:
        import cv2
        h, w = frame_bgr.shape[:2]
        a, b = _eye_centres(face)
        src = np.array([a, b, face[_NOSE, :2], face[_MOUTH_L, :2], face[_MOUTH_R, :2]], dtype=np.float32) * [w, h]
        # the signer's left eye is on image right for a non-mirrored frame; order by x
        if src[0, 0] > src[1, 0]:
            src[[0, 1]] = src[[1, 0]]
            src[[3, 4]] = src[[4, 3]]
        m, _ = cv2.estimateAffinePartial2D(src, self._dst, method=cv2.LMEDS)
        aligned = cv2.warpAffine(frame_bgr, m, (112, 112), borderValue=0)
        x = (cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32) - 127.5) / 127.5
        out = self.session.run(None, {self.input: x.transpose(2, 0, 1)[None]})[0][0]
        return (out / (np.linalg.norm(out) + 1e-8)).astype(np.float32)


def load_embedder(models_dir: Path):
    path = models_dir / "face" / "face_embedder.onnx"
    if path.exists():
        try:
            return OnnxEmbedder(path)
        except Exception:
            log.exception("face_embedder.onnx could not be loaded; using geometry fallback")
    log.warning("Using the geometry-v1 face embedder (demo grade). See docs/SECURITY.md for production use.")
    return GeometryEmbedder()


class Liveness:
    """Blink + head-turn challenge over a short time window."""

    def __init__(self, window_s: float = 6.0, turn_threshold: float = 0.06) -> None:
        self.window_s = window_s
        self.turn_threshold = turn_threshold
        self.reset()

    def reset(self) -> None:
        self.t0 = None
        self.blinked = False
        self._eyes_open_seen = False
        self._yaw_min, self._yaw_max = 1.0, -1.0

    @staticmethod
    def yaw(face: np.ndarray) -> float:
        """Nose position between the cheeks: ~0 when frontal, +/- when the head turns."""
        l, r, n = face[234, 0], face[454, 0], face[_NOSE, 0]
        return float((n - (l + r) / 2) / (abs(r - l) + 1e-6))

    def update(self, face: np.ndarray | None, blend: dict | None, now: float | None = None) -> dict:
        now = time.monotonic() if now is None else now
        if face is None:
            self.reset()
            return {"passed": False, "blink": False, "turn": False, "challenge": "face"}
        if self.t0 is None or now - self.t0 > self.window_s:
            self.reset()
            self.t0 = now
        if blend:
            closed = max(blend.get("eyeBlinkLeft", 0.0), blend.get("eyeBlinkRight", 0.0))
            if closed < 0.25:
                self._eyes_open_seen = True
            elif closed > 0.55 and self._eyes_open_seen:
                self.blinked = True
        y = self.yaw(face)
        self._yaw_min, self._yaw_max = min(self._yaw_min, y), max(self._yaw_max, y)
        turned = (self._yaw_max - self._yaw_min) >= self.turn_threshold
        challenge = "blink" if not self.blinked else ("turn" if not turned else "done")
        return {"passed": self.blinked and turned, "blink": self.blinked, "turn": turned, "challenge": challenge}
