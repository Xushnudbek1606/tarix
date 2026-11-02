"""MediaPipe landmark extraction (Tasks API: Hand + Pose + Face landmarkers).

The three .task model files live in models/mediapipe/ (see scripts/download_models.py).
`extract()` returns one FRAME_DIM vector per camera frame, or None if models are missing.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .features import frame_vector

log = logging.getLogger("govsign.landmarks")

MODEL_FILES = {
    "hand": "hand_landmarker.task",
    "pose": "pose_landmarker_lite.task",
    "face": "face_landmarker.task",
}
MODEL_URLS = {
    "hand": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "pose": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "face": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
}


def models_available(model_dir: Path) -> bool:
    return all((model_dir / f).exists() for f in MODEL_FILES.values())


class LandmarkExtractor:
    """Video-mode landmarkers keep tracking state between frames (faster, steadier)."""

    def __init__(self, model_dir: Path, use_face: bool = True) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self.mp = mp
        self.use_face = use_face
        self._ts = 0
        base = lambda key: BaseOptions(model_asset_path=str(model_dir / MODEL_FILES[key]))
        self.hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=base("hand"), running_mode=vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=0.5, min_tracking_confidence=0.5))
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=base("pose"), running_mode=vision.RunningMode.VIDEO))
        self.face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=base("face"), running_mode=vision.RunningMode.VIDEO, num_faces=1,
            output_face_blendshapes=True)) if use_face else None

    @staticmethod
    def _arr(lms) -> np.ndarray:
        return np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)

    def extract(self, frame_bgr: np.ndarray, timestamp_ms: int | None = None) -> dict:
        """Returns {'vector', 'has_hands', 'face', 'blendshapes'} for one frame.

        The frame is NOT mirrored here. MediaPipe labels hands as seen by the camera; we
        swap to the signer's own left/right so a front-facing camera is anatomically right.
        """
        import cv2
        self._ts = max(self._ts + 1, timestamp_ms if timestamp_ms is not None else self._ts + 33)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

        pose_res = self.pose.detect_for_video(img, self._ts)
        hand_res = self.hand.detect_for_video(img, self._ts)
        pose = self._arr(pose_res.pose_landmarks[0]) if pose_res.pose_landmarks else None
        left = right = None
        for lms, handed in zip(hand_res.hand_landmarks, hand_res.handedness):
            label = handed[0].category_name            # as seen in the (non-mirrored) image
            arr = self._arr(lms)
            if label == "Left":                        # image-left == signer's right hand
                right = arr
            else:
                left = arr
        face = blend = None
        if self.face is not None:
            fr = self.face.detect_for_video(img, self._ts)
            if fr.face_landmarks:
                face = self._arr(fr.face_landmarks[0])
                if fr.face_blendshapes:
                    blend = {c.category_name: c.score for c in fr.face_blendshapes[0]}
        return {"vector": frame_vector(pose, left, right, face), "has_hands": left is not None or right is not None,
                "face": face, "blendshapes": blend, "pose": pose, "left": left, "right": right}

    def close(self) -> None:
        for m in (self.hand, self.pose, self.face):
            if m is not None:
                m.close()
