"""Convert labelled videos into landmark sequences the trainer can use.

    dataset/pasport/ali_001.mp4  ->  dataset/pasport/ali_001.npz

    python -m training.extract_landmarks            # all videos without an .npz yet
    python -m training.extract_landmarks --force

Requires the MediaPipe models (python scripts/download_models.py). The file name prefix before the
last underscore is the speaker id (used for speaker-independent evaluation).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.landmarks import LandmarkExtractor, models_available  # noqa: E402


def extract_video(extractor: LandmarkExtractor, path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(extractor.extract(frame, timestamp_ms=int(i * 1000 / fps))["vector"])
        i += 1
    cap.release()
    return np.stack(frames) if frames else np.zeros((0, 0), np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(ROOT / "dataset"))
    ap.add_argument("--models", default=str(ROOT / "models" / "mediapipe"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if not models_available(Path(a.models)):
        print("MediaPipe models missing. Run: python scripts/download_models.py")
        return 2
    videos = [v for v in Path(a.dataset).glob("*/*.*") if v.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv", ".webm")
              and not v.parent.name.startswith("_")]
    done = 0
    for v in sorted(videos):
        out = v.with_suffix(".npz")
        if out.exists() and not a.force:
            continue
        ex = LandmarkExtractor(Path(a.models))             # fresh tracker state per video
        seq = extract_video(ex, v)
        ex.close()
        if len(seq) < 8:
            print("skip (too short / unreadable):", v)
            continue
        np.savez_compressed(out, sequence=seq.astype(np.float32))
        done += 1
        print(f"{v.relative_to(a.dataset)} -> {len(seq)} frames")
    print(f"converted {done} video(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
