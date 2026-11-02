"""Download the MediaPipe landmark models (one-time, needs internet).

    python scripts/download_models.py
"""
import sys, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ai.landmarks import MODEL_FILES, MODEL_URLS  # noqa: E402

dest = ROOT / "models" / "mediapipe"
dest.mkdir(parents=True, exist_ok=True)
for key, name in MODEL_FILES.items():
    out = dest / name
    if out.exists():
        print("have", name); continue
    print("downloading", name, "...")
    try:
        urllib.request.urlretrieve(MODEL_URLS[key], out)
    except Exception as exc:
        out.unlink(missing_ok=True)
        print("FAILED:", exc, "\nDownload manually from", MODEL_URLS[key], "and place it in", dest)
        sys.exit(1)
print("ok")
