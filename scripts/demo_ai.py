"""Demo AI service WITHOUT a camera: same WebSocket protocol, same recogniser and ONNX model as the real
service, but the "camera" is a test pattern and signs are performed by synthetic landmark sequences.

    python scripts/seed_demo.py
    python scripts/demo_ai.py                      # then, from another terminal:
    curl -X POST localhost:8765/demo/face
    curl -X POST localhost:8765/demo/sign -H 'content-type: application/json' -d '{"signs":["MEN","PASPORT","OLISH","KELDIM"]}'

Needs a trained (even synthetic) model:  python -m training.synthetic && python -m training.train --allow-synthetic
"""
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2                                                    # noqa: E402
import numpy as np                                            # noqa: E402
import uvicorn                                                # noqa: E402
from pydantic import BaseModel                                # noqa: E402

from ai import service as S                                   # noqa: E402
from ai.features import FRAME_DIM                             # noqa: E402
from seed_demo import demo_embedding                          # noqa: E402
from training.synthetic import make_sequence                  # noqa: E402
from training.dataset import load_vocab                       # noqa: E402

svc = S.svc
svc.embedder.name = "demo"            # the seeded demo citizen has a "demo" template
VOCAB = {v["name"]: v["id"] for v in load_vocab(ROOT / "dataset")}
rng = np.random.default_rng(3)
busy = threading.Event()


def fake_camera() -> None:
    svc.camera_ok = True
    svc.status("face_wait")
    n = 0
    while not svc.stop.is_set():
        img = np.zeros((540, 960, 3), np.uint8)
        img[:] = (40 + (n % 120) // 3, 60, 90)
        cv2.putText(img, "DEMO CAMERA (no real video)", (60, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
        cv2.circle(img, (480 + int(200 * np.sin(n / 15)), 400), 30, (0, 200, 255), -1)
        ok, buf = cv2.imencode(".jpg", img)
        svc.jpeg = buf.tobytes()
        n += 1
        time.sleep(1 / 15)


svc.camera_loop = fake_camera


class SignIn(BaseModel):
    signs: list[str]
    speaker: int = 2


@S.api.post("/demo/face")
def demo_face(live: bool = True):
    import httpx
    with svc._http() as c:
        r = c.post("/api/kiosk/identify", json={"embedder": "demo", "embedding": demo_embedding().tolist(), "liveness": live})
    svc.emit({"type": "face", "state": "verified" if r.json().get("matched") else "unknown", "score": r.json().get("score", 0)})
    return r.json()


@S.api.post("/demo/sign")
def demo_sign(body: SignIn):
    if not svc.session_active:
        return {"error": "no active session (call /demo/face or log in with a PIN first)"}
    if busy.is_set():
        return {"error": "already signing"}

    def perform() -> None:
        busy.set()
        try:
            for name in body.signs:
                for vec in make_sequence(VOCAB[name], body.speaker, rng):
                    svc._recognize({"vector": vec}); time.sleep(0.03)
                for _ in range(12):
                    svc._recognize({"vector": np.zeros(FRAME_DIM, np.float32)}); time.sleep(0.03)
            t0 = time.monotonic()
            while time.monotonic() - t0 < svc.recognizer.config.sentence_gap_s + 0.5:      # the pause that ends the sentence
                svc._recognize({"vector": np.zeros(FRAME_DIM, np.float32)}); time.sleep(0.05)
        finally:
            busy.clear()
    threading.Thread(target=perform, daemon=True).start()
    return {"ok": True}


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(S.api, host="127.0.0.1", port=svc.port, log_level="warning")
