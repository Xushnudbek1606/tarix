"""AI Recognition Service - runs on each citizen desk computer.

    python -m ai.service            (or scripts\\start_ai.bat on Windows)

* owns the camera, runs MediaPipe + the temporal model + face ID locally (no video leaves the PC)
* serves the live camera preview as MJPEG on http://127.0.0.1:8765/video for the citizen page
* talks to the backend over a WebSocket (events) and REST (face identify, dataset upload)
* the model is loaded once and stays in memory; `model.reload` hot-swaps a new version
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2                      # noqa: E402
import httpx                    # noqa: E402
import numpy as np              # noqa: E402
import uvicorn                  # noqa: E402
import websockets               # noqa: E402
from fastapi import FastAPI     # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse, StreamingResponse  # noqa: E402

from ai.face import Liveness, load_embedder                     # noqa: E402
from ai.landmarks import LandmarkExtractor, models_available    # noqa: E402
from ai.nlp import SentenceBuilder                              # noqa: E402
from ai.recognizer import Recognizer, RecognizerConfig, SignClassifier  # noqa: E402

log = logging.getLogger("govsign.ai")


def env(name: str, default: str) -> str:
    return os.environ.get(name) or default


class AIService:
    def __init__(self) -> None:
        self._load_dotenv()
        self.backend = env("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
        self.kiosk_key = env("KIOSK_API_KEY", self._dev_secret("kiosk_api_key"))
        self.desk_code = env("DESK_CODE", "DESK-1")
        self.models_dir = Path(env("MODELS_DIR", str(ROOT / "models")))
        self.port = int(env("AI_PORT", "8765"))
        self.verify = env("TLS_CA_BUNDLE", "") or (env("TLS_VERIFY", "1") != "0")

        self.classifier = SignClassifier()
        self.builder = SentenceBuilder(ROOT / "dataset" / "signs.json")
        self.recognizer = Recognizer(self.classifier, self.builder, RecognizerConfig())
        self.embedder = load_embedder(self.models_dir)
        self.liveness = Liveness()
        self.extractor: LandmarkExtractor | None = None

        self.cfg: dict = {"settings": {"confidence_threshold": 0.8, "face_match_threshold": 0.6,
                                       "liveness_required": True},
                          "desk": {"camera_index": 0, "camera_enabled": True}}
        self.session_active = False
        self.camera_ok = False
        self.fps = 0.0
        self.state = "starting"
        self.jpeg: bytes | None = None
        self.stop = threading.Event()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.ws = None
        self._token: str | None = None
        self._last_identify = 0.0
        self._last_status = 0.0
        self._detected_until = 0.0
        self._task: dict | None = None          # active enrol / record job
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ config helpers
    @staticmethod
    def _load_dotenv() -> None:
        f = ROOT / ".env"
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip().strip('"'))

    @staticmethod
    def _dev_secret(name: str) -> str:
        f = ROOT / "data" / "dev_secrets.json"
        try:
            return json.loads(f.read_text())[name]
        except Exception:
            return ""

    # ------------------------------------------------------------------ backend REST
    def _http(self) -> httpx.Client:
        return httpx.Client(base_url=self.backend, verify=self.verify, timeout=10,
                            headers={"Authorization": f"Bearer {self._token}"} if self._token else {})

    def fetch_token(self) -> bool:
        try:
            with httpx.Client(base_url=self.backend, verify=self.verify, timeout=10) as c:
                r = c.post("/api/kiosk/token", json={"desk_code": self.desk_code, "component": "ai"},
                           headers={"X-Kiosk-Key": self.kiosk_key})
            if r.status_code != 200:
                log.error("kiosk token refused: %s %s", r.status_code, r.text[:120])
                return False
            self._token = r.json()["access_token"]
            return True
        except httpx.HTTPError as exc:
            log.warning("backend unreachable: %s", exc)
            return False

    def refresh_config(self) -> None:
        with self._http() as c:
            r = c.get("/api/kiosk/config")
        r.raise_for_status()
        data = r.json()
        self.cfg = data
        self.session_active = bool(data.get("session"))
        st = data["settings"]
        self.recognizer.config.threshold = float(st["confidence_threshold"])
        self.builder.__init__(ROOT / "dataset" / "signs.json")
        if self.classifier.load(self.models_dir) is False:
            log.warning("No trained model yet (models/current.json missing)")

    # ------------------------------------------------------------------ events to backend
    def emit(self, event: dict) -> None:
        if self.loop and self.ws is not None:
            asyncio.run_coroutine_threadsafe(self._send(event), self.loop)

    async def _send(self, event: dict) -> None:
        try:
            await self.ws.send(json.dumps(event))
        except Exception:
            pass

    def status(self, state: str, message: str = "") -> None:
        self.state = state
        self._last_status = time.monotonic()
        self.emit({"type": "status", "state": state, "camera_active": self.camera_ok, "fps": round(self.fps, 1),
                   "model": self.classifier.version or None, "message": message})

    # ------------------------------------------------------------------ websocket client loop
    async def ws_loop(self) -> None:
        self.loop = asyncio.get_running_loop()
        backoff = 1.0
        while not self.stop.is_set():
            if not await asyncio.to_thread(self.fetch_token):
                await asyncio.sleep(min(backoff, 15)); backoff *= 1.7; continue
            try:
                await asyncio.to_thread(self.refresh_config)
            except Exception as exc:
                log.warning("config fetch failed: %s", exc)
            url = self.backend.replace("https://", "wss://").replace("http://", "ws://") + "/ws"
            try:
                async with websockets.connect(url, max_size=2**20, ping_interval=20) as ws:
                    await ws.send(json.dumps({"type": "auth", "token": self._token}))
                    self.ws = ws
                    backoff = 1.0
                    log.info("connected to backend as %s", self.desk_code)
                    self.status("listening" if self.session_active else "face_wait")
                    async for raw in ws:
                        await self.on_event(json.loads(raw))
            except Exception as exc:
                log.warning("backend connection lost: %s", exc)
            finally:
                self.ws = None
            await asyncio.sleep(min(backoff, 10)); backoff *= 1.5

    async def on_event(self, ev: dict) -> None:
        t = ev.get("type")
        if t == "session.started":
            self.session_active = True
            self.recognizer.reset()
            self.status("listening")
        elif t == "session.ended":
            self.session_active = False
            self.recognizer.reset()
            self.liveness.reset()
            self.status("face_wait")
        elif t in ("config.update", "model.reload"):
            await asyncio.to_thread(self.refresh_config)
            if self.extractor is None:
                self.start_extractor()
        elif t == "enroll.request":
            self._task = {"kind": "enroll", "id": ev["request_id"], "embs": [], "t0": time.monotonic(), "live": False}
            self.liveness.reset()
        elif t == "record.request":
            self._task = {"kind": "record", "id": ev["request_id"], "sign": ev["sign"], "speaker": ev.get("speaker_id", "unknown"),
                          "duration": float(ev.get("duration", 3.0)), "frames": [], "t0": time.monotonic(), "go": False}

    # ------------------------------------------------------------------ camera
    def start_extractor(self) -> None:
        if self.extractor is None and models_available(self.models_dir / "mediapipe"):
            self.extractor = LandmarkExtractor(self.models_dir / "mediapipe")
            log.info("MediaPipe landmarkers ready")

    def camera_loop(self) -> None:
        cap = None
        idx_open = None
        n, t_fps = 0, time.monotonic()
        while not self.stop.is_set():
            desk = self.cfg.get("desk", {})
            want = bool(desk.get("camera_enabled", True))
            idx = int(desk.get("camera_index", 0))
            if not want:                                    # camera permission switched off by the admin
                if cap is not None:
                    cap.release(); cap = None
                self.camera_ok = False
                self.jpeg = None
                time.sleep(1.0)
                continue
            if cap is None or idx_open != idx:
                if cap is not None:
                    cap.release()
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(idx)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
                idx_open = idx
            ok, frame = cap.read()
            if not ok:
                if self.camera_ok:
                    self.camera_ok = False
                    self.status("no_camera", "Kamera topilmadi")
                time.sleep(0.5)
                cap.release(); cap = None
                continue
            if not self.camera_ok:
                self.camera_ok = True
                self.status("listening" if self.session_active else "face_wait")
            ok2, buf = cv2.imencode(".jpg", cv2.flip(frame, 1), [cv2.IMWRITE_JPEG_QUALITY, 70])   # mirrored preview
            if ok2:
                self.jpeg = buf.tobytes()
            try:
                self.process(frame)
            except Exception:
                log.exception("frame processing failed")
            n += 1
            if time.monotonic() - t_fps >= 1.0:
                self.fps = n / (time.monotonic() - t_fps); n, t_fps = 0, time.monotonic()
                if time.monotonic() - self._last_status > 2:
                    self.status(self.state)
        if cap is not None:
            cap.release()

    # ------------------------------------------------------------------ per-frame logic
    def process(self, frame: np.ndarray) -> None:
        if self.extractor is None:
            self.start_extractor()
            if self.extractor is None:
                if self.state != "no_models":
                    self.status("no_models", "MediaPipe modellari o‘rnatilmagan")
                return
        res = self.extractor.extract(frame)
        task = self._task
        if task and task["kind"] == "record":
            return self._record(task, res)
        if task and task["kind"] == "enroll":
            return self._enroll(task, res, frame)
        if self.session_active:
            return self._recognize(res)
        self._face_auth(res, frame)

    def _recognize(self, res: dict, now: float | None = None) -> None:
        if not self.classifier.loaded:
            if self.state != "no_model":
                self.status("no_model", "Model o‘qitilmagan")
            return
        for ev in self.recognizer.push(res["vector"], now):
            t = ev["type"]
            if t == "state":
                self.status("recognizing")
            elif t == "recognition":
                self.emit({"type": "recognition", "sign": ev["sign"], "confidence": ev["confidence"]})
                if ev["accepted"]:
                    self._detected_until = time.monotonic() + 1.5
                    self.status("detected", ev["sign"])
                log.info("sign %s %.0f%% (%.0f ms)%s", ev["sign"], ev["confidence"] * 100, ev["latency_ms"],
                         "" if ev["accepted"] else "  [rejected]")
            elif t == "sentence":
                self.emit({"type": "sentence", "text": ev["text"], "items": ev["items"]})
                self.status("listening")
            elif t == "uncertain":
                self.emit({"type": "uncertain", "confidence": ev["confidence"]})
                self.status("listening")
        if self.state in ("detected",) and time.monotonic() > self._detected_until:
            self.status("listening")
        elif self.state == "recognizing" and self.recognizer.state == "idle":
            self.status("listening")

    def _face_auth(self, res: dict, frame: np.ndarray) -> None:
        face = res["face"]
        need_live = bool(self.cfg["settings"].get("liveness_required", True))
        if face is None:
            self.liveness.reset()
            if self.state != "face_wait":
                self.status("face_wait")
            return
        lv = self.liveness.update(face, res["blendshapes"])
        ready = lv["passed"] or not need_live
        self.emit({"type": "face", "state": "detected" if not ready else "verifying", "blink": lv["blink"],
                   "challenge": lv["challenge"] if need_live else "done"}) if time.monotonic() - self._last_identify > 0.5 else None
        if not ready or time.monotonic() - self._last_identify < 1.5:
            return
        self._last_identify = time.monotonic()
        emb = self.embedder.embed(frame, face)
        try:
            with self._http() as c:
                r = c.post("/api/kiosk/identify", json={"embedder": self.embedder.name, "embedding": emb.tolist(),
                                                        "liveness": bool(lv["passed"])})
            data = r.json()
        except Exception as exc:
            log.warning("identify failed: %s", exc)
            return
        if data.get("matched"):
            self.emit({"type": "face", "state": "verified", "score": data["score"]})
            self.liveness.reset()
        else:
            self.emit({"type": "face", "state": "unknown", "reason": data.get("reason", ""), "score": data.get("score", 0)})
            self.liveness.reset()

    # ------------------------------------------------------------------ admin jobs
    def _enroll(self, task: dict, res: dict, frame: np.ndarray) -> None:
        face = res["face"]
        if time.monotonic() - task["t0"] > 25:
            self._task = None
            return self.emit({"type": "enroll.result", "request_id": task["id"], "ok": False, "error": "timeout"})
        if face is None:
            return
        lv = self.liveness.update(face, res["blendshapes"])
        task["live"] = task["live"] or lv["passed"]
        if lv["challenge"] != "done" and not task["live"]:
            return
        if abs(Liveness.yaw(face)) < 0.12 and len(task["embs"]) < 8:        # frontal frames only
            task["embs"].append(self.embedder.embed(frame, face))
        if len(task["embs"]) >= 8:
            m = np.mean(task["embs"], axis=0)
            self._task = None
            self.emit({"type": "enroll.result", "request_id": task["id"], "ok": True, "embedder": self.embedder.name,
                       "embedding": (m / np.linalg.norm(m)).tolist(), "liveness": True})

    def _record(self, task: dict, res: dict) -> None:
        now = time.monotonic()
        if now - task["t0"] < 2.0:                         # 2-second countdown to get ready
            return
        if not task["go"]:
            task["go"], task["t_rec"] = True, now
        task["frames"].append(res["vector"])
        if now - task["t_rec"] >= task["duration"]:
            self._task = None
            threading.Thread(target=self._upload_sample, args=(task,), daemon=True).start()

    def _upload_sample(self, task: dict) -> None:
        seq = np.stack(task["frames"]).astype(np.float32)
        buf = io.BytesIO()
        np.savez_compressed(buf, sequence=seq)
        try:
            with self._http() as c:
                r = c.post("/api/kiosk/dataset", data={"sign_name": task["sign"], "speaker_id": task["speaker"],
                                                       "request_id": task["id"]},
                           files={"landmarks": ("s.npz", buf.getvalue(), "application/octet-stream")})
            r.raise_for_status()
            self.emit({"type": "record.done", "request_id": task["id"]})
        except Exception as exc:
            self.emit({"type": "record.failed", "request_id": task["id"], "error": str(exc)[:100]})


svc = AIService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=svc.camera_loop, daemon=True, name="camera").start()
    task = asyncio.create_task(svc.ws_loop())
    yield
    svc.stop.set()
    task.cancel()


api = FastAPI(title="GovSign AI service", lifespan=lifespan, docs_url=None, redoc_url=None)
api.add_middleware(CORSMiddleware, allow_origins=[svc.backend, "http://127.0.0.1:8000", "http://localhost:8000"],
                   allow_methods=["GET"], allow_headers=["*"])


@api.get("/health")
def health():
    return {"camera": svc.camera_ok, "fps": round(svc.fps, 1), "state": svc.state,
            "model": svc.classifier.version or None, "landmarks": svc.extractor is not None,
            "embedder": svc.embedder.name, "backend_connected": svc.ws is not None}


@api.get("/video")
async def video():
    async def gen():
        while True:
            frame = svc.jpeg
            if frame:
                yield b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n"
            await asyncio.sleep(1 / 20)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=f")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    uvicorn.run(api, host="127.0.0.1", port=svc.port, log_level="warning")


if __name__ == "__main__":
    main()
