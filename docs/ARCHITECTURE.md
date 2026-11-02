# Architecture

```
 Citizen PC (desk)                                   Server                        Employee PC
 ┌──────────────────────────┐   REST + WSS     ┌──────────────────┐   WSS    ┌────────────────┐
 │ AI service  (ai/service) │◄───────────────► │ FastAPI backend  │◄───────► │ Employee UI    │
 │  camera → MediaPipe      │  identify, events│  JWT, RBAC, hub  │          │ (browser)      │
 │  → temporal model (ONNX) │                  │  sessions, audit │          └────────────────┘
 │  → Uzbek NLP → events    │                  │  encryption      │   WSS    ┌────────────────┐
 │  face embedding+liveness │                  │  PostgreSQL      │◄───────► │ Admin panel    │
 └────────────┬─────────────┘                  └────────▲─────────┘          └────────────────┘
   MJPEG preview (127.0.0.1)                             │ WSS
 ┌────────────▼─────────────┐                           │
 │ Citizen monitor (browser,│◄──────────────────────────┘
 │ kiosk mode, large text)  │
 └──────────────────────────┘
```

* **Video never leaves the desk PC.** The AI service owns the camera; the browser only shows a local MJPEG preview from `127.0.0.1:8765`. Only landmarks-derived text, confidence values and a face *embedding* (not an image) go to the backend.
* **Pipeline (per frame):** `ai/landmarks.py` (MediaPipe Hand + Pose + Face landmarkers) → `ai/features.py` (body-relative normalisation: shoulder-centred, shoulder-width scaled, separate hand-shape block, face block) → `ai/recognizer.py` (segments signs by hand activity, resamples each segment to 32 frames, appends velocities) → `ai/model.py` (Transformer / GRU / TCN, ONNX Runtime, loaded once) → `ai/nlp.py` (glosses → Uzbek sentence) → WebSocket.
* **Temporal, not static:** a whole segment (positions of both hands, hand shape, body, face, and frame-to-frame velocity) is classified at once.
* **Latency:** model inference is 1–6 ms on CPU. A sign is reported ≈0.3 s after the hands drop (`end_gap`); the *sentence* is emitted after a 2 s pause (`sentence_gap_s`), which is a deliberate design choice so a sentence is not cut in the middle. Both are configurable in `RecognizerConfig`.
* **Replaceable model:** `models/<version>/{model.onnx, meta.json}` + `models/current.json`. Activating another version (Admin → Training) hot-reloads it on every desk with no restart. The class list lives in `meta.json`, the vocabulary in `dataset/signs.json` / the `signs` table.
* **Backend modules:** `config` (env), `security` (bcrypt, Fernet, JWT, rate limiter), `models` (ORM, field-level encryption), `deps` (auth/RBAC), `hub` (WebSocket routing), `services` (sessions, messages, face index), `routers/*` (REST), `routers/ws.py` (real-time).
* **Frontends:** dependency-free ES modules served by the backend (`frontend/`), no build step, works offline on closed networks. Citizen/employee/admin are separate pages sharing `frontend/shared`.

## Deliberate limits
* The backend keeps WebSocket state in memory → run **one** backend process (use a reverse proxy for TLS, not for load balancing). Moving the hub to Redis pub/sub is the path to multiple instances.
* Rate limiting is in-process per IP.
* Sentence generation is rule-based (templates in `ai/nlp.py`), not a learned translator; unknown combinations fall back to a literal gloss join so nothing is invented.
