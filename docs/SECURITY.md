# Security configuration

| Requirement | Implementation |
|---|---|
| HTTPS / WSS | `USE_TLS=true` + `TLS_CERT`/`TLS_KEY` (uvicorn terminates TLS) or a TLS reverse proxy (IIS ARR / nginx / Caddy) in front. HSTS header is sent when `USE_TLS=true`. Desk PCs trust the internal CA via `TLS_CA_BUNDLE`. |
| Authentication | JWT (HS256, `SECRET_KEY`), 30 min sliding for staff; desk devices use a separate token type issued only against `KIOSK_API_KEY` |
| Passwords / PINs | bcrypt (cost 12); account lockout after 5 failures (15 min); constant-time-ish handling of unknown users |
| RBAC | `ADMIN` / `EMPLOYEE` (staff tokens) / `CITIZEN` (kiosk tokens, display vs ai component). Enforced per endpoint and per WebSocket event; covered by `tests/test_api.py` |
| Encrypted data | `citizens.full_name`, `messages.text`: Fernet field encryption. `face_embeddings.vector_enc`: encrypted. National ID: only an HMAC lookup hash. Keys come from env, never the DB |
| Biometrics | Raw face images are never stored or sent; only an embedding leaves the desk PC, is stored encrypted, and is decrypted only in memory for matching. Admin can delete a citizen's face data or the whole citizen (templates are removed, sessions anonymised) |
| Liveness | active challenge (blink + head turn) on the desk PC; server refuses `identify` without it when `LIVENESS_REQUIRED=true` |
| Fallback auth | citizen code + PIN (rate limited, locks), or employee-assisted session start |
| Sessions | idle sessions end automatically (`SESSION_IDLE_MINUTES`); employee UI logs out after 10 min idle; tokens in `sessionStorage` |
| Audit | login ok/failed/locked, face match/no-match, PIN failures, session start/end, admin changes, model activation, dataset uploads → `audit_logs` (viewable in Admin → Logs) |
| Rate limiting | per-IP (login, API, kiosk identify/PIN) and per-WebSocket |
| Input validation | Pydantic models with length/pattern limits; messages ≤ 500 chars; uploads size-capped and loaded with `allow_pickle=False`; UI renders text only via `textContent` (no HTML injection) |
| Headers | CSP (`default-src 'self'`, no inline scripts), `X-Frame-Options: DENY`, `nosniff`, `Referrer-Policy: no-referrer`, `Permissions-Policy` |
| Camera permission | Admin can block a desk's camera (`camera_enabled`); the AI service stops capturing immediately |
| No external AI | all inference is local; nothing is sent to third-party services (the only internet use is the one-time MediaPipe model download) |

## Production checklist
1. `APP_ENV=production` (the server refuses to start without explicit secrets) and a real PostgreSQL with a least-privilege role (see the end of `database/schema.sql`; make `audit_logs` append-only).
2. Generate secrets (see `.env.example`). **Back up `DATA_ENCRYPTION_KEY` separately**; without it encrypted data is unrecoverable. Restrict `.env` file permissions to the service account.
3. Issue a server certificate from your internal CA; install the CA on every desk PC.
4. Change the initial admin password immediately; create named employee accounts.
5. Windows: run services under a dedicated non-admin account, enable BitLocker, auto-lock, kiosk-mode Edge/Chrome for the citizen PC.
6. Install a real face-embedding model (below) and complete a data-protection impact assessment; biometric processing is regulated (Uzbekistan Law on Personal Data) — obtain consent and define retention.

## Known limitations (be honest with your auditors)
* **Face recognition:** without `models/face/face_embedder.onnx` the system falls back to `geometry-v1`, built from face-mesh geometry. It is **demo grade and not suitable as a sole biometric factor**. Provide an ArcFace/MobileFaceNet ONNX model (112×112 RGB, 128/512-d output, input normalised to [-1,1]) and re-enrol citizens (embeddings of different embedders are never compared).
* **Liveness:** blink + head-turn defeats printed photos and static screens. A pre-recorded video of the right person can still fool a single RGB webcam; high-security desks need an IR/depth camera or a dedicated PAD (presentation attack detection) model.
* The kiosk key is a shared secret per installation; rotate it by changing `KIOSK_API_KEY` and reinstalling the desk shortcut.
* Rate limits and the WebSocket hub are per-process (single backend instance).
