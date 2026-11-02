# API & WebSocket reference

Interactive OpenAPI docs are served at `/api/docs` when the backend runs. All REST bodies are JSON; errors are `{"detail": "..."}`.
Authentication: `Authorization: Bearer <JWT>`. Staff tokens come from `POST /api/auth/login` (30 min, sliding via `/api/auth/refresh`);
desk devices (citizen display, AI service) get a token from `POST /api/kiosk/token` using the shared `X-Kiosk-Key` header.

## REST endpoints

| Method | Path | Auth |
|---|---|---|
| GET | `/api/admin/citizens` | ADMIN |
| POST | `/api/admin/citizens` | ADMIN |
| DELETE | `/api/admin/citizens/{citizen_id}` | ADMIN |
| POST | `/api/admin/citizens/{citizen_id}/enroll-face` | ADMIN |
| DELETE | `/api/admin/citizens/{citizen_id}/face` | ADMIN |
| GET | `/api/admin/dataset` | ADMIN |
| POST | `/api/admin/dataset/record` | ADMIN |
| DELETE | `/api/admin/dataset/{sample_id}` | ADMIN |
| GET | `/api/admin/desks` | ADMIN |
| POST | `/api/admin/desks` | ADMIN |
| PATCH | `/api/admin/desks/{desk_id}` | ADMIN |
| GET | `/api/admin/employees` | ADMIN |
| POST | `/api/admin/employees` | ADMIN |
| DELETE | `/api/admin/employees/{user_id}` | ADMIN |
| GET | `/api/admin/logs/audit` | ADMIN |
| GET | `/api/admin/logs/recognition` | ADMIN |
| GET | `/api/admin/models` | ADMIN |
| POST | `/api/admin/models/{version}/activate` | ADMIN |
| GET | `/api/admin/models/{version}/confusion.png` | ADMIN |
| GET | `/api/admin/models/{version}/metrics` | ADMIN |
| GET | `/api/admin/sessions` | ADMIN |
| GET | `/api/admin/settings` | ADMIN |
| PUT | `/api/admin/settings` | ADMIN |
| GET | `/api/admin/signs` | ADMIN |
| POST | `/api/admin/signs` | ADMIN |
| DELETE | `/api/admin/signs/{sign_id}` | ADMIN |
| GET | `/api/admin/stats` | ADMIN |
| POST | `/api/admin/training/start` | ADMIN |
| GET | `/api/admin/training/status` | ADMIN |
| POST | `/api/auth/login` | public |
| GET | `/api/auth/me` | any token |
| POST | `/api/auth/refresh` | any token |
| GET | `/api/citizens/lookup` | EMPLOYEE/ADMIN |
| GET | `/api/desks` | EMPLOYEE/ADMIN |
| POST | `/api/kiosk/assist` | kiosk (display/ai) |
| GET | `/api/kiosk/config` | kiosk (display/ai) |
| POST | `/api/kiosk/dataset` | kiosk (display/ai) |
| POST | `/api/kiosk/identify` | kiosk (display/ai) |
| POST | `/api/kiosk/pin-login` | kiosk (display/ai) |
| POST | `/api/kiosk/session/end` | kiosk (display/ai) |
| POST | `/api/kiosk/token` | kiosk key |
| POST | `/api/preview-text` | EMPLOYEE/ADMIN |
| GET | `/api/sessions` | EMPLOYEE/ADMIN |
| POST | `/api/sessions` | EMPLOYEE/ADMIN |
| GET | `/api/sessions/{session_id}` | EMPLOYEE/ADMIN |
| POST | `/api/sessions/{session_id}/claim` | EMPLOYEE/ADMIN |
| POST | `/api/sessions/{session_id}/end` | EMPLOYEE/ADMIN |
| POST | `/api/sessions/{session_id}/messages` | EMPLOYEE/ADMIN |
| GET | `/health` | public |

## WebSocket `/ws`

The first frame must authenticate (the token never appears in the URL): `{"type":"auth","token":"<JWT>"}`. Invalid/missing within 8 s => close code 4401.
Frames are JSON, max 64 KB, rate limited per connection.

| Direction | Event | Who | Payload |
|---|---|---|---|
| S→C | `ready` | all | role, desk_id, session (with messages), ai_online, status, settings |
| C→S | `subscribe` | staff | `{desk_id}` — receive that desk's events; replies `subscribed` snapshot |
| C→S | `message.send` | staff | `{session_id, text, lang: "uz"\|"ru"\|"en"}` — stored, then pushed to the desk monitor instantly |
| C→S | `session.end` | staff | `{session_id}` |
| C→S | `assist.request` | display | citizen asks for an employee |
| C→S | `status` / `face` | ai | camera/AI state, face-ID progress (blink / turn / verified) |
| C→S | `recognition` | ai | one recognised sign `{sign, confidence}` |
| C→S | `sentence` | ai | `{text, items:[{sign, confidence}]}` — **re-checked by the server against the confidence threshold and the sign registry** |
| C→S | `enroll.result` / `record.done` / `record.failed` | ai | admin jobs |
| S→C | `message.new` | desk + staff | `{id, sender, text, text_cyr, signs, confidence, created_at}` (both Uzbek scripts included) |
| S→C | `uncertain` | desk + staff | recognition below threshold — shown as "Imo-ishorani aniq aniqlab bo‘lmadi…" |
| S→C | `session.started` / `session.ended` / `desk.update` | desk / staff | |
| S→C | `assist.requested`, `ai.online`, `ai.offline`, `model.reload`, `config.update`, `record.request`, `enroll.request`, `enroll.done`, `record.saved` | various | |

There is no polling: every screen is driven by these events (WebSocket heartbeat `ping` every 20 s only keeps the link alive).
