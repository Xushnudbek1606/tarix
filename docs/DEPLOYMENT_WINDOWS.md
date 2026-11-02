# Windows deployment

Roles: **Server** (backend + PostgreSQL), **Citizen PC** (camera + big monitor: AI service + kiosk browser), **Employee PC** (browser only), **Admin** (browser only).
Requires Python 3.10–3.12 (64-bit) on server and citizen PCs, and Edge or Chrome.

## 1. Server
1. Install PostgreSQL, create DB `govsign` and role `govsign_app` (`database/schema.sql` end). Tables are created automatically on first start.
2. Copy the project to `C:\govsign`, run `scripts\setup.bat`, then edit `.env` (copy of `.env.example`): secrets, `DATABASE_URL`, TLS files, `APP_ENV=production`.
3. `scripts\start_server.bat` → `https://<server>:8443/`. To run at boot use Task Scheduler ("run whether user is logged on", highest privileges off) or NSSM (`nssm install GovSign C:\govsign\.venv\Scripts\python.exe C:\govsign\scripts\run_server.py`). Open the firewall port for the LAN only.
4. Open `/admin/`, log in with `ADMIN_USERNAME` / `ADMIN_PASSWORD`, add desks, employees, citizens.

## 2. Citizen PC (each desk)
1. Copy the project, run `scripts\setup.bat` (downloads the MediaPipe models once — or copy `models\mediapipe\*.task` by USB on air-gapped PCs). Set in `.env`: `BACKEND_URL`, `TLS_CA_BUNDLE`, `KIOSK_API_KEY`, `DESK_CODE`.
2. Put a shortcut to `scripts\start_citizen.bat` in `shell:startup`. It runs the six pre-flight steps (start AI service → camera check → network check → backend connection → status verification → open kiosk UI full-screen) and prints what failed.
3. Windows privacy: Settings → Privacy → Camera → allow desktop apps. Use a kiosk/assigned-access user for the citizen account.

## 3. Employee PC
`scripts\start_employee.bat` (or just open `https://<server>:8443/employee/`). Log in with the personal account.

## 4. First model
Until real data exists, Admin → Training with "allow SYNTHETIC data" produces a *demo* model (flagged DEMO MODEL on every screen). Real workflow: record samples → Start training → check metrics/confusion matrix → Activate.

## 5. GPU
Install `onnxruntime-directml` (any DX12 GPU) or `onnxruntime-gpu` in place of `onnxruntime`; the recogniser picks CUDA → DirectML → CPU automatically.

## 6. Demo without a camera
```
python -m training.synthetic && python -m training.train --allow-synthetic --epochs 25
python scripts\seed_demo.py                 # employee1 / Employee123!, demo citizen (PIN 1234)
python scripts\run_server.py                # (APP_ENV=development, no TLS)
python scripts\demo_ai.py                   # camera-less AI service, same protocol as the real one
curl -X POST localhost:8765/demo/face
curl -X POST localhost:8765/demo/sign -H "content-type: application/json" -d "{\"signs\":[\"MEN\",\"PASPORT\",\"OLISH\",\"KELDIM\"]}"
```
Open `/citizen/#desk=DESK-1&key=<KIOSK_API_KEY from data/dev_secrets.json>` and `/employee/`.

## Troubleshooting
| Symptom | Check |
|---|---|
| Citizen screen: "Imo-ishora tizimi o‘chiq" | AI service not running / cannot reach backend (`http://127.0.0.1:8765/health`) |
| "MediaPipe modellari o‘rnatilmagan" | run `python scripts\download_models.py` |
| Camera black / wrong camera | Admin → Desks → camera #; Windows privacy settings |
| "Model o‘qitilmagan" | no active model: Admin → Training |
| Certificate errors on desk PCs | install the internal CA, set `TLS_CA_BUNDLE` |
