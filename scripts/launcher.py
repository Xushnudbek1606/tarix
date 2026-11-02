"""Desk launcher: pre-flight checks, starts the AI service, opens the right interface full screen.

    python scripts/launcher.py citizen      # citizen station: AI service + kiosk browser
    python scripts/launcher.py employee     # employee desk:   network check + browser
    python scripts/launcher.py admin
Steps: 1 start AI service (citizen)  2 check camera  3 check network  4 connect to backend  5 verify status  6 open UI
"""
import os
import shutil
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402


def env(k, d=""):
    if (ROOT / ".env").exists() and k not in os.environ:
        for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
            if line.startswith(k + "="):
                return line.split("=", 1)[1].strip().strip('"')
    return os.environ.get(k, d)


BACKEND = env("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
AI = f"http://127.0.0.1:{env('AI_PORT', '8765')}"
DESK = env("DESK_CODE", "DESK-1")
VERIFY = env("TLS_CA_BUNDLE") or True


def step(n, msg, ok=None):
    mark = "" if ok is None else ("  [OK]" if ok else "  [FAILED]")
    print(f"[{n}/6] {msg}{mark}", flush=True)


def check_camera():
    import cv2
    idx = int(env("CAMERA_INDEX", "0"))
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(idx)
    ok = cap.isOpened() and cap.read()[0]
    cap.release()
    return ok


def wait_json(url, tries=40, **kw):
    for _ in range(tries):
        try:
            r = httpx.get(url, timeout=3, **kw)
            if r.status_code == 200:
                return r.json()
        except httpx.HTTPError:
            pass
        time.sleep(1)
    return None


def open_ui(url, kiosk):
    edge = shutil.which("msedge") or r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    chrome = shutil.which("chrome") or r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    for exe, flags in ((edge, ["--kiosk", "--edge-kiosk-type=fullscreen"]), (chrome, ["--kiosk"])):
        if exe and Path(exe).exists():
            subprocess.Popen([exe, *(flags if kiosk else ["--start-maximized"]), "--new-window", url])
            return
    webbrowser.open(url)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "citizen"
    if mode == "citizen":
        step(1, "Starting AI recognition service")
        if wait_json(AI + "/health", tries=1) is None:
            flags = 0x08000000 if sys.platform == "win32" else 0          # CREATE_NO_WINDOW
            subprocess.Popen([sys.executable, "-m", "ai.service"], cwd=ROOT, creationflags=flags)
        step(2, "Checking camera")
        cam = check_camera()
        step(2, "Camera", cam)
        if not cam:
            print("   -> Camera not available. Check the USB cable / privacy settings / CAMERA_INDEX. The desk can still use PIN + employee assistance.")
    else:
        step(1, "AI service not needed on this computer"); step(2, "Camera check skipped")
    step(3, "Checking network")
    health = wait_json(BACKEND + "/health", tries=15, verify=VERIFY)
    step(3, f"Backend reachable at {BACKEND}", health is not None)
    if health is None:
        sys.exit("Cannot reach the backend. Check the network cable / server / BACKEND_URL.")
    step(4, "Connected to backend", True)
    if mode == "citizen":
        ai = wait_json(AI + "/health", tries=60)
        step(5, f"AI service: model={ai and ai.get('model')}, camera={ai and ai.get('camera')}, backend link={ai and ai.get('backend_connected')}", ai is not None)
        if ai and not ai.get("model"):
            print("   -> WARNING: no trained model is active. Train one in the Admin panel.")
        key = env("KIOSK_API_KEY")
        open_ui(f"{BACKEND}/citizen/#desk={DESK}&key={key}", kiosk=True)
    else:
        step(5, f"Backend status: {health['status']}, model={(health.get('model') or {}).get('version')}")
        open_ui(f"{BACKEND}/{mode}/", kiosk=False)
    step(6, f"{mode} interface started", True)


if __name__ == "__main__":
    main()
