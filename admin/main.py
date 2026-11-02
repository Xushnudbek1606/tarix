"""Admin Panel launcher (the panel itself is served from frontend/admin)."""
import subprocess, sys
from pathlib import Path
subprocess.call([sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "launcher.py"), "admin"])
