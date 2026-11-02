"""Citizen Application: starts the local AI service, checks camera/network and opens the kiosk UI."""
import subprocess, sys
from pathlib import Path
subprocess.call([sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "launcher.py"), "citizen"])
