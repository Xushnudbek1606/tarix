"""Employee Application: checks the network and opens the employee dashboard."""
import subprocess, sys
from pathlib import Path
subprocess.call([sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "launcher.py"), "employee"])
