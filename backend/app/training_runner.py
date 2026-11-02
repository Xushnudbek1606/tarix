"""Runs the training pipeline as a subprocess so the API stays responsive."""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path


class TrainingRunner:
    def __init__(self, root: Path, data_dir: Path) -> None:
        self.root = root
        self.log_path = data_dir / "training.log"
        self._proc: subprocess.Popen | None = None
        self._started: float | None = None
        self._lock = threading.Lock()
        self.last_exit: int | None = None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, epochs: int, arch: str, allow_synthetic: bool, version: str | None = None) -> None:
        with self._lock:
            if self.running:
                raise RuntimeError("Training is already running")
            cmd = [sys.executable, "-m", "training.train", "--epochs", str(epochs), "--arch", arch]
            if allow_synthetic:
                cmd.append("--allow-synthetic")
            if version:
                cmd += ["--version", version]
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            log = open(self.log_path, "w", encoding="utf-8")
            log.write("$ " + " ".join(cmd) + "\n")
            log.flush()
            self._proc = subprocess.Popen(cmd, cwd=self.root, stdout=log, stderr=subprocess.STDOUT)
            self._started = time.time()
            self.last_exit = None

    def status(self, tail: int = 40) -> dict:
        code = self._proc.poll() if self._proc else None
        if self._proc and code is not None:
            self.last_exit = code
        lines: list[str] = []
        if self.log_path.exists():
            lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-tail:]
        return {"running": self.running, "exit_code": self.last_exit, "started_at": self._started,
                "log": lines}
