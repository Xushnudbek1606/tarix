"""Application settings, loaded from environment variables / .env."""
from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

from cryptography.fernet import Fernet

ROOT = Path(__file__).resolve().parents[2]


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


def _env(name: str, default, cast=str):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    if cast is bool:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return cast(raw)


@dataclass
class Settings:
    env: str = "development"
    dev_mode: bool = False

    secret_key: str = ""
    data_encryption_key: str = ""
    kiosk_api_key: str = ""

    database_url: str = ""
    admin_username: str = "admin"
    admin_password: str = ""

    user_token_minutes: int = 30
    kiosk_token_minutes: int = 600
    max_failed_logins: int = 5
    lockout_minutes: int = 15
    session_idle_minutes: int = 10

    # Recognition / face defaults (can be overridden at runtime in the admin panel)
    confidence_threshold: float = 0.80
    face_match_threshold: float = 0.60
    face_margin: float = 0.03
    liveness_required: bool = True

    rate_limit_login: int = 10          # per minute per IP
    rate_limit_api: int = 600           # per minute per IP
    rate_limit_kiosk: int = 120         # per minute per IP, identify / pin

    cors_origins: list[str] = field(default_factory=list)
    ai_service_url: str = "http://127.0.0.1:8765"
    use_tls: bool = False
    host: str = "127.0.0.1"
    port: int = 8000

    root: Path = ROOT
    data_dir: Path = ROOT / "data"
    dataset_dir: Path = ROOT / "dataset"
    models_dir: Path = ROOT / "models"
    frontend_dir: Path = ROOT / "frontend"

    @classmethod
    def from_env(cls) -> "Settings":
        _load_env_file(ROOT / ".env")
        s = cls()
        s.env = _env("APP_ENV", "development")
        s.dev_mode = _env("DEV_MODE", False, bool)
        s.secret_key = _env("SECRET_KEY", "")
        s.data_encryption_key = _env("DATA_ENCRYPTION_KEY", "")
        s.kiosk_api_key = _env("KIOSK_API_KEY", "")
        s.admin_username = _env("ADMIN_USERNAME", "admin")
        s.admin_password = _env("ADMIN_PASSWORD", "")
        s.user_token_minutes = _env("USER_TOKEN_MINUTES", 30, int)
        s.kiosk_token_minutes = _env("KIOSK_TOKEN_MINUTES", 600, int)
        s.session_idle_minutes = _env("SESSION_IDLE_MINUTES", 10, int)
        s.confidence_threshold = _env("CONFIDENCE_THRESHOLD", 0.80, float)
        s.face_match_threshold = _env("FACE_MATCH_THRESHOLD", 0.60, float)
        s.liveness_required = _env("LIVENESS_REQUIRED", True, bool)
        s.rate_limit_login = _env("RATE_LIMIT_LOGIN", 10, int)
        s.rate_limit_api = _env("RATE_LIMIT_API", 600, int)
        s.rate_limit_kiosk = _env("RATE_LIMIT_KIOSK", 120, int)
        origins = _env("CORS_ORIGINS", "")
        s.cors_origins = [o.strip() for o in origins.split(",") if o.strip()]
        s.ai_service_url = _env("AI_SERVICE_URL", "http://127.0.0.1:8765")
        s.use_tls = _env("USE_TLS", False, bool)
        s.host = _env("HOST", "127.0.0.1")
        s.port = _env("PORT", 8000, int)
        s.data_dir = Path(_env("DATA_DIR", str(ROOT / "data")))
        s.dataset_dir = Path(_env("DATASET_DIR", str(ROOT / "dataset")))
        s.models_dir = Path(_env("MODELS_DIR", str(ROOT / "models")))
        s.database_url = _env("DATABASE_URL", "")
        s.finalize()
        return s

    def finalize(self) -> "Settings":
        """Fill in missing secrets (development only) and derived values."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        missing = [n for n in ("secret_key", "data_encryption_key", "kiosk_api_key", "admin_password")
                   if not getattr(self, n)]
        if missing:
            if self.env == "production":
                raise RuntimeError(
                    "Production requires these settings to be configured explicitly: "
                    + ", ".join(m.upper() for m in missing)
                )
            # Development convenience: generate once and persist next to the data.
            store = self.data_dir / "dev_secrets.json"
            saved = json.loads(store.read_text()) if store.exists() else {}
            generators = {
                "secret_key": lambda: secrets.token_urlsafe(48),
                "data_encryption_key": lambda: Fernet.generate_key().decode(),
                "kiosk_api_key": lambda: secrets.token_urlsafe(24),
                "admin_password": lambda: "admin12345",
            }
            for name in missing:
                saved.setdefault(name, generators[name]())
                setattr(self, name, saved[name])
            store.write_text(json.dumps(saved, indent=2))
        if not self.database_url:
            self.database_url = f"sqlite:///{(self.data_dir / 'govsign.db').as_posix()}"
        return self
