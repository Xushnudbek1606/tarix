"""Cryptography, password hashing, JWT and rate-limiting helpers."""
from __future__ import annotations

import hashlib
import hmac
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from cryptography.fernet import Fernet, InvalidToken

ALGORITHM = "HS256"


# --------------------------------------------------------------------------- #
# Passwords / PINs
# --------------------------------------------------------------------------- #
def hash_secret(secret: str) -> str:
    return bcrypt.hashpw(secret.encode("utf-8")[:72], bcrypt.gensalt(rounds=12)).decode()


def verify_secret(secret: str, hashed: str | None) -> bool:
    if not hashed:
        return False
    try:
        return bcrypt.checkpw(secret.encode("utf-8")[:72], hashed.encode())
    except ValueError:
        return False


# --------------------------------------------------------------------------- #
# Encryption at rest
# --------------------------------------------------------------------------- #
class Vault:
    """Fernet (AES-128-CBC + HMAC) encryption plus a keyed lookup hash."""

    def __init__(self, key: str):
        self._fernet = Fernet(key.encode() if isinstance(key, str) else key)
        self._hmac_key = hashlib.sha256(b"lookup:" + (key.encode() if isinstance(key, str) else key)).digest()

    def encrypt(self, data: bytes) -> bytes:
        return self._fernet.encrypt(data)

    def decrypt(self, token: bytes) -> bytes:
        try:
            return self._fernet.decrypt(token)
        except InvalidToken as exc:  # wrong key or tampered data
            raise ValueError("cannot decrypt stored value (wrong DATA_ENCRYPTION_KEY?)") from exc

    def encrypt_text(self, text: str) -> str:
        return self.encrypt(text.encode("utf-8")).decode("ascii")

    def decrypt_text(self, token: str) -> str:
        return self.decrypt(token.encode("ascii")).decode("utf-8")

    def lookup_hash(self, value: str) -> str:
        """Deterministic keyed hash, for searching on encrypted identifiers."""
        return hmac.new(self._hmac_key, value.strip().lower().encode(), hashlib.sha256).hexdigest()


# --------------------------------------------------------------------------- #
# JWT
# --------------------------------------------------------------------------- #
def create_token(secret: str, subject: str, minutes: int, **claims) -> tuple[str, int]:
    now = datetime.now(timezone.utc)
    payload = {"sub": subject, "iat": now, "exp": now + timedelta(minutes=minutes), **claims}
    return jwt.encode(payload, secret, algorithm=ALGORITHM), minutes * 60


def decode_token(secret: str, token: str) -> dict:
    return jwt.decode(token, secret, algorithms=[ALGORITHM], options={"require": ["exp", "sub"]})


# --------------------------------------------------------------------------- #
# Rate limiting (sliding window, in-process; put a reverse proxy in front for clusters)
# --------------------------------------------------------------------------- #
class RateLimiter:
    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int, window: float = 60.0) -> bool:
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > window:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(now)
            return True

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()
