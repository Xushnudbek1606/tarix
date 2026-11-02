"""FastAPI dependencies: DB session, authentication, RBAC, rate limiting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import jwt
from fastapi import Depends, HTTPException, Request, WebSocket
from sqlalchemy.orm import Session

from .models import User
from .security import decode_token


@dataclass
class Principal:
    kind: str                # "user" | "kiosk"
    role: str                # ADMIN | EMPLOYEE | CITIZEN
    id: int                  # user id, or desk id for kiosks
    name: str
    desk_id: int | None = None
    component: str | None = None   # kiosk only: "display" | "ai"

    @property
    def actor(self) -> str:
        return f"{'user' if self.kind == 'user' else 'desk'}:{self.name}"


def get_db(request: Request) -> Iterator[Session]:
    db = request.app.state.SessionLocal()
    try:
        yield db
    finally:
        db.close()


def client_ip(request: Request | WebSocket) -> str:
    return request.client.host if request.client else "unknown"


def principal_from_token(app, token: str, db: Session) -> Principal:
    try:
        claims = decode_token(app.state.settings.secret_key, token)
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    typ = claims.get("typ")
    if typ == "user":
        user = db.get(User, int(claims["sub"].split(":")[1]))
        if not user or not user.active:
            raise HTTPException(status_code=401, detail="Account disabled")
        desk_id = user.employee.desk_id if user.employee else None
        return Principal("user", user.role, user.id, user.username, desk_id=desk_id)
    if typ == "kiosk":
        return Principal("kiosk", "CITIZEN", int(claims["desk_id"]), claims["desk_code"],
                         desk_id=int(claims["desk_id"]), component=claims.get("component", "display"))
    raise HTTPException(status_code=401, detail="Unknown token type")


def get_principal(request: Request, db: Session = Depends(get_db)) -> Principal:
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated",
                            headers={"WWW-Authenticate": "Bearer"})
    return principal_from_token(request.app, auth[7:].strip(), db)


def require(*roles: str) -> Callable[..., Principal]:
    def dep(p: Principal = Depends(get_principal)) -> Principal:
        if p.role not in roles:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return p
    return dep


def require_kiosk(component: str | None = None) -> Callable[..., Principal]:
    def dep(p: Principal = Depends(require("CITIZEN"))) -> Principal:
        if p.kind != "kiosk":
            raise HTTPException(status_code=403, detail="Kiosk token required")
        if component and p.component != component:
            raise HTTPException(status_code=403, detail=f"'{component}' component required")
        return p
    return dep


def rate_limit(bucket: str, setting: str) -> Callable[..., None]:
    """Per-IP sliding-window limit; the limit is read from Settings at call time."""
    def dep(request: Request) -> None:
        limit = getattr(request.app.state.settings, setting)
        if not request.app.state.limiter.allow(f"{bucket}:{client_ip(request)}", limit):
            raise HTTPException(status_code=429, detail="Too many requests, slow down",
                                headers={"Retry-After": "60"})
    return dep
