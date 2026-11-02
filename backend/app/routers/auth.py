from __future__ import annotations

import hmac
from datetime import timedelta

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..deps import Principal, client_ip, get_db, get_principal, rate_limit
from ..models import Desk, User, utcnow
from ..security import create_token, hash_secret, verify_secret
from ..services import audit

router = APIRouter(prefix="/api", tags=["auth"])

_DUMMY_HASH = hash_secret("dummy-password-for-timing")


class LoginIn(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=72)


class KioskTokenIn(BaseModel):
    desk_code: str = Field(min_length=1, max_length=32)
    component: str = Field(default="display", pattern="^(display|ai)$")


def _user_out(u: User) -> dict:
    return {"id": u.id, "username": u.username, "full_name": u.full_name, "role": u.role,
            "desk_id": u.employee.desk_id if u.employee else None}


@router.post("/auth/login", dependencies=[Depends(rate_limit("login", "rate_limit_login"))])
def login(body: LoginIn, request: Request, db: Session = Depends(get_db)):
    s = request.app.state.settings
    ip = client_ip(request)
    user = db.scalar(select(User).where(User.username == body.username.strip().lower()))
    now = utcnow()
    if user and user.locked_until:
        locked = user.locked_until if user.locked_until.tzinfo else user.locked_until.replace(tzinfo=now.tzinfo)
        if locked > now:
            audit(db, f"user:{user.username}", "login.locked", ip=ip)
            raise HTTPException(423, "Account temporarily locked. Try again later.")
    ok = verify_secret(body.password, user.password_hash if user else _DUMMY_HASH)
    if not user or not ok or not user.active:
        if user:
            user.failed_logins += 1
            if user.failed_logins >= s.max_failed_logins:
                user.locked_until = now + timedelta(minutes=s.lockout_minutes)
                user.failed_logins = 0
            db.commit()
        audit(db, f"user:{body.username[:40]}", "login.failed", ip=ip)
        raise HTTPException(401, "Incorrect username or password")
    user.failed_logins = 0
    user.locked_until = None
    db.commit()
    token, ttl = create_token(s.secret_key, f"u:{user.id}", s.user_token_minutes, typ="user", role=user.role)
    audit(db, f"user:{user.username}", "login.ok", ip=ip)
    return {"access_token": token, "token_type": "bearer", "expires_in": ttl, "user": _user_out(user)}


@router.post("/auth/refresh")
def refresh(request: Request, p: Principal = Depends(get_principal), db: Session = Depends(get_db)):
    """Sliding session: the dashboards call this while the user is active; idle users are logged out."""
    if p.kind != "user":
        raise HTTPException(403, "Only user tokens can be refreshed")
    s = request.app.state.settings
    token, ttl = create_token(s.secret_key, f"u:{p.id}", s.user_token_minutes, typ="user", role=p.role)
    return {"access_token": token, "token_type": "bearer", "expires_in": ttl}


@router.get("/auth/me")
def me(p: Principal = Depends(get_principal), db: Session = Depends(get_db)):
    if p.kind == "user":
        return _user_out(db.get(User, p.id))
    return {"role": p.role, "desk_id": p.desk_id, "desk_code": p.name, "component": p.component}


@router.post("/kiosk/token", dependencies=[Depends(rate_limit("kiosk-token", "rate_limit_login"))])
def kiosk_token(body: KioskTokenIn, request: Request, x_kiosk_key: str = Header(default=""),
                db: Session = Depends(get_db)):
    """Desk devices (citizen display + AI service) authenticate with the shared KIOSK_API_KEY."""
    s = request.app.state.settings
    if not hmac.compare_digest(x_kiosk_key.encode(), s.kiosk_api_key.encode()):
        audit(db, f"desk:{body.desk_code[:30]}", "kiosk.token.denied", ip=client_ip(request))
        raise HTTPException(401, "Invalid kiosk key")
    desk = db.scalar(select(Desk).where(Desk.code == body.desk_code, Desk.active.is_(True)))
    if not desk:
        raise HTTPException(404, "Unknown desk")
    token, ttl = create_token(s.secret_key, f"desk:{desk.id}", s.kiosk_token_minutes, typ="kiosk",
                              role="CITIZEN", desk_id=desk.id, desk_code=desk.code, component=body.component)
    return {"access_token": token, "token_type": "bearer", "expires_in": ttl,
            "desk": {"id": desk.id, "code": desk.code, "name": desk.name}}
