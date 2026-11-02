"""Employee-facing session and desk endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..deps import Principal, client_ip, get_db, require
from ..models import Citizen, Desk, ServiceSession
from ..services import (active_session, add_message, audit, end_session, message_out, session_out,
                        start_session, text_variants)

router = APIRouter(prefix="/api", tags=["sessions"])
staff = require("EMPLOYEE", "ADMIN")


class StartIn(BaseModel):
    desk_id: int
    citizen_id: int | None = None      # known citizen (employee-assisted authentication)
    guest: bool = False                # walk-in without identification


class MessageIn(BaseModel):
    text: str = Field(min_length=1, max_length=600)
    lang: str = Field(default="uz", pattern="^(uz|ru|en)$")


def _get_session(db: Session, session_id: int) -> ServiceSession:
    s = db.get(ServiceSession, session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    return s


def broadcast_message(request: Request, s: ServiceSession, m) -> dict:
    out = message_out(m)
    request.app.state.hub.emit_desk(s.desk_id, {"type": "message.new", "message": out})
    return out


@router.get("/desks")
def list_desks(request: Request, p: Principal = Depends(staff), db: Session = Depends(get_db)):
    hub = request.app.state.hub
    out = []
    for d in db.scalars(select(Desk).where(Desk.active.is_(True)).order_by(Desk.id)):
        cur = active_session(db, d.id)
        out.append({"id": d.id, "code": d.code, "name": d.name, "camera_enabled": d.camera_enabled,
                    "ai_online": hub.ai_online(d.id), "session": session_out(cur, db) if cur else None})
    return out


@router.get("/sessions")
def list_sessions(status: str = Query("ACTIVE", pattern="^(ACTIVE|ENDED|ALL)$"), limit: int = Query(50, le=200),
                  p: Principal = Depends(staff), db: Session = Depends(get_db)):
    stmt = select(ServiceSession).order_by(ServiceSession.id.desc()).limit(limit)
    if status != "ALL":
        stmt = stmt.where(ServiceSession.status == status)
    return [session_out(s, db) for s in db.scalars(stmt)]


@router.get("/sessions/{session_id}")
def get_session(session_id: int, p: Principal = Depends(staff), db: Session = Depends(get_db)):
    return session_out(_get_session(db, session_id), db, with_messages=True)


@router.post("/sessions")
def employee_start_session(body: StartIn, request: Request, p: Principal = Depends(staff),
                           db: Session = Depends(get_db)):
    """Employee-assisted authentication: open a session for a known citizen or a walk-in guest."""
    desk = db.get(Desk, body.desk_id)
    if not desk or not desk.active:
        raise HTTPException(404, "Desk not found")
    citizen = None
    if body.citizen_id is not None:
        citizen = db.get(Citizen, body.citizen_id)
        if not citizen or not citizen.active:
            raise HTTPException(404, "Citizen not found")
    elif not body.guest:
        raise HTTPException(422, "Provide citizen_id or guest=true")
    s, created = start_session(db, desk, citizen, "EMPLOYEE" if citizen else "GUEST", employee_id=p.id)
    if not s.employee_id:
        s.employee_id = p.id
        db.commit()
    out = session_out(s, db)
    if created:
        hub = request.app.state.hub
        hub.emit_desk(desk.id, {"type": "session.started", "session": out})
        hub.emit_staff({"type": "desk.update", "desk_id": desk.id, "session": out})
        audit(db, p.actor, "session.start", target=f"session:{s.id}", detail="employee-assisted", ip=client_ip(request))
    return out


@router.post("/sessions/{session_id}/claim")
def claim_session(session_id: int, request: Request, p: Principal = Depends(staff), db: Session = Depends(get_db)):
    s = _get_session(db, session_id)
    if s.status != "ACTIVE":
        raise HTTPException(409, "Session already ended")
    s.employee_id = p.id
    db.commit()
    out = session_out(s, db)
    request.app.state.hub.emit_staff({"type": "desk.update", "desk_id": s.desk_id, "session": out})
    return out


@router.post("/sessions/{session_id}/end")
def employee_end_session(session_id: int, request: Request, p: Principal = Depends(staff),
                         db: Session = Depends(get_db)):
    s = _get_session(db, session_id)
    if s.status == "ENDED":
        return session_out(s, db)
    end_session(db, s, "employee")
    hub = request.app.state.hub
    hub.emit_desk(s.desk_id, {"type": "session.ended", "session_id": s.id, "reason": "employee"})
    hub.emit_staff({"type": "desk.update", "desk_id": s.desk_id, "session": None})
    audit(db, p.actor, "session.end", target=f"session:{s.id}", ip=client_ip(request))
    return session_out(s, db)


@router.post("/sessions/{session_id}/messages")
def employee_message(session_id: int, body: MessageIn, request: Request, p: Principal = Depends(staff),
                     db: Session = Depends(get_db)):
    """REST twin of the WebSocket 'message.send' event (the dashboard uses the socket)."""
    s = _get_session(db, session_id)
    if s.status != "ACTIVE":
        raise HTTPException(409, "Session already ended")
    if not s.employee_id:
        s.employee_id = p.id
    m = add_message(db, s, "EMPLOYEE", body.text, lang=body.lang)
    return broadcast_message(request, s, m)


@router.post("/preview-text")
def preview_text(body: MessageIn, p: Principal = Depends(staff)):
    """Shows both scripts of a draft, so employees see what the citizen will read."""
    return text_variants(" ".join(body.text.split()), body.lang)


@router.get("/citizens/lookup")
def lookup_citizen(code: str = Query(..., min_length=2, max_length=24), p: Principal = Depends(staff),
                   db: Session = Depends(get_db)):
    """Exact-code lookup for employee-assisted authentication (no browsing of the citizen register)."""
    c = db.scalar(select(Citizen).where(Citizen.code == code.strip().upper(), Citizen.active.is_(True)))
    if not c:
        raise HTTPException(404, "Citizen not found")
    return {"id": c.id, "code": c.code, "name": c.full_name, "script": c.script}
