"""The real-time channel. One WebSocket endpoint; the first message authenticates.

Client -> server events
  any:       ping
  staff:     subscribe{desk_id}, message.send{session_id,text}, session.end{session_id}
  display:   assist.request
  ai:        status, face, recognition, sentence, uncertain, enroll.result, record.done, record.failed

Server -> client events
  ready, subscribed, pong, error, message.new, session.started, session.ended, desk.update,
  assist.requested, ai.online, ai.offline, status, face, recognition, uncertain,
  enroll.request, record.request, enroll.done, record.saved, model.reload, config.update
"""
from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from ..deps import Principal, client_ip, principal_from_token
from ..hub import Conn
from ..models import Desk, Sign
from ..modelreg import current_meta
from ..services import (active_session, add_message, audit, end_session, log_recognition, message_out,
                        runtime_settings, session_out, touch)
from sqlalchemy import select

router = APIRouter()
log = logging.getLogger("govsign.ws")

MAX_FRAME = 64 * 1024


def _model_version(app) -> str:
    meta = current_meta(app.state.settings.models_dir)
    return (meta or {}).get("version", "")


# --------------------------------------------------------------------------- #
# Synchronous DB helpers (run in the thread pool)
# --------------------------------------------------------------------------- #
def _desk_snapshot(app, desk_id: int) -> dict:
    db = app.state.SessionLocal()
    try:
        s = active_session(db, desk_id)
        return {"session": session_out(s, db, with_messages=True) if s else None,
                "ai_online": app.state.hub.ai_online(desk_id),
                "status": app.state.hub.last_status.get(desk_id),
                "settings": runtime_settings(db, app.state.settings)}
    finally:
        db.close()


def _staff_send(app, p: Principal, session_id: int, text: str, lang: str = "uz") -> tuple[dict, int]:
    db = app.state.SessionLocal()
    try:
        from ..models import ServiceSession
        s = db.get(ServiceSession, session_id)
        if not s or s.status != "ACTIVE":
            raise HTTPException(409, "There is no active session")
        if not s.employee_id:
            s.employee_id = p.id
        m = add_message(db, s, "EMPLOYEE", text, lang=lang)
        return message_out(m), s.desk_id
    finally:
        db.close()


def _staff_end(app, p: Principal, session_id: int) -> tuple[dict, int] | None:
    db = app.state.SessionLocal()
    try:
        from ..models import ServiceSession
        s = db.get(ServiceSession, session_id)
        if not s or s.status != "ACTIVE":
            return None
        end_session(db, s, "employee")
        audit(db, p.actor, "session.end", target=f"session:{s.id}")
        return session_out(s, db), s.desk_id
    finally:
        db.close()


def _ai_sentence(app, p: Principal, ev: dict) -> tuple[str, dict | None]:
    """Validate and persist a recognised sentence. Returns (kind, payload)."""
    db = app.state.SessionLocal()
    try:
        s = active_session(db, p.desk_id)
        if not s:
            return "no_session", None
        rt = runtime_settings(db, app.state.settings)
        known = {n for (n,) in db.execute(select(Sign.name).where(Sign.active.is_(True)))}
        items = [i for i in (ev.get("items") or []) if isinstance(i, dict)][:40]
        signs = [str(i.get("sign", ""))[:64] for i in items]
        confs = [max(0.0, min(1.0, float(i.get("confidence", 0.0)))) for i in items]
        if not signs or any(x not in known for x in signs):
            return "invalid", None
        text = str(ev.get("text", "")).strip()
        conf = min(confs)
        version = _model_version(app)
        # Server-side guard: never turn an uncertain prediction into a definitive statement.
        if conf < rt["confidence_threshold"]:
            for sg, cf in zip(signs, confs):
                log_recognition(db, s.id, p.desk_id, sg, cf, False, version)
            return "uncertain", {"confidence": round(conf, 3)}
        for sg, cf in zip(signs, confs):
            log_recognition(db, s.id, p.desk_id, sg, cf, True, version)
        m = add_message(db, s, "CITIZEN", text, signs=signs, confidence=conf)
        return "ok", message_out(m)
    finally:
        db.close()


def _ai_recognition(app, p: Principal, ev: dict) -> dict | None:
    db = app.state.SessionLocal()
    try:
        s = active_session(db, p.desk_id)
        if not s:
            return None
        rt = runtime_settings(db, app.state.settings)
        sign = str(ev.get("sign", ""))[:64]
        conf = max(0.0, min(1.0, float(ev.get("confidence", 0.0))))
        accepted = conf >= rt["confidence_threshold"]
        touch(db, s)
        return {"type": "recognition", "sign": sign if accepted else None, "best": sign,
                "confidence": round(conf, 3), "accepted": accepted}
    finally:
        db.close()


def _store_enrollment(app, p: Principal, ev: dict) -> dict:
    from ..models import Citizen
    db = app.state.SessionLocal()
    try:
        pending = app.state.pending_enrollments.pop(str(ev.get("request_id", "")), None)
        if not pending or pending["desk_id"] != p.desk_id:
            return {"type": "enroll.done", "ok": False, "error": "unknown request"}
        if not ev.get("ok"):
            return {"type": "enroll.done", "ok": False, "error": str(ev.get("error", "failed"))[:120],
                    "citizen_id": pending["citizen_id"]}
        if app.state.settings and runtime_settings(db, app.state.settings)["liveness_required"] and not ev.get("liveness"):
            return {"type": "enroll.done", "ok": False, "error": "liveness check failed",
                    "citizen_id": pending["citizen_id"]}
        citizen = db.get(Citizen, pending["citizen_id"])
        if not citizen:
            return {"type": "enroll.done", "ok": False, "error": "citizen removed"}
        emb = ev.get("embedding")
        if not isinstance(emb, list) or not (16 <= len(emb) <= 4096):
            return {"type": "enroll.done", "ok": False, "error": "invalid embedding"}
        app.state.face_index.enroll(db, citizen, str(ev.get("embedder", "unknown")), [float(x) for x in emb])
        audit(db, p.actor, "face.enroll", target=citizen.code)
        return {"type": "enroll.done", "ok": True, "citizen_id": citizen.id, "citizen_code": citizen.code}
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# Endpoint
# --------------------------------------------------------------------------- #
@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    app = ws.app
    hub = app.state.hub
    await ws.accept()

    # --- authentication: first frame must carry the JWT (keeps tokens out of URLs/logs) ---
    try:
        first = await asyncio.wait_for(ws.receive_text(), timeout=8)
        hello = json.loads(first)
        assert hello.get("type") == "auth"
        db = app.state.SessionLocal()
        try:
            principal = principal_from_token(app, str(hello.get("token", "")), db)
        finally:
            db.close()
    except (asyncio.TimeoutError, WebSocketDisconnect):
        await ws.close(code=4401)
        return
    except Exception:
        await ws.close(code=4401)
        return

    conn = Conn(ws=ws, principal=principal)
    if principal.kind == "kiosk":
        conn.desks = {principal.desk_id}
    elif principal.desk_id:
        conn.desks = {principal.desk_id}
    hub.add(conn)
    ws_key = f"ws:{id(conn)}"
    limit = 3000 if conn.is_ai else 240          # messages per minute

    first_desk = next(iter(conn.desks), None)
    snap = await run_in_threadpool(_desk_snapshot, app, first_desk) if first_desk else {}
    await hub.send(conn, {"type": "ready", "role": principal.role, "kind": principal.kind,
                          "component": principal.component, "desk_id": first_desk, **snap})
    if conn.is_ai:
        await hub.to_desk(principal.desk_id, {"type": "ai.online"}, exclude=conn)

    try:
        while True:
            raw = await ws.receive_text()
            if len(raw) > MAX_FRAME:
                await hub.send(conn, {"type": "error", "error": "frame too large"})
                continue
            if not app.state.limiter.allow(ws_key, limit):
                await hub.send(conn, {"type": "error", "error": "rate limit"})
                continue
            try:
                ev = json.loads(raw)
                kind = ev["type"]
            except (ValueError, KeyError, TypeError):
                await hub.send(conn, {"type": "error", "error": "bad message"})
                continue
            try:
                await _dispatch(app, hub, conn, kind, ev)
            except HTTPException as exc:
                await hub.send(conn, {"type": "error", "error": exc.detail, "for": kind})
            except Exception:                                   # a bad event must not kill the socket
                log.exception("event %s failed", kind)
                await hub.send(conn, {"type": "error", "error": "internal error", "for": kind})
    except WebSocketDisconnect:
        pass
    finally:
        hub.remove(conn)
        if conn.is_ai:
            hub.last_status.pop(principal.desk_id, None)
            await hub.to_desk(principal.desk_id, {"type": "ai.offline"})


async def _dispatch(app, hub, conn: Conn, kind: str, ev: dict) -> None:
    p = conn.principal
    if kind == "ping":
        await hub.send(conn, {"type": "pong"})
        return

    # ------------------------------------------------------------ staff
    if conn.is_staff:
        if kind == "subscribe":
            desk_id = int(ev.get("desk_id", 0))
            snap = await run_in_threadpool(_desk_snapshot, app, desk_id)
            if snap["session"] is None and not _desk_exists(app, desk_id):
                raise HTTPException(404, "Desk not found")
            conn.desks = {desk_id}                       # one desk at a time per dashboard
            await hub.send(conn, {"type": "subscribed", "desk_id": desk_id, **snap})
        elif kind == "message.send" and p.role in ("EMPLOYEE", "ADMIN"):
            out, desk_id = await run_in_threadpool(_staff_send, app, p, int(ev.get("session_id", 0)),
                                                   str(ev.get("text", "")), str(ev.get("lang", "uz")))
            await hub.to_desk(desk_id, {"type": "message.new", "message": out})
        elif kind == "session.end":
            res = await run_in_threadpool(_staff_end, app, p, int(ev.get("session_id", 0)))
            if res:
                out, desk_id = res
                await hub.to_desk(desk_id, {"type": "session.ended", "session_id": out["id"], "reason": "employee"})
                await hub.to_staff({"type": "desk.update", "desk_id": desk_id, "session": None})
        else:
            raise HTTPException(400, f"unknown event '{kind}'")
        return

    # ------------------------------------------------------------ desk devices
    if conn.is_display:
        if kind == "assist.request":
            await hub.to_staff({"type": "assist.requested", "desk_id": p.desk_id, "desk_code": p.name})
        else:
            raise HTTPException(400, f"unknown event '{kind}'")
        return

    # AI service
    if kind in ("status", "face"):
        event = {**{k: v for k, v in ev.items() if k in ("state", "camera_active", "model", "fps", "message",
                                                      "score", "reason", "blink", "challenge")}, "type": kind}
        if kind == "status":
            hub.last_status[p.desk_id] = event
        await hub.to_desk(p.desk_id, event, exclude=conn)
    elif kind == "recognition":
        out = await run_in_threadpool(_ai_recognition, app, p, ev)
        if out:
            await hub.to_desk(p.desk_id, out, exclude=conn)
    elif kind == "sentence":
        result, payload = await run_in_threadpool(_ai_sentence, app, p, ev)
        if result == "ok":
            await hub.to_desk(p.desk_id, {"type": "message.new", "message": payload})
        elif result == "uncertain":
            await hub.to_desk(p.desk_id, {"type": "uncertain", **payload})
        elif result == "no_session":
            await hub.send(conn, {"type": "error", "error": "no active session", "for": kind})
        else:
            await hub.send(conn, {"type": "error", "error": "invalid sentence", "for": kind})
    elif kind == "uncertain":
        await hub.to_desk(p.desk_id, {"type": "uncertain", "confidence": ev.get("confidence")}, exclude=conn)
    elif kind == "enroll.result":
        out = await run_in_threadpool(_store_enrollment, app, p, ev)
        await hub.to_admins(out)
    elif kind in ("record.done", "record.failed"):
        await hub.to_admins({"type": kind, "request_id": ev.get("request_id"), "error": str(ev.get("error", ""))[:120]})
    else:
        raise HTTPException(400, f"unknown event '{kind}'")


def _desk_exists(app, desk_id: int) -> bool:
    db = app.state.SessionLocal()
    try:
        d = db.get(Desk, desk_id)
        return bool(d and d.active)
    finally:
        db.close()
