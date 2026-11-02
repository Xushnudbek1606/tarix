"""Business logic shared by REST routers and the WebSocket handler (synchronous, DB-bound)."""
from __future__ import annotations

import json
import threading
from datetime import timedelta

import numpy as np
from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from common.uzbek_text import canonical_text, is_probably_uzbek, to_script

from .models import (AuditLog, Citizen, Desk, FaceEmbedding, Message, RecognitionLog,
                     ServiceSession, Setting, Sign, User, get_vault, iso, utcnow)

MAX_TEXT = 500


# --------------------------------------------------------------------------- #
# Audit + runtime settings
# --------------------------------------------------------------------------- #
def audit(db: Session, actor: str, action: str, target: str = "", detail: str = "", ip: str = "") -> None:
    db.add(AuditLog(actor=actor[:80], action=action[:64], target=str(target)[:120],
                    detail=detail[:500], ip=ip[:64]))
    db.commit()


_RUNTIME_KEYS = {
    "confidence_threshold": float,
    "face_match_threshold": float,
    "liveness_required": lambda v: str(v).lower() in ("1", "true", "yes", "on"),
    "session_idle_minutes": int,
}


def runtime_settings(db: Session, defaults) -> dict:
    """Admin-editable settings (DB overrides environment defaults)."""
    out = {k: getattr(defaults, k) for k in _RUNTIME_KEYS}
    for row in db.scalars(select(Setting)):
        if row.key in _RUNTIME_KEYS:
            out[row.key] = _RUNTIME_KEYS[row.key](row.value)
    return out


def set_runtime_setting(db: Session, key: str, value) -> None:
    if key not in _RUNTIME_KEYS:
        raise HTTPException(400, f"Unknown setting '{key}'")
    row = db.get(Setting, key)
    if row:
        row.value = str(value)
    else:
        db.add(Setting(key=key, value=str(value)))
    db.commit()


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
def clean_text(text: str) -> str:
    text = " ".join((text or "").split())
    if not text:
        raise HTTPException(422, "Message is empty")
    if len(text) > MAX_TEXT:
        raise HTTPException(422, f"Message is longer than {MAX_TEXT} characters")
    return text


def text_variants(text: str, lang: str = "uz") -> dict:
    """Return the message in both Uzbek scripts (Russian / English text is passed through untouched)."""
    if lang != "uz" or not is_probably_uzbek(text):
        return {"text": text, "text_cyr": text}
    return {"text": to_script(text, "latin"), "text_cyr": to_script(text, "cyrillic")}


# --------------------------------------------------------------------------- #
# Serialisers
# --------------------------------------------------------------------------- #
def message_out(m: Message) -> dict:
    return {"id": m.id, "session_id": m.session_id, "sender": m.sender, "signs": m.signs or [],
            "confidence": m.confidence, "lang": m.lang, "created_at": iso(m.created_at), **text_variants(m.text, m.lang)}


def session_out(s: ServiceSession, db: Session, with_messages: bool = False) -> dict:
    desk = db.get(Desk, s.desk_id)
    citizen = db.get(Citizen, s.citizen_id) if s.citizen_id else None
    employee = db.get(User, s.employee_id) if s.employee_id else None
    out = {
        "id": s.id, "service_number": s.service_number, "status": s.status,
        "auth_method": s.auth_method, "desk_id": s.desk_id, "desk_code": desk.code if desk else "",
        "start_time": iso(s.start_time), "end_time": iso(s.end_time), "end_reason": s.end_reason,
        "citizen": ({"id": citizen.id, "code": citizen.code, "name": citizen.full_name,
                     "script": citizen.script} if citizen else None),
        "employee": ({"id": employee.id, "name": employee.full_name or employee.username}
                     if employee else None),
    }
    if with_messages:
        out["messages"] = [message_out(m) for m in s.messages]
    return out


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #
def active_session(db: Session, desk_id: int) -> ServiceSession | None:
    return db.scalar(select(ServiceSession).where(ServiceSession.desk_id == desk_id,
                                                  ServiceSession.status == "ACTIVE")
                     .order_by(ServiceSession.id.desc()))


def start_session(db: Session, desk: Desk, citizen: Citizen | None, method: str,
                  employee_id: int | None = None) -> tuple[ServiceSession, bool]:
    """Create a session; returns (session, created). Re-uses an identical active session."""
    current = active_session(db, desk.id)
    if current:
        if citizen and current.citizen_id == citizen.id:
            return current, False
        raise HTTPException(409, "This desk already has an active session. End it first.")
    s = ServiceSession(desk_id=desk.id, citizen_id=citizen.id if citizen else None,
                       employee_id=employee_id, auth_method=method)
    db.add(s)
    db.flush()
    s.service_number = f"A{1000 + s.id}"
    db.commit()
    return s, True


def end_session(db: Session, s: ServiceSession, reason: str) -> ServiceSession:
    if s.status != "ENDED":
        s.status = "ENDED"
        s.end_reason = reason
        s.end_time = utcnow()
        db.commit()
    return s


def touch(db: Session, s: ServiceSession) -> None:
    s.last_activity = utcnow()
    db.commit()


def expire_idle_sessions(db: Session, idle_minutes: int) -> list[ServiceSession]:
    cutoff = utcnow() - timedelta(minutes=idle_minutes)
    expired = []
    for s in db.scalars(select(ServiceSession).where(ServiceSession.status == "ACTIVE")):
        last = s.last_activity if s.last_activity.tzinfo else s.last_activity.replace(tzinfo=cutoff.tzinfo)
        if last < cutoff:
            end_session(db, s, "timeout")
            expired.append(s)
    return expired


def add_message(db: Session, s: ServiceSession, sender: str, text: str,
                signs: list[str] | None = None, confidence: float | None = None, lang: str = "uz") -> Message:
    if lang not in ("uz", "ru", "en"):
        raise HTTPException(422, "Unsupported message language")
    text = clean_text(text)
    if sender == "EMPLOYEE" and not any(c.isalpha() for c in text):
        raise HTTPException(422, "Message must contain text")
    m = Message(session_id=s.id, sender=sender, text=canonical_text(text) if lang == "uz" else text,
                lang=lang, signs=signs, confidence=confidence)
    db.add(m)
    s.last_activity = utcnow()
    db.commit()
    return m


def log_recognition(db: Session, session_id: int | None, desk_id: int | None, sign: str | None,
                    confidence: float, accepted: bool, model_version: str = "") -> None:
    db.add(RecognitionLog(session_id=session_id, desk_id=desk_id, sign_name=sign,
                          confidence=float(confidence), accepted=accepted,
                          model_version=model_version[:64]))
    db.commit()


# --------------------------------------------------------------------------- #
# Face identification (templates are encrypted at rest and decrypted only in memory)
# --------------------------------------------------------------------------- #
def _unit(vec) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-8:
        raise HTTPException(422, "Invalid embedding")
    return v / n


class FaceIndex:
    """In-memory cache of decrypted templates, grouped by embedder name."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, tuple[np.ndarray, list[int]]] | None = None

    def invalidate(self) -> None:
        with self._lock:
            self._data = None

    def _build(self, db: Session) -> dict[str, tuple[np.ndarray, list[int]]]:
        groups: dict[str, list[tuple[int, np.ndarray]]] = {}
        vault = get_vault()
        rows = db.execute(select(FaceEmbedding, Citizen.active).join(Citizen)).all()
        for emb, active in rows:
            if not active:
                continue
            vec = np.frombuffer(vault.decrypt(emb.vector_enc), dtype=np.float32)
            groups.setdefault(emb.embedder, []).append((emb.citizen_id, _unit(vec)))
        return {name: (np.stack([v for _, v in items]), [cid for cid, _ in items])
                for name, items in groups.items()}

    def match(self, db: Session, embedder: str, vec, threshold: float,
              margin: float) -> tuple[int | None, float]:
        q = _unit(vec)
        with self._lock:
            if self._data is None:
                self._data = self._build(db)
            entry = self._data.get(embedder)
        if entry is None:
            return None, 0.0
        matrix, ids = entry
        if matrix.shape[1] != q.shape[0]:
            return None, 0.0
        sims = matrix @ q
        best: dict[int, float] = {}
        for cid, sim in zip(ids, sims):
            best[cid] = max(best.get(cid, -1.0), float(sim))
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        top_id, top = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else -1.0
        if top >= threshold and (top - second) >= margin:
            return top_id, top
        return None, top

    def enroll(self, db: Session, citizen: Citizen, embedder: str, vec) -> FaceEmbedding:
        u = _unit(vec)
        row = FaceEmbedding(citizen_id=citizen.id, embedder=embedder[:64], dim=int(u.shape[0]),
                            vector_enc=get_vault().encrypt(u.astype(np.float32).tobytes()))
        db.add(row)
        db.commit()
        self.invalidate()
        return row


# --------------------------------------------------------------------------- #
# Signs <-> dataset/signs.json
# --------------------------------------------------------------------------- #
def sync_signs_file(db: Session, path) -> None:
    signs = db.scalars(select(Sign).where(Sign.active.is_(True)).order_by(Sign.id)).all()
    payload = {"version": 1, "signs": [
        {"id": s.id, "name": s.name, "uz_latin": s.uz_latin, "uz_cyrillic": s.uz_cyrillic,
         "category": s.category, "pos": s.pos} for s in signs]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def counts_by(db: Session, column, *where) -> dict:
    stmt = select(column, func.count()).group_by(column)
    for w in where:
        stmt = stmt.where(w)
    return {k: v for k, v in db.execute(stmt).all()}
