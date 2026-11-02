"""Endpoints used by the citizen desk (display + AI service)."""
from __future__ import annotations

import io
import re
import time
from datetime import timedelta

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..deps import Principal, client_ip, get_db, rate_limit, require_kiosk
from ..models import Citizen, Desk, Sign, SignDataset, utcnow
from ..modelreg import current_meta, public_model_info
from ..security import verify_secret
from .auth import _DUMMY_HASH
from ..services import (active_session, audit, end_session, runtime_settings, session_out,
                        start_session)

router = APIRouter(prefix="/api/kiosk", tags=["kiosk"])

MAX_LANDMARK_BYTES = 8 * 1024 * 1024
MAX_VIDEO_BYTES = 60 * 1024 * 1024


class IdentifyIn(BaseModel):
    embedder: str = Field(min_length=1, max_length=64)
    embedding: list[float] = Field(min_length=16, max_length=4096)
    liveness: bool = False


class PinLoginIn(BaseModel):
    citizen_code: str = Field(min_length=1, max_length=24)
    pin: str = Field(min_length=4, max_length=8, pattern=r"^\d+$")


def _announce_start(request: Request, db: Session, session, created: bool) -> dict:
    out = session_out(session, db)
    if created:
        hub = request.app.state.hub
        event = {"type": "session.started", "session": out}
        hub.emit_desk(session.desk_id, event)
        hub.emit_staff({"type": "desk.update", "desk_id": session.desk_id, "session": out})
    return out


@router.get("/config")
def kiosk_config(request: Request, p: Principal = Depends(require_kiosk()), db: Session = Depends(get_db)):
    s = request.app.state.settings
    desk = db.get(Desk, p.desk_id)
    rt = runtime_settings(db, s)
    signs = db.scalars(select(Sign).where(Sign.active.is_(True)).order_by(Sign.id)).all()
    current = active_session(db, desk.id)
    return {
        "desk": {"id": desk.id, "code": desk.code, "name": desk.name, "camera_index": desk.camera_index,
                 "camera_enabled": desk.camera_enabled},
        "settings": rt,
        "model": public_model_info(current_meta(s.models_dir)),
        "signs": [{"name": x.name, "uz_latin": x.uz_latin, "uz_cyrillic": x.uz_cyrillic, "pos": x.pos,
                   "category": x.category} for x in signs],
        "session": session_out(current, db) if current else None,
        "ai_service_url": s.ai_service_url,
    }


@router.post("/identify", dependencies=[Depends(rate_limit("identify", "rate_limit_kiosk"))])
def identify(body: IdentifyIn, request: Request, p: Principal = Depends(require_kiosk()),
             db: Session = Depends(get_db)):
    """Face ID: match an embedding against the encrypted templates and open a session."""
    st = request.app.state
    rt = runtime_settings(db, st.settings)
    if rt["liveness_required"] and not body.liveness:
        audit(db, p.actor, "face.rejected", detail="liveness missing", ip=client_ip(request))
        return {"matched": False, "reason": "liveness", "score": 0.0}
    desk = db.get(Desk, p.desk_id)
    cid, score = st.face_index.match(db, body.embedder, body.embedding,
                                     rt["face_match_threshold"], st.settings.face_margin)
    if cid is None:
        audit(db, p.actor, "face.nomatch", detail=f"best={score:.2f}", ip=client_ip(request))
        return {"matched": False, "reason": "no_match", "score": round(score, 3)}
    citizen = db.get(Citizen, cid)
    session, created = start_session(db, desk, citizen, "FACE")
    if created:
        audit(db, p.actor, "session.start", target=f"session:{session.id}", detail="face", ip=client_ip(request))
    return {"matched": True, "score": round(score, 3), "session": _announce_start(request, db, session, created)}


@router.post("/pin-login", dependencies=[Depends(rate_limit("pin", "rate_limit_kiosk"))])
def pin_login(body: PinLoginIn, request: Request, p: Principal = Depends(require_kiosk()),
              db: Session = Depends(get_db)):
    """Fallback authentication: citizen code + PIN."""
    st = request.app.state
    now = utcnow()
    citizen = db.scalar(select(Citizen).where(Citizen.code == body.citizen_code.strip().upper(),
                                              Citizen.active.is_(True)))
    ok = False
    if citizen:
        locked = citizen.pin_locked_until
        if locked and (locked if locked.tzinfo else locked.replace(tzinfo=now.tzinfo)) > now:
            raise HTTPException(423, "Too many wrong attempts. Please ask the employee for help.")
        ok = verify_secret(body.pin, citizen.pin_hash)
        if not ok:
            citizen.failed_pins += 1
            if citizen.failed_pins >= 5:
                citizen.pin_locked_until = now + timedelta(minutes=st.settings.lockout_minutes)
                citizen.failed_pins = 0
            db.commit()
        else:
            citizen.failed_pins = 0
            citizen.pin_locked_until = None
            db.commit()
    else:
        verify_secret(body.pin, _DUMMY_HASH)        # keep timing similar
    if not ok:
        audit(db, p.actor, "pin.failed", target=body.citizen_code[:24], ip=client_ip(request))
        raise HTTPException(401, "Citizen code or PIN is incorrect")
    desk = db.get(Desk, p.desk_id)
    session, created = start_session(db, desk, citizen, "PIN")
    if created:
        audit(db, p.actor, "session.start", target=f"session:{session.id}", detail="pin", ip=client_ip(request))
    return {"session": _announce_start(request, db, session, created)}


@router.post("/assist")
def request_assistance(request: Request, p: Principal = Depends(require_kiosk()), db: Session = Depends(get_db)):
    """The citizen asks for a human (employee-assisted authentication)."""
    desk = db.get(Desk, p.desk_id)
    event = {"type": "assist.requested", "desk_id": desk.id, "desk_code": desk.code, "at": time.time()}
    request.app.state.hub.emit_staff(event)
    audit(db, p.actor, "assist.request", ip=client_ip(request))
    return {"ok": True}


@router.post("/session/end")
def kiosk_end_session(request: Request, p: Principal = Depends(require_kiosk()), db: Session = Depends(get_db)):
    s = active_session(db, p.desk_id)
    if not s:
        return {"ok": True}
    end_session(db, s, "citizen")
    hub = request.app.state.hub
    hub.emit_desk(s.desk_id, {"type": "session.ended", "session_id": s.id, "reason": "citizen"})
    hub.emit_staff({"type": "desk.update", "desk_id": s.desk_id, "session": None})
    audit(db, p.actor, "session.end", target=f"session:{s.id}", detail="citizen")
    return {"ok": True}


_SAFE = re.compile(r"[^a-z0-9_\-]+")


@router.post("/dataset")
async def upload_sample(request: Request, sign_name: str = Form(...), speaker_id: str = Form("unknown"),
                        request_id: str = Form(...), landmarks: UploadFile = File(...),
                        video: UploadFile | None = File(None),
                        p: Principal = Depends(require_kiosk("ai")), db: Session = Depends(get_db)):
    """The AI service uploads a recorded sample (landmark sequence + optional video)."""
    st = request.app.state
    pending = st.pending_records.pop(request_id, None)
    if not pending or pending["desk_id"] != p.desk_id:
        raise HTTPException(403, "No matching recording request")
    sign = db.scalar(select(Sign).where(Sign.name == sign_name.upper()))
    if not sign or sign.id != pending["sign_id"]:
        raise HTTPException(404, "Unknown sign")
    raw = await landmarks.read(MAX_LANDMARK_BYTES + 1)
    if len(raw) > MAX_LANDMARK_BYTES:
        raise HTTPException(413, "Landmark file too large")
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as z:       # never unpickle uploads
            seq = z["sequence"]
    except Exception:
        raise HTTPException(422, "landmarks must be an .npz with a 'sequence' array")
    if seq.ndim != 2 or seq.shape[0] < 8 or seq.shape[0] > 600:
        raise HTTPException(422, "Unexpected sequence shape")
    folder = st.settings.dataset_dir / _SAFE.sub("", sign.name.lower())
    folder.mkdir(parents=True, exist_ok=True)
    spk = _SAFE.sub("", speaker_id.lower())[:32] or "unknown"
    stem = f"{spk}_{int(time.time() * 1000)}"
    (folder / f"{stem}.npz").write_bytes(raw)
    video_rel = None
    if video is not None:
        vraw = await video.read(MAX_VIDEO_BYTES + 1)
        if len(vraw) > MAX_VIDEO_BYTES:
            raise HTTPException(413, "Video too large")
        if vraw:
            (folder / f"{stem}.mp4").write_bytes(vraw)
            video_rel = f"{folder.name}/{stem}.mp4"
    row = SignDataset(sign_id=sign.id, speaker_id=spk, landmarks_path=f"{folder.name}/{stem}.npz",
                      video_path=video_rel, frames=int(seq.shape[0]), translation=sign.uz_latin)
    db.add(row)
    db.commit()
    audit(db, p.actor, "dataset.sample", target=sign.name, detail=f"speaker={spk}", ip=client_ip(request))
    request.app.state.hub.emit_admins({"type": "record.saved", "sign": sign.name, "sample_id": row.id,
                                       "frames": row.frames})
    return {"ok": True, "sample_id": row.id}
