"""Administrator endpoints: employees, citizens, face enrolment, signs, dataset, models, settings, logs."""
from __future__ import annotations

import secrets
import shutil
import time
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from common.uzbek_text import canonical_text, to_script

from ..deps import Principal, client_ip, get_db, require
from ..models import (AuditLog, Citizen, Desk, Employee, FaceEmbedding, ModelVersion, RecognitionLog,
                      ServiceSession, Sign, SignDataset, User, iso, utcnow)
from ..modelreg import activate, current_meta, current_version, list_models, public_model_info
from ..security import hash_secret
from ..services import (audit, counts_by, runtime_settings, session_out, set_runtime_setting,
                        sync_signs_file)

router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(require("ADMIN"))])
admin_only = require("ADMIN")


# --------------------------------------------------------------------------- #
# Desks
# --------------------------------------------------------------------------- #
class DeskIn(BaseModel):
    code: str = Field(min_length=1, max_length=32, pattern=r"^[A-Za-z0-9_\-]+$")
    name: str = Field(min_length=1, max_length=120)
    camera_index: int = Field(default=0, ge=0, le=16)
    camera_enabled: bool = True


class DeskPatch(BaseModel):
    name: str | None = Field(default=None, max_length=120)
    camera_index: int | None = Field(default=None, ge=0, le=16)
    camera_enabled: bool | None = None
    active: bool | None = None


def _desk_out(d: Desk, hub) -> dict:
    return {"id": d.id, "code": d.code, "name": d.name, "camera_index": d.camera_index,
            "camera_enabled": d.camera_enabled, "active": d.active, "ai_online": hub.ai_online(d.id)}


@router.get("/desks")
def desks(request: Request, db: Session = Depends(get_db)):
    return [_desk_out(d, request.app.state.hub) for d in db.scalars(select(Desk).order_by(Desk.id))]


@router.post("/desks", status_code=201)
def create_desk(body: DeskIn, request: Request, p: Principal = Depends(admin_only), db: Session = Depends(get_db)):
    d = Desk(code=body.code.upper(), name=body.name, camera_index=body.camera_index,
             camera_enabled=body.camera_enabled)
    db.add(d)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "Desk code already exists")
    audit(db, p.actor, "desk.create", target=d.code, ip=client_ip(request))
    return _desk_out(d, request.app.state.hub)


@router.patch("/desks/{desk_id}")
def patch_desk(desk_id: int, body: DeskPatch, request: Request, p: Principal = Depends(admin_only),
               db: Session = Depends(get_db)):
    d = db.get(Desk, desk_id)
    if not d:
        raise HTTPException(404, "Desk not found")
    for k, v in body.model_dump(exclude_none=True).items():
        setattr(d, k, v)
    db.commit()
    hub = request.app.state.hub
    hub.emit_desk(d.id, {"type": "config.update"}, only="ai")           # camera index / permission changed
    audit(db, p.actor, "desk.update", target=d.code, detail=str(body.model_dump(exclude_none=True)),
          ip=client_ip(request))
    return _desk_out(d, hub)


# --------------------------------------------------------------------------- #
# Employees
# --------------------------------------------------------------------------- #
class EmployeeIn(BaseModel):
    username: str = Field(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9_.\-]+$")
    password: str = Field(min_length=8, max_length=72)
    full_name: str = Field(min_length=1, max_length=160)
    desk_id: int | None = None
    position: str = Field(default="", max_length=120)
    role: str = Field(default="EMPLOYEE", pattern="^(EMPLOYEE|ADMIN)$")


def _employee_out(u: User) -> dict:
    return {"id": u.id, "username": u.username, "full_name": u.full_name, "role": u.role, "active": u.active,
            "desk_id": u.employee.desk_id if u.employee else None,
            "position": u.employee.position if u.employee else ""}


@router.get("/employees")
def employees(db: Session = Depends(get_db)):
    return [_employee_out(u) for u in db.scalars(select(User).order_by(User.id))]


@router.post("/employees", status_code=201)
def create_employee(body: EmployeeIn, request: Request, p: Principal = Depends(admin_only),
                    db: Session = Depends(get_db)):
    if body.desk_id is not None and not db.get(Desk, body.desk_id):
        raise HTTPException(404, "Desk not found")
    u = User(username=body.username.lower(), password_hash=hash_secret(body.password),
             role=body.role, full_name=body.full_name)
    u.employee = Employee(desk_id=body.desk_id, position=body.position)
    db.add(u)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "Username already exists")
    audit(db, p.actor, "employee.create", target=u.username, ip=client_ip(request))
    return _employee_out(u)


@router.delete("/employees/{user_id}")
def remove_employee(user_id: int, request: Request, p: Principal = Depends(admin_only),
                    db: Session = Depends(get_db)):
    """Accounts are deactivated rather than deleted, so audit history stays intact."""
    u = db.get(User, user_id)
    if not u:
        raise HTTPException(404, "User not found")
    if u.id == p.id:
        raise HTTPException(409, "You cannot remove your own account")
    u.active = False
    db.commit()
    audit(db, p.actor, "employee.remove", target=u.username, ip=client_ip(request))
    return {"ok": True}


# --------------------------------------------------------------------------- #
# Citizens + Face ID enrolment
# --------------------------------------------------------------------------- #
class CitizenIn(BaseModel):
    full_name: str = Field(min_length=1, max_length=160)
    national_id: str | None = Field(default=None, max_length=32)
    pin: str | None = Field(default=None, min_length=4, max_length=8, pattern=r"^\d+$")
    script: str = Field(default="latin", pattern="^(latin|cyrillic)$")


class EnrollIn(BaseModel):
    desk_id: int


def _citizen_out(c: Citizen) -> dict:
    return {"id": c.id, "code": c.code, "full_name": c.full_name, "script": c.script, "active": c.active,
            "has_pin": bool(c.pin_hash), "face_templates": len(c.embeddings), "created_at": iso(c.created_at)}


@router.get("/citizens")
def citizens(q: str = Query("", max_length=60), db: Session = Depends(get_db)):
    rows = db.scalars(select(Citizen).where(Citizen.active.is_(True)).order_by(Citizen.id.desc()).limit(300)).all()
    out = [_citizen_out(c) for c in rows]
    if q:                                    # names are encrypted, so filtering happens in memory
        ql = q.strip().lower()
        out = [c for c in out if ql in c["full_name"].lower() or ql in c["code"].lower()]
    return out


@router.post("/citizens", status_code=201)
def create_citizen(body: CitizenIn, request: Request, p: Principal = Depends(admin_only),
                   db: Session = Depends(get_db)):
    vault = request.app.state.vault
    nid_hash = vault.lookup_hash(body.national_id) if body.national_id else None
    if nid_hash and db.scalar(select(Citizen).where(Citizen.national_id_hash == nid_hash)):
        raise HTTPException(409, "A citizen with this ID already exists")
    c = Citizen(code="TMP", full_name=canonical_text(body.full_name.strip()), national_id_hash=nid_hash,
                pin_hash=hash_secret(body.pin) if body.pin else None, script=body.script)
    db.add(c)
    db.flush()
    c.code = f"C-{c.id:04d}"
    db.commit()
    audit(db, p.actor, "citizen.create", target=c.code, ip=client_ip(request))
    return _citizen_out(c)


@router.delete("/citizens/{citizen_id}")
def delete_citizen(citizen_id: int, request: Request, p: Principal = Depends(admin_only),
                   db: Session = Depends(get_db)):
    """Removes the citizen's biometric templates and personal data (sessions are anonymised)."""
    c = db.get(Citizen, citizen_id)
    if not c:
        raise HTTPException(404, "Citizen not found")
    for s in db.scalars(select(ServiceSession).where(ServiceSession.citizen_id == c.id)):
        s.citizen_id = None
    code = c.code
    db.delete(c)
    db.commit()
    request.app.state.face_index.invalidate()
    audit(db, p.actor, "citizen.delete", target=code, ip=client_ip(request))
    return {"ok": True}


@router.delete("/citizens/{citizen_id}/face")
def delete_face(citizen_id: int, request: Request, p: Principal = Depends(admin_only), db: Session = Depends(get_db)):
    c = db.get(Citizen, citizen_id)
    if not c:
        raise HTTPException(404, "Citizen not found")
    for e in list(c.embeddings):
        db.delete(e)
    db.commit()
    request.app.state.face_index.invalidate()
    audit(db, p.actor, "face.delete", target=c.code, ip=client_ip(request))
    return {"ok": True}


@router.post("/citizens/{citizen_id}/enroll-face")
def enroll_face(citizen_id: int, body: EnrollIn, request: Request, p: Principal = Depends(admin_only),
                db: Session = Depends(get_db)):
    """Asks the AI service at a desk to capture the citizen's face (live) and return an embedding."""
    c = db.get(Citizen, citizen_id)
    if not c or not c.active:
        raise HTTPException(404, "Citizen not found")
    st = request.app.state
    if not st.hub.ai_online(body.desk_id):
        raise HTTPException(409, "The AI service of this desk is offline")
    req_id = secrets.token_urlsafe(8)
    st.pending_enrollments[req_id] = {"citizen_id": c.id, "desk_id": body.desk_id, "ts": time.time()}
    st.hub.emit_desk(body.desk_id, {"type": "enroll.request", "request_id": req_id, "citizen_code": c.code},
                     only="ai")
    audit(db, p.actor, "face.enroll.request", target=c.code, ip=client_ip(request))
    return {"request_id": req_id}


# --------------------------------------------------------------------------- #
# Signs & dataset
# --------------------------------------------------------------------------- #
class SignIn(BaseModel):
    name: str = Field(min_length=2, max_length=64, pattern=r"^[A-Za-z0-9_]+$")
    uz_latin: str = Field(min_length=1, max_length=120)
    uz_cyrillic: str | None = Field(default=None, max_length=120)
    category: str = Field(default="general", max_length=40)
    pos: str = Field(default="noun", pattern="^(noun|verb|pron|adj|interj|adv)$")


class RecordIn(BaseModel):
    sign_id: int
    desk_id: int
    speaker_id: str = Field(default="unknown", max_length=64)
    duration: float = Field(default=3.0, ge=1.0, le=10.0)


def _sign_out(s: Sign, samples: int = 0) -> dict:
    return {"id": s.id, "name": s.name, "uz_latin": s.uz_latin, "uz_cyrillic": s.uz_cyrillic,
            "category": s.category, "pos": s.pos, "active": s.active, "samples": samples}


@router.get("/signs")
def signs(db: Session = Depends(get_db)):
    counts = counts_by(db, SignDataset.sign_id)
    return [_sign_out(s, counts.get(s.id, 0)) for s in db.scalars(select(Sign).order_by(Sign.id))]


@router.post("/signs", status_code=201)
def create_sign(body: SignIn, request: Request, p: Principal = Depends(admin_only), db: Session = Depends(get_db)):
    """New sign classes are plain data: no code change; retrain to teach the model the new class."""
    latin = canonical_text(body.uz_latin.strip())
    s = Sign(name=body.name.upper(), uz_latin=latin,
             uz_cyrillic=body.uz_cyrillic or to_script(latin, "cyrillic"), category=body.category, pos=body.pos)
    db.add(s)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "A sign with this name already exists")
    sync_signs_file(db, request.app.state.settings.dataset_dir / "signs.json")
    audit(db, p.actor, "sign.create", target=s.name, ip=client_ip(request))
    return _sign_out(s)


@router.delete("/signs/{sign_id}")
def deactivate_sign(sign_id: int, request: Request, p: Principal = Depends(admin_only), db: Session = Depends(get_db)):
    s = db.get(Sign, sign_id)
    if not s:
        raise HTTPException(404, "Sign not found")
    s.active = False
    db.commit()
    sync_signs_file(db, request.app.state.settings.dataset_dir / "signs.json")
    audit(db, p.actor, "sign.deactivate", target=s.name, ip=client_ip(request))
    return {"ok": True}


@router.get("/dataset")
def dataset_stats(db: Session = Depends(get_db)):
    rows = db.execute(select(SignDataset.sign_id, func.count(), func.count(func.distinct(SignDataset.speaker_id)))
                      .group_by(SignDataset.sign_id)).all()
    return {"total": sum(r[1] for r in rows),
            "per_sign": {r[0]: {"samples": r[1], "speakers": r[2]} for r in rows}}


@router.post("/dataset/record")
def record_sample(body: RecordIn, request: Request, p: Principal = Depends(admin_only), db: Session = Depends(get_db)):
    sign = db.get(Sign, body.sign_id)
    if not sign or not sign.active:
        raise HTTPException(404, "Sign not found")
    st = request.app.state
    if not st.hub.ai_online(body.desk_id):
        raise HTTPException(409, "The AI service of this desk is offline")
    req_id = secrets.token_urlsafe(8)
    st.pending_records[req_id] = {"sign_id": sign.id, "desk_id": body.desk_id, "ts": time.time()}
    st.hub.emit_desk(body.desk_id, {"type": "record.request", "request_id": req_id, "sign": sign.name,
                                    "speaker_id": body.speaker_id, "duration": body.duration}, only="ai")
    return {"request_id": req_id}


@router.delete("/dataset/{sample_id}")
def delete_sample(sample_id: int, request: Request, p: Principal = Depends(admin_only), db: Session = Depends(get_db)):
    row = db.get(SignDataset, sample_id)
    if not row:
        raise HTTPException(404, "Sample not found")
    base = request.app.state.settings.dataset_dir
    for rel in (row.landmarks_path, row.video_path):
        if rel and (base / rel).resolve().is_relative_to(base.resolve()):
            (base / rel).unlink(missing_ok=True)
    db.delete(row)
    db.commit()
    audit(db, p.actor, "dataset.delete", target=str(sample_id), ip=client_ip(request))
    return {"ok": True}


# --------------------------------------------------------------------------- #
# Models + training
# --------------------------------------------------------------------------- #
class TrainIn(BaseModel):
    epochs: int = Field(default=60, ge=1, le=500)
    arch: str = Field(default="transformer", pattern="^(transformer|gru|tcn)$")
    allow_synthetic: bool = False


def _sync_model_rows(db: Session, models_dir) -> None:
    known = {m.version for m in db.scalars(select(ModelVersion))}
    for meta in list_models(models_dir):
        if meta["version"] in known:
            continue
        mt = meta.get("metrics", {})
        db.add(ModelVersion(version=meta["version"], arch=meta.get("arch", ""), num_classes=meta.get("num_classes", 0),
                            synthetic=bool(meta.get("synthetic")), val_accuracy=mt.get("val_accuracy"),
                            test_accuracy=mt.get("test_accuracy"), path=meta["version"]))
    db.commit()


@router.get("/models")
def models(request: Request, db: Session = Depends(get_db)):
    d = request.app.state.settings.models_dir
    _sync_model_rows(db, d)
    cur = current_version(d)
    return [{**public_model_info(m), "active": m["version"] == cur} for m in list_models(d)]


@router.get("/models/{version}/metrics")
def model_metrics(version: str, request: Request):
    d = request.app.state.settings.models_dir
    meta = next((m for m in list_models(d) if m["version"] == version), None)
    if not meta:
        raise HTTPException(404, "Model not found")
    return meta


@router.get("/models/{version}/confusion.png")
def model_confusion(version: str, request: Request):
    from fastapi.responses import FileResponse
    d = request.app.state.settings.models_dir
    if not any(m["version"] == version for m in list_models(d)):      # also blocks path traversal
        raise HTTPException(404, "Model not found")
    f = d / version / "confusion_matrix.png"
    if not f.exists():
        raise HTTPException(404, "No confusion matrix")
    return FileResponse(f, media_type="image/png")


@router.post("/models/{version}/activate")
def activate_model(version: str, request: Request, p: Principal = Depends(admin_only), db: Session = Depends(get_db)):
    d = request.app.state.settings.models_dir
    try:
        activate(d, version)
    except FileNotFoundError:
        raise HTTPException(404, "Model not found")
    request.app.state.hub.emit_ai({"type": "model.reload", "version": version})
    audit(db, p.actor, "model.activate", target=version, ip=client_ip(request))
    return {"ok": True, "version": version}


@router.post("/training/start")
def start_training(body: TrainIn, request: Request, p: Principal = Depends(admin_only), db: Session = Depends(get_db)):
    try:
        request.app.state.trainer.start(body.epochs, body.arch, body.allow_synthetic)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    audit(db, p.actor, "training.start", detail=body.model_dump_json(), ip=client_ip(request))
    return {"ok": True}


@router.get("/training/status")
def training_status(request: Request):
    return request.app.state.trainer.status()


# --------------------------------------------------------------------------- #
# Settings, stats, logs
# --------------------------------------------------------------------------- #
class SettingsIn(BaseModel):
    confidence_threshold: float | None = Field(default=None, ge=0.3, le=0.99)
    face_match_threshold: float | None = Field(default=None, ge=0.2, le=0.99)
    liveness_required: bool | None = None
    session_idle_minutes: int | None = Field(default=None, ge=1, le=240)


@router.get("/settings")
def get_settings(request: Request, db: Session = Depends(get_db)):
    return runtime_settings(db, request.app.state.settings)


@router.put("/settings")
def put_settings(body: SettingsIn, request: Request, p: Principal = Depends(admin_only), db: Session = Depends(get_db)):
    for k, v in body.model_dump(exclude_none=True).items():
        set_runtime_setting(db, k, v)
    request.app.state.hub.emit_ai({"type": "config.update"})
    audit(db, p.actor, "settings.update", detail=str(body.model_dump(exclude_none=True)), ip=client_ip(request))
    return runtime_settings(db, request.app.state.settings)


@router.get("/stats")
def stats(request: Request, db: Session = Depends(get_db)):
    now = utcnow()
    day = now - timedelta(days=1)
    week = now - timedelta(days=7)
    total = db.scalar(select(func.count()).select_from(RecognitionLog).where(RecognitionLog.created_at >= week)) or 0
    accepted = db.scalar(select(func.count()).select_from(RecognitionLog).where(
        RecognitionLog.created_at >= week, RecognitionLog.accepted.is_(True))) or 0
    avg_conf = db.scalar(select(func.avg(RecognitionLog.confidence)).where(RecognitionLog.created_at >= week)) or 0.0
    per_sign = db.execute(select(RecognitionLog.sign_name, func.count(), func.avg(RecognitionLog.confidence))
                          .where(RecognitionLog.created_at >= week, RecognitionLog.sign_name.is_not(None))
                          .group_by(RecognitionLog.sign_name).order_by(func.count().desc()).limit(20)).all()
    return {
        "active_sessions": db.scalar(select(func.count()).select_from(ServiceSession).where(ServiceSession.status == "ACTIVE")),
        "sessions_24h": db.scalar(select(func.count()).select_from(ServiceSession).where(ServiceSession.start_time >= day)),
        "citizens": db.scalar(select(func.count()).select_from(Citizen).where(Citizen.active.is_(True))),
        "employees": db.scalar(select(func.count()).select_from(User).where(User.active.is_(True))),
        "recognitions_7d": total, "accepted_7d": accepted,
        "accept_rate_7d": round(accepted / total, 3) if total else None,
        "avg_confidence_7d": round(float(avg_conf), 3),
        "per_sign": [{"sign": r[0], "count": r[1], "avg_confidence": round(float(r[2]), 3)} for r in per_sign],
        "model": public_model_info(current_meta(request.app.state.settings.models_dir)),
        "ai_services_online": sum(1 for d in db.scalars(select(Desk)) if request.app.state.hub.ai_online(d.id)),
    }


@router.get("/sessions")
def admin_sessions(status: str = Query("ACTIVE", pattern="^(ACTIVE|ENDED|ALL)$"), db: Session = Depends(get_db)):
    stmt = select(ServiceSession).order_by(ServiceSession.id.desc()).limit(100)
    if status != "ALL":
        stmt = stmt.where(ServiceSession.status == status)
    return [session_out(s, db) for s in db.scalars(stmt)]


@router.get("/logs/audit")
def audit_logs(limit: int = Query(100, le=500), action: str = Query("", max_length=64), db: Session = Depends(get_db)):
    stmt = select(AuditLog).order_by(AuditLog.id.desc()).limit(limit)
    if action:
        stmt = stmt.where(AuditLog.action.like(f"{action}%"))
    return [{"id": a.id, "ts": iso(a.ts), "actor": a.actor, "action": a.action, "target": a.target,
             "detail": a.detail, "ip": a.ip} for a in db.scalars(stmt)]


@router.get("/logs/recognition")
def recognition_logs(limit: int = Query(100, le=500), db: Session = Depends(get_db)):
    return [{"id": r.id, "ts": iso(r.created_at), "session_id": r.session_id, "desk_id": r.desk_id, "sign": r.sign_name,
             "confidence": round(r.confidence, 3), "accepted": r.accepted, "model": r.model_version}
            for r in db.scalars(select(RecognitionLog).order_by(RecognitionLog.id.desc()).limit(limit))]
