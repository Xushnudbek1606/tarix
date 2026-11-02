"""SQLAlchemy ORM models (PostgreSQL in production, SQLite for local development)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (JSON, Boolean, DateTime, Float, ForeignKey, Integer, LargeBinary,
                        String, Text, TypeDecorator)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from .security import Vault

ROLES = ("ADMIN", "EMPLOYEE", "CITIZEN")

# The vault is installed once at application start-up (see main.create_app).
_VAULT: Vault | None = None


def set_vault(vault: Vault) -> None:
    global _VAULT
    _VAULT = vault


def get_vault() -> Vault:
    if _VAULT is None:
        raise RuntimeError("Vault not initialised")
    return _VAULT


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


class EncryptedText(TypeDecorator):
    """Transparent field-level encryption for personal data."""
    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        return None if value is None else get_vault().encrypt_text(value)

    def process_result_value(self, value, dialect):
        return None if value is None else get_vault().decrypt_text(value)


class Base(DeclarativeBase):
    pass


class Desk(Base):
    __tablename__ = "desks"
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(32), unique=True)
    name: Mapped[str] = mapped_column(String(120))
    camera_index: Mapped[int] = mapped_column(Integer, default=0)
    camera_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(128))
    role: Mapped[str] = mapped_column(String(16))          # ADMIN | EMPLOYEE
    full_name: Mapped[str] = mapped_column(String(160), default="")
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    failed_logins: Mapped[int] = mapped_column(Integer, default=0)
    locked_until: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    employee: Mapped[Optional["Employee"]] = relationship(back_populates="user", uselist=False,
                                                          cascade="all, delete-orphan")


class Employee(Base):
    __tablename__ = "employees"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), unique=True)
    desk_id: Mapped[Optional[int]] = mapped_column(ForeignKey("desks.id"), nullable=True)
    position: Mapped[str] = mapped_column(String(120), default="")
    user: Mapped[User] = relationship(back_populates="employee")


class Citizen(Base):
    __tablename__ = "citizens"
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(24), unique=True, index=True)   # public id, e.g. C-0001
    full_name: Mapped[str] = mapped_column(EncryptedText)
    national_id_hash: Mapped[Optional[str]] = mapped_column(String(64), unique=True, nullable=True)
    pin_hash: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    failed_pins: Mapped[int] = mapped_column(Integer, default=0)
    pin_locked_until: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    script: Mapped[str] = mapped_column(String(10), default="latin")        # latin | cyrillic
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    embeddings: Mapped[list["FaceEmbedding"]] = relationship(back_populates="citizen",
                                                             cascade="all, delete-orphan")


class FaceEmbedding(Base):
    """Encrypted face template. Raw face images are never stored."""
    __tablename__ = "face_embeddings"
    id: Mapped[int] = mapped_column(primary_key=True)
    citizen_id: Mapped[int] = mapped_column(ForeignKey("citizens.id", ondelete="CASCADE"))
    embedder: Mapped[str] = mapped_column(String(64))
    dim: Mapped[int] = mapped_column(Integer)
    vector_enc: Mapped[bytes] = mapped_column(LargeBinary)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    citizen: Mapped[Citizen] = relationship(back_populates="embeddings")


class ServiceSession(Base):
    __tablename__ = "sessions"
    id: Mapped[int] = mapped_column(primary_key=True)
    service_number: Mapped[str] = mapped_column(String(24), default="")
    citizen_id: Mapped[Optional[int]] = mapped_column(ForeignKey("citizens.id"), nullable=True)
    employee_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), nullable=True)
    desk_id: Mapped[int] = mapped_column(ForeignKey("desks.id"))
    auth_method: Mapped[str] = mapped_column(String(16))     # FACE | PIN | EMPLOYEE | GUEST
    status: Mapped[str] = mapped_column(String(12), default="ACTIVE", index=True)   # ACTIVE | ENDED
    end_reason: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    end_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_activity: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    citizen: Mapped[Optional[Citizen]] = relationship()
    messages: Mapped[list["Message"]] = relationship(back_populates="session",
                                                     order_by="Message.id", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    sender: Mapped[str] = mapped_column(String(10))          # CITIZEN | EMPLOYEE | SYSTEM
    text: Mapped[str] = mapped_column(EncryptedText)
    lang: Mapped[str] = mapped_column(String(4), default="uz")          # uz | ru | en (non-uz text is never transliterated)
    signs: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    session: Mapped[ServiceSession] = relationship(back_populates="messages")


class Sign(Base):
    __tablename__ = "signs"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    uz_latin: Mapped[str] = mapped_column(String(120))
    uz_cyrillic: Mapped[str] = mapped_column(String(120), default="")
    category: Mapped[str] = mapped_column(String(40), default="general")
    pos: Mapped[str] = mapped_column(String(16), default="noun")
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class SignDataset(Base):
    """One recorded training sample (landmark sequence + optional video)."""
    __tablename__ = "sign_datasets"
    id: Mapped[int] = mapped_column(primary_key=True)
    sign_id: Mapped[int] = mapped_column(ForeignKey("signs.id"), index=True)
    speaker_id: Mapped[str] = mapped_column(String(64), default="unknown")
    language: Mapped[str] = mapped_column(String(16), default="uzsl")
    video_path: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
    landmarks_path: Mapped[str] = mapped_column(String(300))
    frames: Mapped[int] = mapped_column(Integer, default=0)
    translation: Mapped[str] = mapped_column(String(200), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    sign: Mapped[Sign] = relationship()


class RecognitionLog(Base):
    __tablename__ = "recognition_logs"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[Optional[int]] = mapped_column(ForeignKey("sessions.id"), nullable=True, index=True)
    desk_id: Mapped[Optional[int]] = mapped_column(ForeignKey("desks.id"), nullable=True)
    sign_name: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    accepted: Mapped[bool] = mapped_column(Boolean, default=False)
    model_version: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    actor: Mapped[str] = mapped_column(String(80))           # "user:admin", "desk:DESK-1", "anonymous"
    action: Mapped[str] = mapped_column(String(64), index=True)
    target: Mapped[str] = mapped_column(String(120), default="")
    detail: Mapped[str] = mapped_column(String(500), default="")
    ip: Mapped[str] = mapped_column(String(64), default="")


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[int] = mapped_column(primary_key=True)
    version: Mapped[str] = mapped_column(String(64), unique=True)
    arch: Mapped[str] = mapped_column(String(32), default="")
    num_classes: Mapped[int] = mapped_column(Integer, default=0)
    synthetic: Mapped[bool] = mapped_column(Boolean, default=False)
    val_accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    test_accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    path: Mapped[str] = mapped_column(String(300), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Setting(Base):
    __tablename__ = "settings"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(String(200))
