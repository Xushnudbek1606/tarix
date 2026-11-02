"""GovSign backend: FastAPI + WebSocket.

Run with:  uvicorn backend.app.main:create_app --factory
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:                      # lets `common` and `ai` be imported
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, Request                      # noqa: E402
from fastapi.middleware.cors import CORSMiddleware        # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles               # noqa: E402
from sqlalchemy import select, text                       # noqa: E402
from starlette.concurrency import run_in_threadpool       # noqa: E402

from .config import Settings                              # noqa: E402
from .database import make_engine, make_session_factory   # noqa: E402
from .deps import client_ip                               # noqa: E402
from .hub import Hub                                      # noqa: E402
from .models import Desk, Employee, Sign, User, set_vault  # noqa: E402
from .modelreg import current_meta, public_model_info     # noqa: E402
from .routers import admin, auth, kiosk, sessions, ws     # noqa: E402
from .security import RateLimiter, Vault, hash_secret     # noqa: E402
from .services import FaceIndex, expire_idle_sessions, runtime_settings  # noqa: E402
from .training_runner import TrainingRunner               # noqa: E402

log = logging.getLogger("govsign")


def seed(app: FastAPI) -> None:
    s: Settings = app.state.settings
    db = app.state.SessionLocal()
    try:
        if not db.scalar(select(Desk)):
            db.add(Desk(code="DESK-1", name="Service desk 1"))
            db.commit()
        if not db.scalar(select(User).where(User.role == "ADMIN")):
            admin_user = User(username=s.admin_username.lower(), password_hash=hash_secret(s.admin_password),
                              role="ADMIN", full_name="Administrator")
            admin_user.employee = Employee()
            db.add(admin_user)
            db.commit()
            log.warning("Created initial administrator '%s'", s.admin_username)
        if not db.scalar(select(Sign)):
            f = s.dataset_dir / "signs.json"
            if f.exists():
                for item in json.loads(f.read_text(encoding="utf-8"))["signs"]:
                    db.add(Sign(id=item["id"], name=item["name"], uz_latin=item["uz_latin"],
                                uz_cyrillic=item.get("uz_cyrillic", ""), category=item.get("category", "general"),
                                pos=item.get("pos", "noun")))
                db.commit()
    finally:
        db.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async def idle_watchdog() -> None:
            """Automatic logout: end sessions that have been idle too long."""
            while True:
                await asyncio.sleep(30)
                try:
                    def work():
                        db = app.state.SessionLocal()
                        try:
                            minutes = runtime_settings(db, settings)["session_idle_minutes"]
                            return [(x.id, x.desk_id) for x in expire_idle_sessions(db, minutes)]
                        finally:
                            db.close()
                    for sid, desk_id in await run_in_threadpool(work):
                        await app.state.hub.to_desk(desk_id, {"type": "session.ended", "session_id": sid,
                                                              "reason": "timeout"})
                        await app.state.hub.to_staff({"type": "desk.update", "desk_id": desk_id, "session": None})
                except Exception:
                    log.exception("idle watchdog failed")

        task = asyncio.create_task(idle_watchdog())
        yield
        task.cancel()

    app = FastAPI(title="GovSign – Uzbek Sign Language Service System", version="1.0.0", lifespan=lifespan,
                  docs_url="/api/docs", openapi_url="/api/openapi.json", redoc_url=None)
    app.state.settings = settings
    app.state.vault = Vault(settings.data_encryption_key)
    set_vault(app.state.vault)
    app.state.engine = make_engine(settings.database_url)
    app.state.SessionLocal = make_session_factory(app.state.engine)
    app.state.hub = Hub()
    app.state.limiter = RateLimiter()
    app.state.face_index = FaceIndex()
    app.state.pending_enrollments = {}
    app.state.pending_records = {}
    app.state.trainer = TrainingRunner(settings.root, settings.data_dir)
    seed(app)

    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=False,
                           allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"], allow_headers=["*"])

    ai = settings.ai_service_url.rstrip("/")
    csp = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
           f"img-src 'self' data: blob: {ai}; connect-src 'self' ws: wss: {ai}; "
           "frame-ancestors 'none'; base-uri 'none'; form-action 'self'")

    @app.middleware("http")
    async def security_and_limits(request: Request, call_next):
        path = request.url.path
        if path.startswith("/api/") and not request.app.state.limiter.allow(
                f"api:{client_ip(request)}", settings.rate_limit_api):
            return JSONResponse({"detail": "Too many requests"}, status_code=429, headers={"Retry-After": "60"})
        response = await call_next(request)
        h = response.headers
        h["X-Content-Type-Options"] = "nosniff"
        h["X-Frame-Options"] = "DENY"
        h["Referrer-Policy"] = "no-referrer"
        h["Permissions-Policy"] = "camera=(self), microphone=(), geolocation=()"
        if not path.startswith("/api/docs"):
            h["Content-Security-Policy"] = csp
        if path.startswith("/api/"):
            h["Cache-Control"] = "no-store"
        if settings.use_tls:
            h["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

    app.include_router(auth.router)
    app.include_router(kiosk.router)
    app.include_router(sessions.router)
    app.include_router(admin.router)
    app.include_router(ws.router)

    @app.get("/health", tags=["system"])
    def health(request: Request):
        db = request.app.state.SessionLocal()
        try:
            db.execute(text("SELECT 1"))
            desks = db.scalars(select(Desk).where(Desk.active.is_(True))).all()
            return {"status": "ok", "database": "ok", "version": app.version,
                    "model": public_model_info(current_meta(settings.models_dir)),
                    "desks": [{"code": d.code, "ai_online": request.app.state.hub.ai_online(d.id)} for d in desks]}
        finally:
            db.close()

    landing = """<!doctype html><meta charset=utf-8><title>GovSign</title>
<body style="font-family:system-ui;max-width:640px;margin:10vh auto;padding:0 1rem;line-height:1.7">
<h1>GovSign</h1><p>Uzbek Sign Language government service system.</p>
<ul><li><a href=/citizen/>Citizen station</a></li><li><a href=/employee/>Employee dashboard</a></li>
<li><a href=/admin/>Admin panel</a></li><li><a href=/api/docs>API documentation</a></li></ul></body>"""

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index():
        return landing

    fe = settings.frontend_dir
    for name in ("shared", "citizen", "employee", "admin"):
        if (fe / name).exists():
            app.mount(f"/{name}", StaticFiles(directory=fe / name, html=True), name=name)
    return app

