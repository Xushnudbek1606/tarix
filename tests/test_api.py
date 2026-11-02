import json

import pytest
from sqlalchemy import text

from tests.conftest import kiosk_headers, login, unit


def mk_citizen(client, admin, name="Aliyev Vali", pin="1234", nid=None):
    r = client.post("/api/admin/citizens", headers=admin, json={"full_name": name, "pin": pin, "national_id": nid})
    assert r.status_code == 201, r.text
    return r.json()


def enroll(app, citizen_id, seed):
    from backend.app.models import Citizen
    db = app.state.SessionLocal()
    try:
        app.state.face_index.enroll(db, db.get(Citizen, citizen_id), "test-embedder", unit(seed))
    finally:
        db.close()


# ----------------------------------------------------------------------------- auth / RBAC
def test_login_and_rbac(client, admin):
    assert client.get("/api/admin/stats").status_code == 401
    r = client.post("/api/admin/employees", headers=admin, json={"username": "emp1", "password": "longpassword", "full_name": "Emp One", "desk_id": 1})
    assert r.status_code == 201
    emp = login(client, "emp1", "longpassword")
    assert client.get("/api/desks", headers=emp).status_code == 200
    assert client.get("/api/admin/stats", headers=emp).status_code == 403          # employees are not admins
    disp, _ = kiosk_headers(client)
    assert client.get("/api/desks", headers=disp).status_code == 403                # kiosk is not staff
    assert client.get("/api/admin/stats", headers=disp).status_code == 403


def test_account_lockout(client, admin):
    client.post("/api/admin/employees", headers=admin, json={"username": "emp2", "password": "rightpassword", "full_name": "E2"})
    for _ in range(5):
        assert client.post("/api/auth/login", json={"username": "emp2", "password": "wrong"}).status_code == 401
    assert client.post("/api/auth/login", json={"username": "emp2", "password": "rightpassword"}).status_code == 423


def test_kiosk_token_requires_key(client):
    assert client.post("/api/kiosk/token", json={"desk_code": "DESK-1"}, headers={"X-Kiosk-Key": "nope"}).status_code == 401
    assert client.post("/api/kiosk/token", json={"desk_code": "NOPE"}, headers={"X-Kiosk-Key": "kiosk-test-key"}).status_code == 404


def test_expired_or_forged_token_rejected(client, settings):
    import jwt
    bad = jwt.encode({"sub": "u:1", "typ": "user", "exp": 1}, settings.secret_key, algorithm="HS256")
    assert client.get("/api/desks", headers={"Authorization": "Bearer " + bad}).status_code == 401
    forged = jwt.encode({"sub": "u:1", "typ": "user", "exp": 9999999999}, "other-key-other-key-other-key-0000", algorithm="HS256")
    assert client.get("/api/desks", headers={"Authorization": "Bearer " + forged}).status_code == 401


def test_rate_limit(app):
    from fastapi.testclient import TestClient
    app.state.settings.rate_limit_login = 3
    with TestClient(app) as c:
        codes = [c.post("/api/auth/login", json={"username": "x", "password": "y"}).status_code for _ in range(5)]
    assert codes[:3] == [401] * 3 and 429 in codes[3:]


# ----------------------------------------------------------------------------- data protection
def test_personal_data_encrypted_at_rest(app, client, admin):
    c = mk_citizen(client, admin, name="Sensitive Name", nid="AB1234567")
    enroll(app, c["id"], 1)
    with app.state.engine.connect() as conn:
        raw_name = conn.execute(text("SELECT full_name FROM citizens")).scalar()
        raw_nid = conn.execute(text("SELECT national_id_hash FROM citizens")).scalar()
        raw_vec = conn.execute(text("SELECT vector_enc FROM face_embeddings")).scalar()
        raw_pin = conn.execute(text("SELECT pin_hash FROM citizens")).scalar()
    assert "Sensitive" not in raw_name and "AB1234567" not in raw_nid
    assert raw_pin.startswith("$2") and "1234" not in raw_pin
    import numpy as np
    assert unit(1).tobytes() not in bytes(raw_vec)                                  # encrypted, not raw float bytes
    assert client.get("/api/admin/citizens", headers=admin).json()[0]["full_name"] == "Sensitive Name"   # but readable via the app


def test_duplicate_national_id_rejected(client, admin):
    mk_citizen(client, admin, nid="X1")
    assert client.post("/api/admin/citizens", headers=admin, json={"full_name": "Other", "national_id": "x1"}).status_code == 409


# ----------------------------------------------------------------------------- Face ID + PIN
def test_face_identify_flow(app, client, admin):
    a, b = mk_citizen(client, admin, "Alice"), mk_citizen(client, admin, "Bob")
    enroll(app, a["id"], 1); enroll(app, b["id"], 2)
    disp, _ = kiosk_headers(client)
    def ident(vec, live=True):
        return client.post("/api/kiosk/identify", headers=disp, json={"embedder": "test-embedder", "embedding": vec.tolist(), "liveness": live}).json()
    assert ident(unit(1), live=False) == {"matched": False, "reason": "liveness", "score": 0.0}   # photo / no liveness
    assert ident(unit(99))["matched"] is False                                                      # stranger
    r = ident(unit(1))
    assert r["matched"] and r["session"]["citizen"]["name"] == "Alice" and r["session"]["auth_method"] == "FACE"
    assert ident(unit(1))["session"]["id"] == r["session"]["id"]                                    # idempotent for same person
    assert client.post("/api/kiosk/identify", headers=disp, json={"embedder": "test-embedder", "embedding": unit(2).tolist(), "liveness": True}).status_code == 409   # desk busy


def test_unknown_embedder_never_matches(app, client, admin):
    c = mk_citizen(client, admin); enroll(app, c["id"], 1)
    disp, _ = kiosk_headers(client)
    r = client.post("/api/kiosk/identify", headers=disp, json={"embedder": "other-model", "embedding": unit(1).tolist(), "liveness": True}).json()
    assert r["matched"] is False


def test_pin_fallback_and_lockout(client, admin):
    c = mk_citizen(client, admin, pin="4321")
    disp, _ = kiosk_headers(client)
    for _ in range(5):
        assert client.post("/api/kiosk/pin-login", headers=disp, json={"citizen_code": c["code"], "pin": "0000"}).status_code == 401
    assert client.post("/api/kiosk/pin-login", headers=disp, json={"citizen_code": c["code"], "pin": "4321"}).status_code == 423   # locked
    assert client.post("/api/kiosk/pin-login", headers=disp, json={"citizen_code": "C-9999", "pin": "1234"}).status_code == 401    # unknown looks identical


def test_pin_login_success(client, admin):
    c = mk_citizen(client, admin, pin="4321")
    disp, _ = kiosk_headers(client)
    r = client.post("/api/kiosk/pin-login", headers=disp, json={"citizen_code": c["code"].lower(), "pin": "4321"})
    assert r.status_code == 200 and r.json()["session"]["auth_method"] == "PIN"


def test_delete_citizen_removes_biometrics(app, client, admin):
    c = mk_citizen(client, admin); enroll(app, c["id"], 1)
    assert client.delete(f"/api/admin/citizens/{c['id']}", headers=admin).status_code == 200
    with app.state.engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM face_embeddings")).scalar() == 0
    disp, _ = kiosk_headers(client)
    assert client.post("/api/kiosk/identify", headers=disp, json={"embedder": "test-embedder", "embedding": unit(1).tolist(), "liveness": True}).json()["matched"] is False


# ----------------------------------------------------------------------------- sessions + messages
def test_session_lifecycle_and_idle_timeout(app, client, admin):
    emp = admin
    s = client.post("/api/sessions", headers=emp, json={"desk_id": 1, "guest": True}).json()
    assert s["service_number"].startswith("A") and s["auth_method"] == "GUEST"
    assert client.post("/api/sessions", headers=emp, json={"desk_id": 1, "guest": True}).status_code == 409
    m = client.post(f"/api/sessions/{s['id']}/messages", headers=emp, json={"text": "Pasportingizni kameraga ko'rsating."}).json()
    assert m["text"] == "Pasportingizni kameraga ko‘rsating." and m["text_cyr"] == "Паспортингизни камерага кўрсатинг."
    ru = client.post(f"/api/sessions/{s['id']}/messages", headers=emp, json={"text": "Подождите, пожалуйста", "lang": "ru"}).json()
    assert ru["text"] == ru["text_cyr"] == "Подождите, пожалуйста"                    # Russian never transliterated
    assert client.post(f"/api/sessions/{s['id']}/messages", headers=emp, json={"text": "   "}).status_code in (422,)
    full = client.get(f"/api/sessions/{s['id']}", headers=emp).json()
    assert [x["sender"] for x in full["messages"]] == ["EMPLOYEE", "EMPLOYEE"]
    # idle expiry
    from backend.app.models import ServiceSession
    from backend.app.services import expire_idle_sessions
    from datetime import timedelta
    from backend.app.models import utcnow
    db = app.state.SessionLocal()
    row = db.get(ServiceSession, s["id"]); row.last_activity = utcnow() - timedelta(minutes=30); db.commit()
    assert [x.id for x in expire_idle_sessions(db, 10)] == [s["id"]]
    db.close()
    assert client.get(f"/api/sessions/{s['id']}", headers=emp).json()["end_reason"] == "timeout"
    assert client.post(f"/api/sessions/{s['id']}/messages", headers=emp, json={"text": "late"}).status_code == 409


def test_message_length_and_html_are_data_only(client, admin):
    s = client.post("/api/sessions", headers=admin, json={"desk_id": 1, "guest": True}).json()
    assert client.post(f"/api/sessions/{s['id']}/messages", headers=admin, json={"text": "x" * 601}).status_code == 422
    r = client.post(f"/api/sessions/{s['id']}/messages", headers=admin, json={"text": "<script>alert(1)</script>"}).json()
    assert r["text"] == "<script>alert(1)</script>"        # stored verbatim; the UI renders via textContent only


# ----------------------------------------------------------------------------- WebSocket: the real-time path
def ws_open(client, token):
    ws = client.websocket_connect("/ws")
    c = ws.__enter__()
    c.send_json({"type": "auth", "token": token})
    return ws, c


def drain_until(c, kind, limit=20):
    for _ in range(limit):
        ev = c.receive_json()
        if ev["type"] == kind:
            return ev
    raise AssertionError(f"never received {kind}")


def test_ws_rejects_bad_token(client):
    from starlette.websockets import WebSocketDisconnect
    with client.websocket_connect("/ws") as c:
        c.send_json({"type": "auth", "token": "garbage"})
        with pytest.raises(WebSocketDisconnect):
            c.receive_json()


def test_end_to_end_two_way_communication(app, client, admin):
    cit = mk_citizen(client, admin, "Karimov Ali")
    enroll(app, cit["id"], 7)
    emp_token = admin["Authorization"][7:]
    disp_h, disp_t = kiosk_headers(client, "display")
    ai_h, ai_t = kiosk_headers(client, "ai")

    ws_e, emp = ws_open(client, emp_token); assert drain_until(emp, "ready")
    ws_d, dis = ws_open(client, disp_t); assert drain_until(dis, "ready")
    ws_a, ai = ws_open(client, ai_t); assert drain_until(ai, "ready")
    emp.send_json({"type": "subscribe", "desk_id": 1}); assert drain_until(emp, "subscribed")["ai_online"] is True

    # 1. AI service identifies the face -> session starts on both screens
    r = client.post("/api/kiosk/identify", headers=ai_h, json={"embedder": "test-embedder", "embedding": unit(7).tolist(), "liveness": True}).json()
    assert r["matched"]
    sid = r["session"]["id"]
    assert drain_until(dis, "session.started")["session"]["citizen"]["name"] == "Karimov Ali"
    assert drain_until(emp, "desk.update")["session"]["id"] == sid

    # 2. citizen signs -> AI sentence -> employee sees text (and the display too)
    ai.send_json({"type": "sentence", "text": "Men pasport olish uchun keldim.",
                  "items": [{"sign": s, "confidence": .95} for s in ("MEN", "PASPORT", "OLISH", "KELDIM")]})
    m = drain_until(emp, "message.new")["message"]
    assert m["sender"] == "CITIZEN" and m["text"] == "Men pasport olish uchun keldim." and m["signs"] == ["MEN", "PASPORT", "OLISH", "KELDIM"]
    assert m["text_cyr"] == "Мен паспорт олиш учун келдим."
    assert drain_until(dis, "message.new")["message"]["id"] == m["id"]

    # 3. employee answers over the socket -> citizen monitor receives it instantly
    emp.send_json({"type": "message.send", "session_id": sid, "text": "Pasportingizni kameraga ko'rsating."})
    got = drain_until(dis, "message.new")["message"]
    assert got["sender"] == "EMPLOYEE" and got["text"] == "Pasportingizni kameraga ko‘rsating." and got["text_cyr"].startswith("Паспортингизни")

    # 4. low-confidence recognition is NOT stored as a statement
    ai.send_json({"type": "sentence", "text": "Men yordam", "items": [{"sign": "YORDAM", "confidence": .4}]})
    assert drain_until(dis, "uncertain")["confidence"] == 0.4
    assert len(client.get(f"/api/sessions/{sid}", headers=admin).json()["messages"]) == 2

    # 5. made-up sign names are rejected
    ai.send_json({"type": "sentence", "text": "hack", "items": [{"sign": "NOT_A_SIGN", "confidence": .99}]})
    assert drain_until(ai, "error")["error"] == "invalid sentence"

    # 6. a citizen display cannot post employee messages or end sessions
    dis.send_json({"type": "message.send", "session_id": sid, "text": "evil"})
    assert drain_until(dis, "error")
    # 7. end session
    emp.send_json({"type": "session.end", "session_id": sid})
    assert drain_until(dis, "session.ended")["reason"] == "employee"
    # confidence logs recorded
    logs = client.get("/api/admin/logs/recognition", headers=admin).json()
    assert {l["sign"] for l in logs} >= {"PASPORT", "YORDAM"} and any(not l["accepted"] for l in logs)
    for w in (ws_e, ws_d, ws_a):
        w.__exit__(None, None, None)


def test_ai_offline_event_and_camera_permission(client, admin):
    ai_h, ai_t = kiosk_headers(client, "ai")
    ws_e, emp = ws_open(client, admin["Authorization"][7:]); drain_until(emp, "ready")
    emp.send_json({"type": "subscribe", "desk_id": 1}); assert drain_until(emp, "subscribed")["ai_online"] is False
    ws_a, ai = ws_open(client, ai_t); drain_until(ai, "ready")
    assert drain_until(emp, "ai.online")
    ws_a.__exit__(None, None, None)
    assert drain_until(emp, "ai.offline")
    assert client.patch("/api/admin/desks/1", headers=admin, json={"camera_enabled": False}).json()["camera_enabled"] is False
    assert client.get("/api/kiosk/config", headers=ai_h).json()["desk"]["camera_enabled"] is False
    ws_e.__exit__(None, None, None)


def test_sign_management_without_code_changes(app, client, admin, settings):
    r = client.post("/api/admin/signs", headers=admin, json={"name": "MAKTAB", "uz_latin": "maktab", "category": "place"})
    assert r.status_code == 201 and r.json()["uz_cyrillic"] == "мактаб"
    names = [s["name"] for s in json.loads((settings.dataset_dir / "signs.json").read_text(encoding="utf-8"))["signs"]]
    assert "MAKTAB" in names                                                   # registry file stays in sync for training
    assert client.post("/api/admin/signs", headers=admin, json={"name": "MAKTAB", "uz_latin": "x"}).status_code == 409
    cfg = client.get("/api/kiosk/config", headers=kiosk_headers(client)[0]).json()
    assert "MAKTAB" in [s["name"] for s in cfg["signs"]]


def test_audit_log_records_security_events(client, admin):
    client.post("/api/auth/login", json={"username": "admin", "password": "bad"})
    actions = {a["action"] for a in client.get("/api/admin/logs/audit", headers=admin).json()}
    assert {"login.ok", "login.failed"} <= actions


def test_security_headers(client):
    h = client.get("/health").headers
    assert h["x-content-type-options"] == "nosniff" and h["x-frame-options"] == "DENY" and "default-src 'self'" in h["content-security-policy"]


def test_static_frontends_served(client):
    for p in ("/citizen/", "/employee/", "/admin/", "/shared/common.js"):
        assert client.get(p).status_code == 200, p
