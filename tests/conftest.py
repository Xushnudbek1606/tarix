import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
from cryptography.fernet import Fernet


@pytest.fixture()
def settings(tmp_path):
    from backend.app.config import Settings
    s = Settings(env="test", secret_key="test-secret-key-0123456789-0123456789",
                 data_encryption_key=Fernet.generate_key().decode(), kiosk_api_key="kiosk-test-key",
                 admin_password="admin-test-pass", database_url="sqlite:///:memory:",
                 rate_limit_login=1000, rate_limit_api=100000, rate_limit_kiosk=100000)
    s.data_dir = tmp_path / "data"
    s.dataset_dir = tmp_path / "dataset"
    s.models_dir = tmp_path / "models"
    for d in (s.data_dir, s.dataset_dir, s.models_dir):
        d.mkdir()
    src = ROOT / "dataset" / "signs.json"
    (s.dataset_dir / "signs.json").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    s.frontend_dir = ROOT / "frontend"
    return s


@pytest.fixture()
def app(settings):
    from backend.app.main import create_app
    return create_app(settings)


@pytest.fixture()
def client(app):
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c


def login(client, username="admin", password="admin-test-pass"):
    r = client.post("/api/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": "Bearer " + r.json()["access_token"]}


def kiosk_headers(client, component="display", desk="DESK-1"):
    r = client.post("/api/kiosk/token", json={"desk_code": desk, "component": component},
                    headers={"X-Kiosk-Key": "kiosk-test-key"})
    assert r.status_code == 200, r.text
    return {"Authorization": "Bearer " + r.json()["access_token"]}, r.json()["access_token"]


@pytest.fixture()
def admin(client):
    return login(client)


def unit(seed, dim=128):
    v = np.random.default_rng(seed).normal(size=dim).astype("float32")
    return v / np.linalg.norm(v)
