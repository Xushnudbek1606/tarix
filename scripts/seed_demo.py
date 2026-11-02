"""Create demo data directly in the database: a demo citizen (PIN 1234 + demo face template) and an employee.

    python scripts/seed_demo.py

Employee login:  employee1 / Employee123!     Demo citizen code is printed (PIN 1234).
The "demo" face template is only used by scripts/demo_ai.py (no camera needed).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                                      # noqa: E402
from sqlalchemy import select                                           # noqa: E402

from backend.app.config import Settings                                 # noqa: E402
from backend.app.database import make_engine, make_session_factory      # noqa: E402
from backend.app.main import seed                                       # noqa: E402
from backend.app.models import Citizen, Employee, FaceEmbedding, User, get_vault, set_vault  # noqa: E402
from backend.app.security import Vault, hash_secret                     # noqa: E402
from fastapi import FastAPI                                             # noqa: E402

DEMO_SEED = 4242


def demo_embedding() -> np.ndarray:
    v = np.random.default_rng(DEMO_SEED).normal(size=128).astype("float32")
    return v / np.linalg.norm(v)


def main() -> None:
    s = Settings.from_env()
    set_vault(Vault(s.data_encryption_key))
    app = FastAPI(); app.state.settings = s
    app.state.SessionLocal = make_session_factory(make_engine(s.database_url))
    seed(app)
    db = app.state.SessionLocal()
    if not db.scalar(select(User).where(User.username == "employee1")):
        u = User(username="employee1", password_hash=hash_secret("Employee123!"), role="EMPLOYEE", full_name="Demo Xodim")
        u.employee = Employee(desk_id=1, position="Operator")
        db.add(u); db.commit()
    c = db.scalar(select(Citizen).where(Citizen.national_id_hash == get_vault().lookup_hash("DEMO-0001")))
    if not c:
        c = Citizen(code="TMP", full_name="Karimov Ali Valiyevich", national_id_hash=get_vault().lookup_hash("DEMO-0001"), pin_hash=hash_secret("1234"))
        db.add(c); db.flush(); c.code = f"C-{c.id:04d}"
        db.add(FaceEmbedding(citizen_id=c.id, embedder="demo", dim=128, vector_enc=get_vault().encrypt(demo_embedding().tobytes())))
        db.commit()
    print(f"employee1 / Employee123!   |   demo citizen {c.code}  PIN 1234")


if __name__ == "__main__":
    main()
