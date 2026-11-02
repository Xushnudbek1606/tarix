"""Start the backend (HTTPS/WSS when USE_TLS=true).   python scripts/run_server.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402

from backend.app.config import Settings  # noqa: E402


def main() -> None:
    import os
    s = Settings.from_env()
    kw = {}
    if s.use_tls:
        cert, key = os.environ.get("TLS_CERT"), os.environ.get("TLS_KEY")
        if not (cert and key and Path(cert).exists() and Path(key).exists()):
            sys.exit("USE_TLS=true but TLS_CERT / TLS_KEY are missing. See docs/SECURITY.md (certificates).")
        kw = {"ssl_certfile": cert, "ssl_keyfile": key}
    elif s.env == "production":
        print("WARNING: production without TLS. Passwords, tokens and messages travel unencrypted.", file=sys.stderr)
    uvicorn.run("backend.app.main:create_app", factory=True, host=s.host, port=s.port, proxy_headers=False,
                log_level="info", **kw)


if __name__ == "__main__":
    main()
