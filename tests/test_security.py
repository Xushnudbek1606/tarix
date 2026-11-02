import pytest
from cryptography.fernet import Fernet

from backend.app.security import RateLimiter, Vault, create_token, decode_token, hash_secret, verify_secret


def test_password_hashing():
    h = hash_secret("s3cret-pass")
    assert h != "s3cret-pass" and verify_secret("s3cret-pass", h) and not verify_secret("wrong", h) and not verify_secret("x", None)


def test_vault_roundtrip_and_tamper():
    v = Vault(Fernet.generate_key().decode())
    tok = v.encrypt_text("Karimov Ali")
    assert "Karimov" not in tok and v.decrypt_text(tok) == "Karimov Ali"
    with pytest.raises(ValueError):
        v.decrypt(b"x" + v.encrypt(b"data"))
    other = Vault(Fernet.generate_key().decode())
    with pytest.raises(ValueError):
        other.decrypt(v.encrypt(b"data"))                      # wrong key
    assert v.lookup_hash(" AB123 ") == v.lookup_hash("ab123") != other.lookup_hash("ab123")


def test_jwt_expiry():
    import jwt
    tok, _ = create_token("k" * 40, "u:1", 1, typ="user")
    assert decode_token("k" * 40, tok)["sub"] == "u:1"
    expired, _ = create_token("k" * 40, "u:1", -1, typ="user")
    with pytest.raises(jwt.ExpiredSignatureError):
        decode_token("k" * 40, expired)
    with pytest.raises(jwt.InvalidSignatureError):
        decode_token("z" * 40, tok)


def test_rate_limiter_window():
    r = RateLimiter()
    assert [r.allow("a", 3) for _ in range(5)] == [True, True, True, False, False]
    assert r.allow("b", 3)


def test_production_requires_explicit_secrets(tmp_path):
    from backend.app.config import Settings
    s = Settings(env="production", data_dir=tmp_path)
    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        s.finalize()
