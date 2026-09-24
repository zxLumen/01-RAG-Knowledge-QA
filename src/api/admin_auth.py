"""Persistable admin password.

The admin password can be changed at runtime and is stored hashed at
``qdrant_data/admin.json``. When no file exists, the ``ADMIN_TOKEN`` env var
acts as the fallback password (backward compatible).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading
from pathlib import Path

from src.config import settings

PASSWORD_PATH = Path("./qdrant_data/admin.json")
_ITERATIONS = 200_000
_lock = threading.RLock()


def _hash_password(password: str, *, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), _ITERATIONS
    ).hex()
    return f"pbkdf2_sha256${_ITERATIONS}${salt}${digest}"


def _verify_hash(password: str, stored: str) -> bool:
    try:
        algo, iters, salt, digest = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        computed = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt.encode("utf-8"), int(iters)
        ).hex()
        return hmac.compare_digest(computed, digest)
    except (ValueError, TypeError):
        return False


def _read_hash() -> str:
    try:
        data = json.loads(PASSWORD_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    value = data.get("password_hash")
    return value if isinstance(value, str) else ""


def password_set() -> bool:
    """True when a password is configured (file or env fallback)."""
    return bool(_read_hash() or settings.admin_token)


def verify(provided: str) -> bool:
    """Validate a candidate password against the file hash, then env fallback."""
    if not provided:
        return False
    with _lock:
        stored = _read_hash()
        if stored:
            return _verify_hash(provided, stored)
        expected = settings.admin_token
        return bool(expected) and hmac.compare_digest(provided, expected)


def set_password(new_password: str) -> None:
    """Hash and persist a new admin password."""
    if not new_password or len(new_password) < 4:
        raise ValueError("密码至少 4 位")
    PASSWORD_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"password_hash": _hash_password(new_password)})
    PASSWORD_PATH.write_text(payload, encoding="utf-8")
    try:
        os.chmod(PASSWORD_PATH, 0o600)
    except OSError:
        pass


def change_password(old_password: str, new_password: str) -> None:
    if not verify(old_password):
        raise ValueError("原密码不正确")
    set_password(new_password)
