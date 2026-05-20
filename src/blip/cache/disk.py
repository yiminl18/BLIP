from __future__ import annotations
import hashlib
import json
import os
import tempfile
from pathlib import Path


def sha256_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class DiskCache:
    """Content-addressed disk cache. Atomic writes via temp-file + rename."""

    def __init__(self, cache_dir: Path, namespace: str = "default") -> None:
        self._dir = cache_dir / namespace
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        h = sha256_key(key)
        return self._dir / f"{h}.json"

    def get(self, key: str):
        p = self._path(key)
        if p.exists():
            return json.loads(p.read_text())
        return None

    def set(self, key: str, value) -> None:
        p = self._path(key)
        data = json.dumps(value, ensure_ascii=False)
        # atomic write
        fd, tmp = tempfile.mkstemp(dir=self._dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(data)
            os.replace(tmp, p)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def has(self, key: str) -> bool:
        return self._path(key).exists()

    def clear(self) -> None:
        for f in self._dir.glob("*.json"):
            f.unlink()
