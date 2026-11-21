"""Registry of databases the agent may query, keyed by opaque ``db_id``."""

import sqlite3
import time
import uuid
from pathlib import Path
from typing import Dict, Optional

SQLITE_MAGIC = b"SQLite format 3\x00"
ALLOWED_SUFFIXES = {".db", ".sqlite", ".sqlite3"}


class InvalidDatabase(Exception):
    pass


class DatabaseRegistry:
    def __init__(
        self,
        upload_dir: str = "temp_uploads",
        max_bytes: int = 200 * 1024 * 1024,
        ttl_s: int = 24 * 3600,
        default_db: Optional[str] = None,
    ):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.ttl_s = ttl_s
        self._paths: Dict[str, Path] = {}
        self.default_id: Optional[str] = None
        if default_db and Path(default_db).exists():
            self.default_id = "default"
            self._paths["default"] = Path(default_db)

    def add_upload(self, filename: str, content: bytes) -> str:
        """Validate and store an uploaded database; return its ``db_id``."""
        if Path(filename).suffix.lower() not in ALLOWED_SUFFIXES:
            raise InvalidDatabase("Only .db, .sqlite or .sqlite3 files are accepted.")
        if len(content) > self.max_bytes:
            raise InvalidDatabase("File exceeds the upload size limit.")
        if not content.startswith(SQLITE_MAGIC):
            raise InvalidDatabase("File is not a valid SQLite database.")

        db_id = uuid.uuid4().hex[:12]
        path = self.upload_dir / f"{db_id}.db"  # never reuse the client filename
        path.write_bytes(content)
        try:
            conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
            try:
                conn.execute("SELECT name FROM sqlite_master LIMIT 1").fetchall()
            finally:
                conn.close()
        except sqlite3.DatabaseError as exc:
            path.unlink(missing_ok=True)
            raise InvalidDatabase(f"Database could not be opened: {exc}") from exc

        self._paths[db_id] = path
        return db_id

    def resolve(self, db_id: Optional[str]) -> Path:
        key = db_id or self.default_id
        if key is None or key not in self._paths:
            raise KeyError(f"Unknown database '{db_id}'.")
        return self._paths[key]

    def cleanup_expired(self) -> int:
        """Delete uploads older than the TTL. Returns the number removed."""
        now = time.time()
        removed = 0
        for db_id, path in list(self._paths.items()):
            if db_id == "default":
                continue
            if path.exists() and now - path.stat().st_mtime > self.ttl_s:
                path.unlink(missing_ok=True)
                del self._paths[db_id]
                removed += 1
        return removed
