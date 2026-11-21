"""Registry of databases the agent may query, keyed by opaque ``db_id``."""

import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, Optional

SQLITE_MAGIC = b"SQLite format 3\x00"
ALLOWED_SUFFIXES = {".db", ".sqlite", ".sqlite3"}
_UPLOAD_ID = re.compile(r"^[0-9a-f]{12}$")  # what add_upload() names its files


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
        # In-memory time of last use per upload. Kept here, never as the file's
        # mtime, because we must not touch the user's file.
        self._last_used: Dict[str, float] = {}
        self.default_id: Optional[str] = None
        # Called with a db_id whenever an upload is deleted (to drop its caches).
        self.on_remove: Optional[Callable[[str], None]] = None
        if default_db and Path(default_db).exists():
            self.default_id = "default"
            self._paths["default"] = Path(default_db)
        self.rescan()

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
        self._last_used[db_id] = time.time()
        return db_id

    def resolve(self, db_id: Optional[str]) -> Path:
        key = db_id or self.default_id
        if key is None or key not in self._paths:
            raise KeyError(f"Unknown database '{db_id}'.")
        self._last_used[key] = time.time()  # in use: do not expire it
        return self._paths[key]

    def rescan(self) -> int:
        """Re-register uploads already on disk, e.g. after a restart.

        Only files named like ``add_upload`` names them are considered; anything
        else in the folder is left alone. Expired files are deleted by the
        ``cleanup_expired`` call that follows. Returns the number registered.
        """
        restored = 0
        for path in self.upload_dir.glob("*.db"):
            db_id = path.stem
            if _UPLOAD_ID.match(db_id) and db_id not in self._paths:
                self._paths[db_id] = path
                restored += 1
        self.cleanup_expired()
        return restored

    def _last_activity(self, db_id: str, path: Path) -> float:
        """Newer of the save time (mtime) and the last in-memory use."""
        return max(path.stat().st_mtime, self._last_used.get(db_id, 0.0))

    def cleanup_expired(self) -> int:
        """Delete uploads older than the TTL. Returns the number removed."""
        now = time.time()
        removed = 0
        for db_id, path in list(self._paths.items()):
            if db_id == "default":
                continue
            if not path.exists():  # deleted outside the app: stop tracking it
                del self._paths[db_id]
            elif now - self._last_activity(db_id, path) > self.ttl_s:
                path.unlink(missing_ok=True)
                del self._paths[db_id]
                removed += 1
            else:
                continue
            self._last_used.pop(db_id, None)
            if self.on_remove:
                self.on_remove(db_id)
        return removed
