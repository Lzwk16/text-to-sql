import asyncio
import os
import time

import pytest

from api import cleanup_loop
from utils.db_registry import DatabaseRegistry


def make_registry(up, **kw):
    return DatabaseRegistry(upload_dir=str(up), ttl_s=kw.pop("ttl_s", 3600), **kw)


def age(path, seconds, reg=None):
    """Make a file look old; pass ``reg`` to also forget its in-memory last use."""
    t = time.time() - seconds
    os.utime(path, (t, t))
    if reg is not None:
        reg._last_used.clear()


@pytest.fixture
def db_bytes(shop_db):
    return shop_db.read_bytes()


def test_uploads_survive_a_restart(tmp_path, db_bytes):
    up = tmp_path / "up"
    first = make_registry(up)
    db_id = first.add_upload("a.db", db_bytes)
    second = make_registry(up)  # new process: empty memory, same folder
    assert second.resolve(db_id) == up / f"{db_id}.db"


def test_expired_upload_is_deleted_at_startup(tmp_path, db_bytes):
    up = tmp_path / "up"
    first = make_registry(up)
    old = first.add_upload("a.db", db_bytes)
    fresh = first.add_upload("b.db", db_bytes)
    age(up / f"{old}.db", 7200)  # older than the 3600s TTL
    second = make_registry(up)
    assert not (up / f"{old}.db").exists()
    with pytest.raises(KeyError):
        second.resolve(old)
    assert second.resolve(fresh)


def test_scan_ignores_files_it_did_not_create(tmp_path, db_bytes):
    up = tmp_path / "up"
    make_registry(up)
    stray = up / "notes.db"
    stray.write_bytes(db_bytes)
    age(stray, 999999)
    reg = make_registry(up)
    assert stray.exists()  # not ours: never registered, never deleted
    with pytest.raises(KeyError):
        reg.resolve("notes")


def test_default_db_is_never_expired(tmp_path, shop_db):
    age(shop_db, 999999)
    reg = make_registry(tmp_path / "up", default_db=str(shop_db))
    assert shop_db.exists() and reg.resolve(None) == shop_db


def test_removal_notifies_cache_owner(tmp_path, db_bytes):
    up = tmp_path / "up"
    reg = make_registry(up)
    db_id = reg.add_upload("a.db", db_bytes)
    seen = []
    reg.on_remove = seen.append
    age(up / f"{db_id}.db", 7200, reg)
    assert reg.cleanup_expired() == 1
    assert seen == [db_id]


def test_background_loop_deletes_without_an_upload(tmp_path, db_bytes):
    up = tmp_path / "up"
    reg = make_registry(up)
    db_id = reg.add_upload("a.db", db_bytes)
    age(up / f"{db_id}.db", 7200, reg)

    async def run():
        task = asyncio.create_task(cleanup_loop(reg, 0.01))
        await asyncio.sleep(0.1)
        task.cancel()

    asyncio.run(run())
    assert not (up / f"{db_id}.db").exists()


def test_recently_used_upload_is_not_expired(tmp_path, db_bytes):
    up = tmp_path / "up"
    reg = make_registry(up)
    db_id = reg.add_upload("a.db", db_bytes)
    age(up / f"{db_id}.db", 7200, reg)  # saved long ago, not used since
    reg.resolve(db_id)  # a request uses it now
    assert reg.cleanup_expired() == 0
    assert (up / f"{db_id}.db").exists()


def test_unused_upload_still_expires(tmp_path, db_bytes):
    up = tmp_path / "up"
    reg = make_registry(up)
    db_id = reg.add_upload("a.db", db_bytes)
    age(up / f"{db_id}.db", 7200, reg)
    assert reg.cleanup_expired() == 1


def test_use_does_not_modify_the_file(tmp_path, db_bytes):
    up = tmp_path / "up"
    reg = make_registry(up)
    db_id = reg.add_upload("a.db", db_bytes)
    path = up / f"{db_id}.db"
    age(path, 7200, reg)
    before = (path.stat().st_mtime, path.read_bytes())
    reg.resolve(db_id)
    assert (path.stat().st_mtime, path.read_bytes()) == before
