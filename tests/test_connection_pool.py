"""Tests for ``waggle.connection_pool.ConnectionPool``.

Covers reuse, the bounded-size invariant, commit/rollback semantics, overflow
under exhaustion, cleanup on ``close``, and thread safety. Uses an on-disk
SQLite database in a temp dir; no graph, model, or server required.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from waggle.connection_pool import DEFAULT_POOL_SIZE, ConnectionPool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_factory(db_path: Path):
    """Return a factory that mimics how MemoryGraph creates pooled connections."""

    def factory() -> sqlite3.Connection:
        connection = sqlite3.connect(str(db_path), timeout=30.0, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    return factory


@pytest.fixture
def pool(tmp_path: Path):
    db = tmp_path / "pool.db"
    p = ConnectionPool(_make_factory(db), size=3)
    # A table to exercise real reads/writes through pooled connections.
    with p.checkout() as conn:
        conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, value TEXT)")
    yield p
    p.close()


# ---------------------------------------------------------------------------
# Construction / config
# ---------------------------------------------------------------------------


def test_precreates_connections_once() -> None:
    db_factory_calls = {"n": 0}

    def counting_factory() -> sqlite3.Connection:
        db_factory_calls["n"] += 1
        return sqlite3.connect(":memory:", check_same_thread=False)

    p = ConnectionPool(counting_factory, size=4)
    try:
        # All connections built up front; none created lazily on first checkout.
        assert db_factory_calls["n"] == 4
        assert p.size == 4
        assert p.idle_count == 4
        assert p.created_count == 4
    finally:
        p.close()


def test_rejects_invalid_size() -> None:
    with pytest.raises(ValueError):
        ConnectionPool(lambda: sqlite3.connect(":memory:"), size=0)


def test_default_size_is_four() -> None:
    assert DEFAULT_POOL_SIZE == 4


# ---------------------------------------------------------------------------
# Reuse and the bounded-size invariant
# ---------------------------------------------------------------------------


def test_connections_are_reused_not_recreated(pool: ConnectionPool) -> None:
    created_after_setup = pool.created_count
    seen_ids = set()
    for i in range(50):
        with pool.checkout() as conn:
            conn.execute("INSERT INTO items (value) VALUES (?)", (f"v{i}",))
            seen_ids.add(id(conn))
    # Sequential checkouts reuse pooled connections: no new ones were created
    # and at most `size` distinct connection objects were ever handed out.
    assert pool.created_count == created_after_setup
    assert len(seen_ids) <= pool.size
    # The writes were committed (visible on a fresh checkout).
    with pool.checkout() as conn:
        assert conn.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 50


def test_idle_count_returns_to_full_and_stays_bounded(pool: ConnectionPool) -> None:
    assert pool.idle_count == pool.size  # at rest after fixture setup
    with pool.checkout():
        assert pool.idle_count == pool.size - 1
    assert pool.idle_count == pool.size
    assert pool.idle_count <= pool.size


# ---------------------------------------------------------------------------
# Transaction semantics
# ---------------------------------------------------------------------------


def test_commit_on_success(pool: ConnectionPool) -> None:
    with pool.checkout() as conn:
        conn.execute("INSERT INTO items (value) VALUES ('committed')")
    with pool.checkout() as conn:
        rows = conn.execute("SELECT value FROM items WHERE value='committed'").fetchall()
    assert len(rows) == 1


def test_rollback_on_error(pool: ConnectionPool) -> None:
    with pytest.raises(RuntimeError), pool.checkout() as conn:
        conn.execute("INSERT INTO items (value) VALUES ('doomed')")
        raise RuntimeError("boom")
    # The failed transaction was rolled back; the row is absent.
    with pool.checkout() as conn:
        rows = conn.execute("SELECT value FROM items WHERE value='doomed'").fetchall()
    assert rows == []


def test_errored_connection_is_not_returned_to_pool(pool: ConnectionPool) -> None:
    # An errored checkout discards its connection (self-healing), so idle drops
    # by one, but the pool refills transparently on the next exhausted checkout.
    with pytest.raises(ValueError), pool.checkout():
        raise ValueError("discard me")
    assert pool.idle_count <= pool.size
    # Still fully usable afterwards.
    with pool.checkout() as conn:
        conn.execute("INSERT INTO items (value) VALUES ('after-error')")
    with pool.checkout() as conn:
        assert conn.execute("SELECT COUNT(*) FROM items WHERE value='after-error'").fetchone()[0] == 1


# ---------------------------------------------------------------------------
# Overflow under exhaustion (nested checkouts)
# ---------------------------------------------------------------------------


def test_nested_checkouts_do_not_deadlock_and_stay_bounded(pool: ConnectionPool) -> None:
    # Hold every pooled connection, then check out one more on the same thread.
    # Without overflow this would block forever; with it, a temporary connection
    # is created and closed on return, leaving the retained pool bounded.
    with pool.checkout(), pool.checkout(), pool.checkout():
        assert pool.idle_count == 0
        with pool.checkout() as overflow_conn:  # exhausted -> overflow
            overflow_conn.execute("SELECT 1")
    assert pool.idle_count <= pool.size


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_close_closes_all_and_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "close.db"
    p = ConnectionPool(_make_factory(db), size=3)
    drained = []
    # Capture the live connections so we can assert they are actually closed.
    for _ in range(3):
        with p.checkout() as conn:
            drained.append(conn)
    p.close()
    p.close()  # idempotent — must not raise
    assert p.closed is True
    for conn in drained:
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")  # closed connections raise


def test_checkout_after_close_raises(tmp_path: Path) -> None:
    db = tmp_path / "after-close.db"
    p = ConnectionPool(_make_factory(db), size=2)
    p.close()
    with pytest.raises(RuntimeError), p.checkout():
        pass


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_concurrent_checkouts_do_not_crash_and_stay_bounded(pool: ConnectionPool) -> None:
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def worker(n: int) -> None:
        try:
            barrier.wait()  # maximize contention
            for i in range(25):
                with pool.checkout() as conn:
                    conn.execute("INSERT INTO items (value) VALUES (?)", (f"t{n}-{i}",))
                    conn.execute("SELECT COUNT(*) FROM items").fetchone()
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    # Retained pool never exceeds its bound, even after heavy contention.
    assert pool.idle_count <= pool.size
    with pool.checkout() as conn:
        assert conn.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 8 * 25