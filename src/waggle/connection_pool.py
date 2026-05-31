"""A small, thread-safe SQLite connection pool.

Every graph operation used to call ``MemoryGraph._connect()``, which creates a
fresh ``sqlite3.Connection``, sets ``row_factory``, and runs seven ``PRAGMA``
statements. With WAL mode SQLite supports concurrent readers plus a single
writer, so connections can be safely reused, skipping that per-call setup cost.

``ConnectionPool`` pre-creates a small number of fully-configured connections
and hands them out through a context manager:

    with pool.checkout() as conn:
        conn.execute(...)

The ``checkout`` context manager mirrors the semantics of using a
``sqlite3.Connection`` as a context manager: it commits on success and rolls
back on error. On exit the connection is returned to the pool instead of being
closed, so the next caller reuses it.

Design notes
------------
- **Bounded.** At most ``size`` connections are *retained*. SQLite WAL allows a
  single writer regardless of pool size, so a large pool only wastes file
  handles; the default of 4 is plenty.
- **Overflow instead of blocking.** If every pooled connection is already
  checked out (nested or concurrent callers), ``checkout`` creates a temporary
  connection rather than blocking. That temporary connection is closed on
  return once the retained pool is full, so the pool can never deadlock and the
  retained count stays bounded.
- **Cross-thread reuse.** Pooled connections are created with
  ``check_same_thread=False`` (the factory's responsibility) because callers may
  run on different worker threads. The pool guarantees a connection is only ever
  handed to one caller at a time, so this is safe.
- **Self-healing.** A connection that errors out is discarded rather than
  returned, so a connection left in a bad transaction state is never reused; the
  next exhausted checkout transparently creates a replacement.
"""

from __future__ import annotations

import contextlib
import queue
import sqlite3
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager

DEFAULT_POOL_SIZE = 4


class ConnectionPool:
    """A bounded, thread-safe pool of pre-configured SQLite connections."""

    def __init__(self, factory: Callable[[], sqlite3.Connection], *, size: int = DEFAULT_POOL_SIZE) -> None:
        if size < 1:
            raise ValueError("Connection pool size must be at least 1.")
        self._factory = factory
        self._size = size
        self._idle: queue.Queue[sqlite3.Connection] = queue.Queue(maxsize=size)
        self._lock = threading.Lock()
        self._closed = False
        self._created = 0
        # Pre-create the connections eagerly so the PRAGMA setup happens once,
        # up front, rather than on the first checkout.
        for _ in range(size):
            self._idle.put_nowait(self._new_connection())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @contextmanager
    def checkout(self) -> Iterator[sqlite3.Connection]:
        """Borrow a connection for the duration of the ``with`` block.

        Commits on success, rolls back on error, and returns the connection to
        the pool on exit (closing it if it was a temporary overflow connection).
        """
        if self._closed:
            raise RuntimeError("ConnectionPool is closed.")

        try:
            connection = self._idle.get_nowait()
        except queue.Empty:
            # Pool exhausted by nested or concurrent checkouts — make a
            # temporary connection so the caller never blocks or deadlocks.
            connection = self._new_connection()

        try:
            yield connection
        except BaseException:
            # Roll back and discard: never hand a possibly-poisoned connection
            # back to the next caller.
            self._rollback_quietly(connection)
            self._close_quietly(connection)
            raise
        else:
            try:
                connection.commit()
            except Exception:
                self._close_quietly(connection)
                raise
            self._return_to_pool(connection)

    def close(self) -> None:
        """Close every retained connection. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        while True:
            try:
                connection = self._idle.get_nowait()
            except queue.Empty:
                break
            self._close_quietly(connection)

    # ------------------------------------------------------------------
    # Introspection (used by tests)
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Maximum number of connections the pool retains."""
        return self._size

    @property
    def idle_count(self) -> int:
        """Number of connections currently available for checkout (approximate)."""
        return self._idle.qsize()

    @property
    def created_count(self) -> int:
        """Total connections ever created, including temporary overflow ones."""
        with self._lock:
            return self._created

    @property
    def closed(self) -> bool:
        return self._closed

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _new_connection(self) -> sqlite3.Connection:
        connection = self._factory()
        with self._lock:
            self._created += 1
        return connection

    def _return_to_pool(self, connection: sqlite3.Connection) -> None:
        if self._closed:
            self._close_quietly(connection)
            return
        try:
            self._idle.put_nowait(connection)
        except queue.Full:
            # Retained pool already at capacity — this was an overflow
            # connection, so close it rather than exceeding the bound.
            self._close_quietly(connection)

    @staticmethod
    def _rollback_quietly(connection: sqlite3.Connection) -> None:
        with contextlib.suppress(Exception):
            connection.rollback()

    @staticmethod
    def _close_quietly(connection: sqlite3.Connection) -> None:
        with contextlib.suppress(Exception):
            connection.close()