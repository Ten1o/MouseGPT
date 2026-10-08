"""Bounded read-response reuse, invalidated by commits from any SQLite writer."""
from collections import OrderedDict
from concurrent.futures import Future
from pathlib import Path
import sqlite3
import threading


class ReadResponseCache:
    def __init__(self, data_dir, *, max_entries=64, max_bytes=32 * 1024 * 1024):
        self.path = Path(data_dir).resolve() / 'benchmark.sqlite3'
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self.lock = threading.RLock()
        self.connection = None
        self.identity = None
        self.version = None
        self.entries = OrderedDict()
        self.pending = {}
        self.size = 0

    def _version(self):
        stat = self.path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if self.connection is None or identity != self.identity:
            if self.connection is not None:
                self.connection.close()
            self.connection = sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True,
                                             isolation_level=None, check_same_thread=False)
            self.identity = identity
            self.version = None
        version = (identity, self.connection.execute('PRAGMA data_version').fetchone()[0])
        if version != self.version:
            self.entries.clear()
            self.size = 0
            self.version = version
        return version

    def get(self, key, compute):
        """Return immutable serialized bytes; concurrent identical reads share work.

        Individual evidence, images, and write validation do not use this cache.
        A write during computation prevents publishing that result to the cache.
        """
        with self.lock:
            version = self._version()
            if key in self.entries:
                self.entries.move_to_end(key)
                return self.entries[key]
            pending_key = (version, key)
            future = self.pending.get(pending_key)
            owner = future is None
            if owner:
                future = Future()
                self.pending[pending_key] = future
        if not owner:
            return future.result()
        try:
            result = compute()
            if not isinstance(result, bytes):
                raise TypeError('Read response cache stores only serialized bytes')
            with self.lock:
                if self._version() == version and len(result) <= self.max_bytes:
                    self.entries[key] = result
                    self.size += len(result)
                    while len(self.entries) > self.max_entries or self.size > self.max_bytes:
                        _, removed = self.entries.popitem(last=False)
                        self.size -= len(removed)
                self.pending.pop(pending_key, None)
                future.set_result(result)
            return result
        except BaseException as exc:
            with self.lock:
                self.pending.pop(pending_key, None)
                future.set_exception(exc)
            raise

    def close(self):
        with self.lock:
            if self.connection is not None:
                self.connection.close()
                self.connection = None
            self.entries.clear()
            self.size = 0
