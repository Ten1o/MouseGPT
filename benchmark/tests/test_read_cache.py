"""Response reuse must remain fresh across external workers and concurrent reads."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest

from mousebench.read_cache import ReadResponseCache


class ReadCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.writer = sqlite3.connect(self.root / 'benchmark.sqlite3')
        self.writer.execute('PRAGMA journal_mode=WAL')
        self.writer.execute('CREATE TABLE sample(value INTEGER)')
        self.writer.execute('INSERT INTO sample VALUES(1)')
        self.writer.commit()
        self.cache = ReadResponseCache(self.root, max_entries=2, max_bytes=10)

    def tearDown(self):
        self.cache.close()
        self.writer.close()
        self.temp.cleanup()

    def test_external_commit_invalidates_cached_results(self):
        calls = []
        def compute():
            calls.append(1)
            return str(self.writer.execute('SELECT value FROM sample').fetchone()[0]).encode()
        self.assertEqual(self.cache.get('summary', compute), b'1')
        self.assertEqual(self.cache.get('summary', compute), b'1')
        self.assertEqual(len(calls), 1)
        self.writer.execute('UPDATE sample SET value=2')
        self.writer.commit()
        self.assertEqual(self.cache.get('summary', compute), b'2')
        self.assertEqual(len(calls), 2)

    def test_concurrent_identical_requests_compute_once(self):
        barrier = threading.Barrier(5)
        started, release = threading.Event(), threading.Event()
        calls = []
        def compute():
            calls.append(1)
            started.set()
            self.assertTrue(release.wait(5))
            return b'answer'
        def request():
            barrier.wait(5)
            return self.cache.get('review', compute)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(request) for _ in range(4)]
            barrier.wait(5)
            self.assertTrue(started.wait(5))
            release.set()
            self.assertEqual([future.result(5) for future in futures], [b'answer'] * 4)
        self.assertEqual(len(calls), 1)

    def test_commit_during_computation_prevents_stale_publication(self):
        started, release = threading.Event(), threading.Event()
        def old_snapshot():
            started.set()
            self.assertTrue(release.wait(5))
            return b'old'
        with ThreadPoolExecutor(max_workers=1) as pool:
            old = pool.submit(self.cache.get, 'overview', old_snapshot)
            self.assertTrue(started.wait(5))
            self.writer.execute('UPDATE sample SET value=2')
            self.writer.commit()
            self.assertEqual(self.cache.get('overview', lambda: b'new'), b'new')
            release.set()
            self.assertEqual(old.result(5), b'old')
        self.assertEqual(self.cache.get('overview', lambda: self.fail('New result was replaced')), b'new')

    def test_errors_are_not_reused_and_storage_is_bounded(self):
        def fail():
            raise ValueError('temporary failure')
        with self.assertRaises(ValueError):
            self.cache.get('a', fail)
        self.assertEqual(self.cache.get('a', lambda: b'aaaaaa'), b'aaaaaa')
        self.cache.get('b', lambda: b'bbbbbb')
        self.assertEqual(list(self.cache.entries), ['b'])
        self.assertLessEqual(self.cache.size, 10)
        self.cache.get('huge', lambda: b'x' * 11)
        self.assertNotIn('huge', self.cache.entries)


if __name__ == '__main__':
    unittest.main()
