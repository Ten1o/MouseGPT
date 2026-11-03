"""Credential destination isolation and real subprocess cancellation regressions."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import server
from process_control import process_group_options, terminate_process_tree


class CredentialDestinations(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = Path(self.temp.name) / 'settings.json'
        self.patches = [patch.object(server, 'SETTINGS', self.settings),
                        patch.dict(os.environ, {}, clear=True)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def shared(self, **changes):
        return {'provider': 'openai', 'endpoint': 'https://api.openai.com/v1',
                'model': 'gpt-4o', 'api_key': 'old-fixture-key', **changes}

    def test_fresh_interpreters_are_unselected_and_saved_settings_survive(self):
        with patch.object(server, 'ENVIRONMENTS', server.EnvironmentManager(Path(self.temp.name) / 'home')):
            self.assertTrue(all(value == '' for value in server.public_settings()['interpreters'].values()))
            self.assertTrue(all(profile['suggested_python'] == ''
                                for profile in server.environment_status()['profiles']))
            self.assertFalse(server.environment_access('describe')['ready'])
            fresh = server.read_settings()
            fresh['interpreters']['analysis'] = 'not-a-persisted-selection'
            self.assertEqual(server.read_settings()['interpreters']['analysis'], '')
            server.save_settings({'shared': self.shared(), 'interpreters': {'analysis': sys.executable}})
            before = self.settings.read_bytes()
            saved = server.public_settings()
            self.assertEqual(saved['interpreters']['analysis'], str(Path(sys.executable).absolute()))
            self.assertEqual(saved['interpreters']['pose'], '')
            self.assertTrue(saved['shared']['api_key_configured'])
            self.assertEqual(self.settings.read_bytes(), before)
            server.save_settings({'roles': {'VISION': {'model': 'another-model'}}})
            self.assertEqual(server.read_settings()['interpreters']['analysis'], str(Path(sys.executable).absolute()))
            self.assertEqual(server.read_settings()['shared']['api_key'], 'old-fixture-key')

    def test_shared_service_change_requires_fresh_key_and_is_atomic(self):
        server.save_settings({'shared': self.shared()})
        previous = self.settings.read_bytes()
        for blank in ('', '   '):
            with self.subTest(blank=blank), self.assertRaisesRegex(ValueError, 'new Default API key'):
                server.save_settings({'shared': self.shared(
                    provider='openai_compatible', endpoint='https://new.example/v1', api_key=blank)})
            self.assertEqual(self.settings.read_bytes(), previous)
        server.save_settings({'shared': self.shared(
            provider='openai_compatible', endpoint='https://new.example/v1', api_key='new-fixture-key')})
        selected = server.load_config('VISION', env=server.environment())
        self.assertEqual((selected.endpoint, selected.api_key),
                         ('https://new.example/v1', 'new-fixture-key'))

    def test_same_destination_retains_key_for_model_and_slash_changes(self):
        server.save_settings({'shared': self.shared()})
        server.save_settings({'shared': {'model': 'another-model', 'api_key': '',
                                         'endpoint': 'https://api.openai.com/v1/'}})
        selected = server.load_config('VISION', env=server.environment())
        self.assertEqual(selected.api_key, 'old-fixture-key')
        self.assertEqual(selected.model, 'another-model')

    def test_role_service_change_cannot_reuse_its_saved_key(self):
        role = {'provider': 'openai_compatible', 'endpoint': 'https://one.example/v1',
                'model': 'vision-model', 'api_key': 'old-role-fixture-key'}
        server.save_settings({'roles': {'VISION': role}})
        before = self.settings.read_bytes()
        with self.assertRaisesRegex(ValueError, 'new VISION API key'):
            server.save_settings({'roles': {'VISION': {'endpoint': 'https://two.example/v1', 'api_key': ''}}})
        self.assertEqual(self.settings.read_bytes(), before)
        server.save_settings({'roles': {'VISION': {'endpoint': 'https://two.example/v1', 'api_key': 'new-role-fixture-key'}}})
        selected = server.load_config('VISION', env=server.environment())
        self.assertEqual(selected.api_key, 'new-role-fixture-key')

    def test_inherited_destination_change_requires_refreshing_role_key(self):
        server.save_settings({'shared': self.shared(), 'roles': {
            'ANALYSIS': {'model': 'analysis-model', 'api_key': 'analysis-fixture-key'}}})
        before = self.settings.read_bytes()
        changed = self.shared(provider='openai_compatible', endpoint='https://new.example/v1',
                              api_key='new-shared-fixture-key')
        with self.assertRaisesRegex(ValueError, 'new ANALYSIS API key'):
            server.save_settings({'shared': changed})
        self.assertEqual(self.settings.read_bytes(), before)
        server.save_settings({'shared': changed, 'inherit': ['ANALYSIS']})
        selected = server.load_config('ANALYSIS', env=server.environment())
        self.assertEqual(selected.api_key, 'new-shared-fixture-key')

    def test_role_cannot_reuse_a_shell_key_for_a_new_service(self):
        env = {'MOUSEGPT_VISION_API_KEY': 'shell-fixture-key',
               'MOUSEGPT_VISION_MODEL': 'gpt-4o'}
        with patch.dict(os.environ, env), self.assertRaisesRegex(ValueError, 'new VISION API key'):
            server.save_settings({'roles': {'VISION': {'provider': 'openai_compatible',
                'endpoint': 'https://new.example/v1', 'model': 'vision-model', 'api_key': ''}}})
        self.assertFalse(self.settings.exists())

    def test_invalid_key_type_is_rejected_without_mutation(self):
        server.save_settings({'shared': self.shared()})
        before = self.settings.read_bytes()
        with self.assertRaisesRegex(ValueError, 'Invalid API setting'):
            server.save_settings({'roles': {'VISION': {'api_key': False}}})
        self.assertEqual(self.settings.read_bytes(), before)

    def test_incomplete_unused_model_does_not_block_interpreter_changes(self):
        self.settings.write_text(json.dumps({'roles': {'ANALYSIS': {
            'provider': 'openai_compatible', 'endpoint': 'https://one.example/v1',
            'api_key': 'fixture-key'}}}))
        server.save_settings({'interpreters': {'analysis': sys.executable}})
        saved = server.read_settings()
        self.assertEqual(saved['roles']['ANALYSIS']['api_key'], 'fixture-key')
        self.assertEqual(saved['interpreters']['analysis'], str(Path(sys.executable).absolute()))
        with self.assertRaisesRegex(ValueError, 'new ANALYSIS API key'):
            server.save_settings({'roles': {'ANALYSIS': {
                'endpoint': 'https://two.example/v1', 'model': 'configured-now', 'api_key': ''}}})


@unittest.skipIf(os.name == 'nt', 'POSIX process groups; Windows uses taskkill /T.')
class ProcessCancellation(unittest.TestCase):
    @staticmethod
    def child_command(folder):
        child = (
            'import signal,time;from pathlib import Path;'
            'signal.signal(signal.SIGTERM, signal.SIG_IGN);'
            f'Path({str(folder / "ready")!r}).write_text("ready");'
            'time.sleep(1.5);'
            f'Path({str(folder / "late-output")!r}).write_text("must not exist")'
        )
        parent = f'import subprocess,sys;subprocess.run([sys.executable,"-c",{child!r}],check=True)'
        return [sys.executable, '-c', parent]

    def wait_ready(self, folder):
        deadline = time.monotonic() + 5
        while not (folder / 'ready').exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue((folder / 'ready').exists(), 'Child did not start')

    def test_termination_kills_child_that_ignores_sigterm(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            process = subprocess.Popen(self.child_command(folder), **process_group_options())
            try:
                self.wait_ready(folder)
                terminate_process_tree(process, grace=0.05)
                self.assertIsNotNone(process.poll())
                time.sleep(1.6)
                self.assertFalse((folder / 'late-output').exists())
            finally:
                terminate_process_tree(process, grace=0.05)

    def test_cancel_releases_task_slot_after_descendants_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            job = {'folder': str(folder), 'op': 'filter', 'fields': {}, 'status': 'queued', 'log': []}
            with patch.object(server, 'worker_command', return_value=(self.child_command(folder), dict(os.environ))):
                self.assertTrue(server.BUSY.acquire(blocking=False))
                thread = threading.Thread(target=server.run_job, args=(job,), daemon=True)
                thread.start()
                try:
                    self.wait_ready(folder)
                    self.assertTrue(server.BUSY.locked())
                    server.cancel_job(job)
                    thread.join(5)
                    self.assertFalse(thread.is_alive())
                    self.assertEqual(job['status'], 'cancelled')
                    self.assertFalse(server.BUSY.locked())
                    self.assertNotIn('_process_lock', server.public_job(job))
                    time.sleep(0.6)
                    self.assertFalse((folder / 'late-output').exists())
                finally:
                    server.cancel_job(job)
                    thread.join(5)

    def test_cancel_queued_task_never_starts_a_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = {'folder': tmp, 'op': 'filter', 'fields': {}, 'status': 'queued', 'log': []}
            server.cancel_job(job)
            with patch.object(server, 'worker_command', return_value=([], {})), patch.object(server.subprocess, 'Popen') as spawn:
                self.assertTrue(server.BUSY.acquire(blocking=False))
                server.run_job(job)
            spawn.assert_not_called()
            self.assertEqual(job['status'], 'cancelled')
            self.assertFalse(server.BUSY.locked())


if __name__ == '__main__':
    unittest.main()
