import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import run_jobs
from mousebench.store import digest

class JobTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        (self.root/'image.jpg').write_bytes(b'fixture')
        sha=hashlib.sha256(b'fixture').hexdigest()
        payload={'messages':[],'attachments':[{'attachment_id':'canonical_image','sha256':sha}]}
        self.job={'frame_id':'240420_C10M1_00003972','task':'inference','payload_template':payload,'payload_sha256':digest(payload),
            'attachment_manifest':[{'path':'image.jpg','attachment_id':'canonical_image','sha256':sha}]}
    def tearDown(self):self.temp.cleanup()
    def test_preview_checksum_and_attachment_integrity(self):
        p=self.root/'jobs.json';p.write_text(json.dumps({'jobs':[self.job]}))
        self.assertEqual(len(run_jobs.read_jobs(p)),1)
        self.assertTrue(run_jobs.validate_attachments([self.job],self.root))
        (self.root/'image.jpg').write_bytes(b'changed')
        with self.assertRaises(ValueError):run_jobs.validate_attachments([self.job],self.root)
    def test_dry_run_never_loads_adapter(self):
        p=self.root/'jobs.json';p.write_text(json.dumps([self.job]))
        # Resolve builtins before replacing importlib's process-wide function.
        with patch('sys.argv',['run_jobs.py',str(p),'--adapter','missing_adapter']),patch('builtins.print'),patch('run_jobs.importlib.import_module') as module:
            run_jobs.main();module.assert_not_called()
    def test_failed_adapter_is_logged_once_and_returns_failure(self):
        from types import SimpleNamespace
        p=self.root/'jobs.json';p.write_text(json.dumps([self.job]));out=self.root/'log.jsonl'
        calls=[]
        def fail(job):
            calls.append(job)
            raise RuntimeError('fixture failure')
        args=['run_jobs.py',str(p),'--execute','--adapter','fixture_adapter','--output',str(out),'--data',str(self.root)]
        with patch('sys.argv',args),patch('builtins.print'),patch('run_jobs.importlib.import_module',return_value=SimpleNamespace(execute=fail)):
            with self.assertRaises(SystemExit) as exc:run_jobs.main()
        self.assertEqual(exc.exception.code,1)
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0]['execution_context']['data_directory'],str(self.root.resolve()))
        rows=[json.loads(line) for line in out.read_text().splitlines()]
        self.assertEqual([r['event'] for r in rows],['attempt','result'])
        self.assertEqual(rows[-1]['status'],'failed')

    def test_tampered_payload_rejected(self):
        self.job['payload_template']['messages']=['modified']
        p=self.root/'jobs.json';p.write_text(json.dumps([self.job]))
        with self.assertRaises(ValueError):run_jobs.read_jobs(p)

if __name__=='__main__':unittest.main()
