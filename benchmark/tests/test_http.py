"""Local HTTP integration tests using a fresh temporary benchmark per test.

No production data, NAS access, image downloads or model providers are used.
"""

import copy
import gzip
import hashlib
import http.client
import json
import tempfile
import threading
import unittest
from unittest.mock import patch
from http.server import ThreadingHTTPServer
from pathlib import Path

import app
from mousebench.store import Store, FIELDS

FRAME = "240420_C10M1_00003972"


class QuietHandler(app.Handler):
    def log_message(self, *args):
        pass


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="mousebench-http-test-")
        self.data_dir = Path(self.temp.name)
        store = Store(self.data_dir)
        store.import_catalog({
            "frames": [{"id": FRAME, "frame_id": "00003972", "recording_id": "240420_C10M1"}],
            "assets": [], "ratings": [{"id": "rating-fixture", "frame_id": FRAME, "source": "fixture.xlsx#D3", "score": 4}],
        })
        store.register_image(FRAME, b"test-fixture-not-a-real-scientific-image", "/fixture/source.jpg")
        store.close()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), QuietHandler)
        self.server.data_dir = self.data_dir
        self.server.csrf_token = "isolated-test-token"
        self.server.read_cache = app.ReadResponseCache(self.data_dir)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .02}, daemon=True)
        self.thread.start()
        self.content = {
            "facts": [{"id": "f1", "text": "Fixture fact, not a scientific label.",
                       "field": "Overall", "evidence": "image", "required": True}],
            "sections": {field: "Test fixture only." for field in FIELDS}, "uncertainties": [],
        }

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.server.read_cache.close()
        self.temp.cleanup()

    def test_cached_summary_observes_external_database_commits(self):
        self.assertEqual(self.request('GET', '/api/summary')[1]['frames'], 1)
        self.assertEqual(self.request('GET', '/api/summary')[1]['frames'], 1)
        with patch.object(Store, 'summary', side_effect=AssertionError('Should reuse unchanged summary')):
            self.assertEqual(self.request('GET', '/api/summary')[0], 200)
        other = Store(self.data_dir)
        try:
            other.import_catalog({'frames': [{'id': 'other-frame', 'frame_id': 'other-frame'}]})
        finally:
            other.close()
        self.assertEqual(self.request('GET', '/api/summary')[1]['frames'], 2)

    def test_json_compression_preserves_response_and_respects_opt_out(self):
        expected = self.request('GET', '/api/workflow')[1]
        conn = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        try:
            conn.request('GET', '/api/workflow', headers={'Accept-Encoding': 'gzip'})
            response = conn.getresponse()
            self.assertEqual(response.getheader('Content-Encoding'), 'gzip')
            self.assertEqual(response.getheader('Vary'), 'Accept-Encoding')
            self.assertEqual(json.loads(gzip.decompress(response.read())), expected)
        finally:
            conn.close()
        status, body, headers = self.request('GET', '/api/workflow', extra_headers={'Accept-Encoding': 'gzip;q=0'})
        self.assertEqual(status, 200)
        self.assertNotIn('Content-Encoding', headers)
        self.assertEqual(body, expected)

    def request(self, method, path, body=None, token=True, extra_headers=None):
        headers = {}
        if method == "POST" and token:
            headers["X-Mousebench-Token"] = self.server.csrf_token
        if body is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(body).encode()
        headers.update(extra_headers or {})
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        try:
            conn.request(method, path, body=body, headers=headers)
            response = conn.getresponse()
            data, received = response.read(), dict(response.getheaders())
            if received.get("Content-Type", "").startswith("application/json"):
                data = json.loads(data)
            return response.status, data, received
        finally:
            conn.close()

    def freeze_input(self, prompt="Describe the fixture in six fields.", revision=0):
        status, data, _ = self.request("POST", f"/api/frames/{FRAME}/input-contract", {
            "prompt": prompt, "reviewer": "HTTP test", "expected_revision": revision,
        })
        self.assertEqual(status, 200, data)
        return data

    def save_gt(self, state, revision, content=None):
        return self.request("POST", f"/api/frames/{FRAME}/ground-truth", {
            "status": state, "human_confirmed": state=="approved", "expected_revision": revision, "reviewer": "HTTP test",
            "content": content or self.content,
        })

    def approve(self):
        contract = self.freeze_input()
        self.assertEqual(self.save_gt("draft", 0)[0], 200)
        self.assertEqual(self.save_gt("reviewed", 1)[0], 200)
        status, gt, _ = self.save_gt("approved", 2)
        self.assertEqual(status, 200, gt)
        return contract, gt

    def import_run(self, output="Fixture answer.", **record_metadata):
        status, run, _ = self.request("POST", "/api/runs/import", {
            "name": "HTTP fixture run", "model": "declared-test-model", "track": "historical",
            "records": [{"frame_id": FRAME, "output": output, **record_metadata}],
        })
        self.assertEqual(status, 200, run)
        return run

    def make_judgment(self, run, contract, gt):
        status, records, _ = self.request("GET", "/api/runs/" + run["id"])
        self.assertEqual(status, 200)
        return {"schema_version": "mousebench.judgment.v1", "frame_id": FRAME, "run_id": run["id"],
                "protocol_version": "http-fixture-v1", "judge": {"provider": "fixture", "model": "fixture", "version": "1"},
                "rubric_sha256": "a" * 64, "input_sha256": contract["input_hash"],
                "ground_truth_sha256": gt["content_sha256"], "prediction_sha256": records[0]["prediction_sha256"],
                "claims": [{"text": "Fixture fact.", "label": "supported"}], "supported_fact_ids": ["f1"],
                "structure_score": 1, "instruction_score": 1}

    def test_readonly_endpoints_security_headers_and_no_extra_writes(self):
        for path in ("/api/summary", "/api/overview", "/api/frames", "/api/frames/" + FRAME, "/api/runs", "/api/scorecards"):
            status, _, headers = self.request("GET", path)
            self.assertEqual(status, 200, path)
            self.assertEqual(headers["Cache-Control"], "no-store")
            self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
        store = Store(self.data_dir)
        self.assertEqual(store.versions("gt", FRAME), [])
        self.assertEqual(store.versions("input", FRAME), [])
        store.close()

    def test_import_recorded_input_and_serve_exact_view_bytes(self):
        import base64
        from test_recorded_inputs import PROMPT
        blob = b'\xff\xd8\xfffixture-view\xff\xd9'
        source = {'messages':[{'role':'system','content':'Original fixture system.\n'},
                  {'role':'user','content':[{'type':'text','text':PROMPT},
                  *[{'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(blob).decode(),'detail':'high'}} for _ in range(2)]]}]}
        raw=json.dumps(source).encode();sha=hashlib.sha256(raw).hexdigest()
        path='objects/'+sha+'.json';(self.data_dir/path).write_bytes(raw)
        store=Store(self.data_dir)
        with store.transaction():
            store.db.execute('INSERT INTO records VALUES(?,?,?,?)',('raw_records','http-source',FRAME,json.dumps({
                'id':'http-source','frame_id':FRAME,'frame_membership_verified':True,'role':'inference_log',
                'sha256':sha,'path':path,'nas_path':'/fixture/log.json'})))
        store.close()
        status, result, _=self.request('POST','/api/workflow/action',{'action':'freeze_recorded_inputs','frame_ids':[FRAME],'reviewer':'Fixture importer'})
        self.assertEqual(status,200,result)
        self.assertEqual(result['saved'],1)
        status, detail, _=self.request('GET','/api/frames/'+FRAME)
        self.assertEqual(detail['input_contract']['prompt'],PROMPT)
        status, image, headers=self.request('GET',f'/api/input-image/{FRAME}/input_image_1')
        self.assertEqual(status,200);self.assertEqual(image,blob)
        self.assertEqual(headers['Content-Type'],'image/jpeg')
        self.assertEqual(self.request('GET',f'/api/input-image/{FRAME}/input_image_9')[0],404)
        (self.data_dir/detail['input_contract']['input_images'][0]['path']).write_bytes(b'damaged')
        self.assertEqual(self.request('GET',f'/api/input-image/{FRAME}/input_image_1')[0],400)

    def test_csrf_and_foreign_host_origin_are_rejected(self):
        status, session, _ = self.request("GET", "/api/session")
        self.assertEqual(status, 200)
        self.assertEqual(session["token"], self.server.csrf_token)
        path = f"/api/frames/{FRAME}/input-contract"
        body = {"prompt": "fixture", "reviewer": "test"}
        self.assertEqual(self.request("POST", path, body, token=False)[0], 403)
        self.assertEqual(self.request("POST", path, body, extra_headers={"Origin": "https://untrusted.example"})[0], 403)
        self.assertEqual(self.request("GET", "/api/session", extra_headers={"Host": "untrusted.example"})[0], 403)
        store = Store(self.data_dir)
        self.assertIsNone(store.latest("input", FRAME))
        store.close()

    def test_gt_approval_history_conflict_and_input_change(self):
        self.freeze_input()
        self.assertEqual(self.save_gt("approved", 0)[0], 400)
        self.assertEqual(self.save_gt("draft", 0)[0], 200)
        self.assertEqual(self.save_gt("reviewed", 1)[0], 200)
        self.assertEqual(self.save_gt("approved", 1)[0], 409)
        self.assertEqual(self.save_gt("approved", 2)[0], 200)
        self.freeze_input("Changed task instructions.", 1)
        status, _, _ = self.request("POST", "/api/jobs/preview", {"task": "judge", "frame_ids": [FRAME]})
        self.assertEqual(status, 400)
        status, detail, _ = self.request("GET", "/api/frames/" + FRAME)
        self.assertEqual(status, 200)
        self.assertEqual(len(detail["history"]), 3)
        self.assertEqual(detail["history"][0]["status"], "approved")

    def test_one_click_approval_with_optional_identity_is_one_versioned_write(self):
        contract = self.freeze_input()
        self.assertEqual(self.save_gt("draft", 0)[0], 200)
        payload = {
            "status": "approved", "actor_type": "human", "human_confirmed": True,
            "approval_action": "approve_reference", "expected_revision": 1,
            "expected_input_hash": contract["input_hash"], "reviewer": "", "notes": "",
            "content": self.content,
        }
        status, approved, _ = self.request("POST", f"/api/frames/{FRAME}/ground-truth", payload)
        self.assertEqual(status, 200, approved)
        self.assertEqual(approved["status"], "approved")
        self.assertEqual(approved["revision"], 2)
        self.assertEqual(approved["reviewer"], "")
        self.assertEqual(approved["content"], self.content)
        self.assertTrue(approved["provenance"]["human_confirmed"])
        # Replaying a stale click must not add another approved revision.
        self.assertEqual(self.request("POST", f"/api/frames/{FRAME}/ground-truth", payload)[0], 409)
        status, detail, _ = self.request("GET", "/api/frames/" + FRAME)
        self.assertEqual(status, 200)
        self.assertEqual([row["status"] for row in detail["history"]], ["approved", "draft"])

    def test_versions_and_prediction_exports_preserve_newlines_and_provenance(self):
        self.approve()
        answer = 'Line one.\nLine two with "quotes".'
        run = self.import_run(answer, record_kind="prediction", source_path="/fixture/answer.txt")
        status, payload, headers = self.request("GET", "/api/export/runs")
        self.assertEqual(status, 200, payload)
        self.assertIn("application/x-ndjson", headers["Content-Type"])
        exported = [json.loads(line) for line in payload.splitlines()]
        predictions = [row for row in exported if row["record_kind"] == "prediction"]
        self.assertEqual(predictions[0]["output"], answer)
        self.assertEqual(predictions[0]["source_path"], "/fixture/answer.txt")
        status, payload, _ = self.request("GET", "/api/export/ground-truth")
        self.assertEqual(status, 200)
        self.assertEqual(len([json.loads(line) for line in payload.splitlines()]), 4)

    def test_judgment_import_scorecard_and_stale_exclusion_end_to_end(self):
        contract, gt = self.approve()
        run = self.import_run()
        j = self.make_judgment(run, contract, gt)
        status, _, _ = self.request("POST", "/api/judgments/import", {"records": [dict(j, prediction_sha256="f" * 64)]})
        self.assertEqual(status, 400)
        status, result, _ = self.request("POST", "/api/judgments/import", {"records": [j]})
        self.assertEqual(status, 200, result)
        status, scorecards, _ = self.request("GET", "/api/scorecards")
        self.assertEqual(status, 200, scorecards)
        self.assertTrue(scorecards["provisional"])
        self.assertEqual(scorecards["models"][0]["dimension_means"]["factual_f1"], 1)
        self.assertEqual(scorecards["models"][0]["benchmark_coverage"], 1)
        self.save_gt("draft", 3)
        status, scorecards, _ = self.request("GET", "/api/scorecards")
        self.assertEqual(status, 200)
        self.assertEqual(scorecards["stale_judgments_excluded"], 1)
        self.assertEqual(scorecards["models"], [])

    def test_preview_remains_disabled_and_uses_canonical_image_and_id(self):
        self.approve()
        run = self.import_run()
        status, preview, _ = self.request("POST", "/api/jobs/preview", {"task": "judge", "frame_ids": [FRAME], "run_id": run["id"]})
        self.assertEqual(status, 200, preview)
        self.assertFalse(preview["execution_enabled"])
        self.assertEqual(preview["network_requests"], 0)
        self.assertEqual(preview["jobs"][0]["frame_id"], FRAME)
        self.assertTrue(preview["jobs"][0]["attachment_manifest"])
        blind = preview["jobs"][0]["payload_template"]
        self.assertNotIn("declared-test-model", json.dumps(blind))
        self.assertNotIn("fixture.xlsx", json.dumps(blind))
        self.assertNotIn("/fixture/source.jpg", json.dumps(blind))

    def test_different_judge_configurations_are_separate_groups(self):
        contract, gt = self.approve()
        run = self.import_run()
        j = self.make_judgment(run, contract, gt)
        other = copy.deepcopy(j)
        other["judge"]["version"] = "2"
        status, result, _ = self.request("POST", "/api/judgments/import", {"records": [j, other]})
        self.assertEqual(status, 200, result)
        status, cards, _ = self.request("GET", "/api/scorecards")
        self.assertEqual(status, 200, cards)
        self.assertEqual(len(cards["groups"]), 2)

    def test_conflicting_batch_rolls_back_without_importing_partial_scores(self):
        contract, gt = self.approve()
        run = self.import_run()
        j = self.make_judgment(run, contract, gt)
        other = dict(j, instruction_score=.5)
        status, _, _ = self.request("POST", "/api/judgments/import", {"records": [j, other]})
        self.assertEqual(status, 400)
        status, data, _ = self.request("GET", "/api/export/judgments")
        self.assertEqual(status, 200)
        self.assertEqual(data, b"")

    def test_traversal_and_unknown_resource_are_not_exposed(self):
        self.assertEqual(self.request("GET", "/%2e%2e/mousebench/store.py")[0], 404)
        self.assertEqual(self.request("GET", "/api/frames/unknown")[0], 404)
        self.assertEqual(self.request("GET", "/api/export/unknown")[0], 404)
        self.assertEqual(self.request("POST", "/api/execute", {})[0], 404)

    def workflow_action(self, action, **payload):
        status, value, _ = self.request("POST", "/api/workflow/action", dict(payload, action=action))
        self.assertEqual(status, 200, value)
        return value

    def workflow_rubric(self):
        status, summary, _ = self.request("GET", "/api/workflow")
        self.assertEqual(status, 200, summary)
        saved = self.workflow_action("save_rubric", content=summary["default_rubric"], author="HTTP fixture researcher")
        return self.workflow_action("freeze_rubric", id=saved["id"], reviewer="HTTP fixture researcher")

    def test_workflow_summary_is_offline_and_scientific_records_remain_empty(self):
        with patch("mousebench.api_providers.execute_job") as execute:
            status, summary, headers = self.request("GET", "/api/workflow")
            self.assertEqual(status, 200, summary)
            self.assertEqual(summary["readiness"]["frames"], 1)
            self.assertEqual(summary["readiness"]["input_contracts"], 0)
            self.assertEqual(summary["readiness"]["gt_approved"], 0)
            self.assertTrue(summary["default_prompt"])
            self.assertTrue(summary["default_rubric"]["scoring"]["index"])
            self.assertEqual(summary["batches"], [])
            self.assertEqual(summary["live_execution"], "cli_only")
            self.assertEqual(headers["Cache-Control"], "no-store")
            status, issues, _ = self.request("GET", "/api/workflow/issues")
            self.assertEqual((status, issues), (200, []))
        execute.assert_not_called()
        store = Store(self.data_dir)
        self.assertEqual(store.versions("input", FRAME), [])
        self.assertEqual(store.versions("gt", FRAME), [])
        store.close()

    def test_workflow_action_requires_csrf_and_valid_origin_before_mutation(self):
        payload = {"action": "freeze_inputs", "frame_ids": [FRAME], "prompt": "HTTP fixture prompt.", "reviewer": "HTTP human"}
        with patch("mousebench.api_providers.execute_job") as execute:
            self.assertEqual(self.request("POST", "/api/workflow/action", payload, token=False)[0], 403)
            self.assertEqual(self.request("POST", "/api/workflow/action", payload, extra_headers={"Origin": "https://untrusted.example"})[0], 403)
            status, detail, _ = self.request("GET", "/api/frames/" + FRAME)
            self.assertEqual(status, 200)
            self.assertIsNone(detail["input_contract"])
            status, result, _ = self.request("POST", "/api/workflow/action", payload,
                                            extra_headers={"Origin": "http://127.0.0.1:" + str(self.server.server_port)})
            self.assertEqual(status, 200, result)
            self.assertEqual(result["saved"], 1)
            status, detail, _ = self.request("GET", "/api/frames/" + FRAME)
            self.assertEqual(detail["input_contract"]["prompt"], "HTTP fixture prompt.")
            self.assertEqual(self.request("POST", "/api/workflow/action", {"action": "execute_model"})[0], 400)
        execute.assert_not_called()

    def test_workflow_batch_export_submission_and_issue_resolution_over_http(self):
        self.freeze_input()
        rubric = self.workflow_rubric()
        with patch("mousebench.api_providers.execute_job") as execute:
            created = self.workflow_action("create_batch", task="ground_truth", executor="codex", model="HTTP fixture assistant",
                                           frame_ids=[FRAME], rubric_id=rubric["id"])
            batch_id = created["batch"]["id"]
            status, batch, _ = self.request("GET", "/api/workflow/batches/" + batch_id)
            self.assertEqual(status, 200, batch)
            self.assertEqual(batch["batch"]["counts"]["pending"], 1)
            self.assertNotIn("lease_token", batch["jobs"][0])
            self.assertNotIn("packet", batch["jobs"][0])
            status, exported, headers = self.request("GET", "/api/workflow/export/" + batch_id)
            self.assertEqual(status, 200, exported)
            self.assertIn("attachment", headers["Content-Disposition"])
            packet = exported["jobs"][0]
            self.assertEqual(packet["status"], "pending")
            self.assertEqual(packet["frame_id"], FRAME)
            self.assertEqual(packet["expected_result_schema"]["type"], "object")
            self.assertEqual(packet["image_base_directory"], str(self.data_dir.resolve()))
            content = dict(self.content, reference_description="HTTP fixture mouse description.")
            issue = {"severity": "warning", "category": "ambiguity", "message": "Inspect fixture orientation."}
            receipt = {"job_id": packet["job_id"], "worker": "HTTP fixture Codex", "model_version": "fixture-v1",
                       "result": {"content": content, "issues": [issue]}}
            submitted = self.workflow_action("submit_results", batch_id=batch_id, records=[receipt])
            self.assertEqual(submitted["imported"], 1)
            status, detail, _ = self.request("GET", "/api/frames/" + FRAME)
            self.assertEqual(detail["ground_truth"]["status"], "draft")
            self.assertTrue(detail["ground_truth"]["provenance"]["assistant_generated"])
            status, issues, _ = self.request("GET", "/api/workflow/issues")
            self.assertEqual(status, 200, issues)
            self.assertEqual(len(issues), 1)
            self.assertTrue(issues[0]["current"])
            self.assertEqual(issues[0]["status"], "open")
            resolved = self.workflow_action("resolve_issue", id=issues[0]["id"], status="resolved", reviewer="HTTP human",
                                            resolution="Inspected the fixture and documented the decision.")
            self.assertEqual(resolved["status"], "resolved")
            status, exported, _ = self.request("GET", "/api/workflow/export/" + batch_id)
            self.assertEqual(exported["jobs"], [])
            status, batch, _ = self.request("GET", "/api/workflow/batches/" + batch_id)
            self.assertEqual(batch["batch"]["counts"]["succeeded"], 1)
            conflicting = copy.deepcopy(receipt)
            conflicting["result"]["content"]["reference_description"] = "A conflicting replacement."
            self.assertEqual(self.request("POST", "/api/workflow/action", {"action": "submit_results", "records": [conflicting]})[0], 409)
        execute.assert_not_called()

    def test_workflow_release_readback_preserves_approved_snapshot(self):
        rubric = self.workflow_rubric()
        contract, gt = self.approve()
        release = self.workflow_action("create_release", name="HTTP fixture release", reviewer="HTTP human",
                                      frame_ids=[FRAME], rubric_id=rubric["id"])
        status, readback, headers = self.request("GET", "/api/workflow/releases/" + release["id"])
        self.assertEqual(status, 200, readback)
        self.assertEqual(readback, release)
        self.assertEqual(readback["samples"][0]["input"]["input_hash"], contract["input_hash"])
        self.assertEqual(readback["samples"][0]["ground_truth"]["content_sha256"], gt["content_sha256"])
        self.assertEqual(readback["rubric"]["sha256"], rubric["sha256"])
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.save_gt("draft", 3)
        self.assertEqual(self.request("GET", "/api/workflow/releases/" + release["id"])[1], release)

    def test_workflow_calibration_download_has_blank_human_labels_and_exact_bindings(self):
        rubric = self.workflow_rubric()
        contract, gt = self.approve()
        run = self.import_run(input_sha256=contract['input_hash'])
        judgment = self.make_judgment(run, contract, gt)
        judgment['rubric_sha256'] = rubric['sha256']
        judgment['protocol_version'] = rubric['content']['id']
        judgment['schema_version'] = 'mousebench.judgment.v2'
        required = [fact for fact in gt['content']['facts'] if fact['required']]
        judgment['claims'] = [{'id':'c1','text':required[0]['text'],'field':'Head',
            'correctness':1,'error_type':'none','severity':'none','evidence':'HTTP synthetic fixture.',
            'fact_ids':[required[0]['id']]}]
        judgment.pop('supported_fact_ids')
        judgment['fact_coverage'] = [{'fact_id':fact['id'],'coverage':1 if i==0 else 0,
            'claim_ids':['c1'] if i==0 else [],'evidence':'HTTP synthetic fixture coverage.'}
            for i,fact in enumerate(required)]
        judgment['criterion_assessments'] = {criterion['id']:{'verdict':'met','evidence':'HTTP synthetic fixture.'}
            for criterion in rubric['content']['analytic_criteria']}
        status, imported, _ = self.request("POST", "/api/judgments/import", {"records": [judgment]})
        self.assertEqual(status, 200, imported)
        status, template, headers = self.request("GET", "/api/workflow/calibration-template?run_id=" + run["id"])
        self.assertEqual(status, 200, template)
        self.assertIn("human-judge-calibration.json", headers["Content-Disposition"])
        self.assertTrue(template['blinded'])
        self.assertEqual(len(template["records"]), 3 + len(required))
        for record in template["records"]:
            self.assertFalse(record["human_verified"])
            self.assertEqual(record["verified_by"], "")
            self.assertIsNone(record["human_label"])
            self.assertEqual(record["prediction_sha256"], judgment["prediction_sha256"])
            self.assertEqual(record["input_sha256"], contract["input_hash"])
            self.assertNotIn('judge', record)
            self.assertNotIn('judge_label', record)
            self.assertIsNone(record['split'])
            self.assertEqual(record['context']['candidate_text'], 'Fixture answer.')
            self.assertEqual(record['context']['canonical_input']['prompt'], contract['prompt'])
            self.assertEqual(record['context']['image']['sha256'], contract['image_sha256'])
        unverified = {'action': 'import_calibration', 'rubric_id': rubric['id'], 'records': template['records']}
        self.assertEqual(self.request('POST', '/api/workflow/action', unverified)[0], 400)
        labeled = copy.deepcopy(template['records'])
        for record in labeled:
            record.update(split='heldout', human_verified=True, verified_by='Independent HTTP human',
                          human_label='supported' if record['kind'] == 'claim' else 1)
        imported = self.workflow_action('import_calibration', rubric_id=rubric['id'], records=labeled)
        self.assertEqual(imported['report']['n'], 3 + len(required))
        self.assertEqual(imported['records'][0]['judge_label'], 1)
        self.assertEqual(imported['records'][0]['judge'], judgment['judge'])
        labeled[0]['judge_label'] = 0
        self.assertEqual(self.request('POST', '/api/workflow/action',
                                     {'action': 'import_calibration', 'rubric_id': rubric['id'], 'records': labeled})[0], 400)
        self.save_gt("draft", 3)
        self.assertEqual(self.request("GET", "/api/workflow/calibration-template?run_id=" + run["id"])[1]["records"], [])

    def test_workflow_unknown_resources_do_not_leak_or_mutate(self):
        for path in ("/api/workflow/batches/unknown", "/api/workflow/export/unknown", "/api/workflow/releases/unknown"):
            self.assertEqual(self.request("GET", path)[0], 404, path)
        self.assertEqual(self.request("POST", "/api/workflow/action", {"action": "unknown"})[0], 400)
        self.assertEqual(self.request("POST", "/api/workflow/run", {"execute": True})[0], 404)

    def test_scoring_review_routes_bind_original_versions_and_images(self):
        from mousebench.workflow import Workflow
        from test_workflow import default_rubric
        contract, gt = self.approve()
        run = self.import_run()
        store = Store(self.data_dir)
        workflow = Workflow(store)
        saved = workflow.save_rubric({'content': default_rubric(), 'author': 'HTTP fixture'})
        rubric = workflow.freeze_rubric({'id': saved['id'], 'reviewer': 'HTTP fixture'})
        store.close()
        judgment = self.make_judgment(run, contract, gt)
        judgment.update(rubric_sha256=rubric['sha256'], protocol_version=rubric['content']['id'])
        judgment['criterion_assessments'] = {c['id']: {'verdict': 'met', 'evidence': 'HTTP fixture only.'}
            for c in rubric['content']['analytic_criteria']}
        status, payload, _ = self.request('POST', '/api/judgments/import', {'records': [judgment]})
        self.assertEqual(status, 200, payload)
        status, listing, _ = self.request('GET', '/api/scoring-review?run_id=' + run['id'])
        self.assertEqual(status, 200, listing)
        self.assertEqual(listing['counts']['judged'], 1)
        selected = listing['frames'][0]['judgment_id']
        status, detail, _ = self.request('GET', '/api/scoring-review/' + selected)
        self.assertEqual(status, 200, detail)
        self.assertEqual(detail['prediction']['text'], 'Fixture answer.')
        self.assertEqual(detail['input']['input_hash'], contract['input_hash'])
        image_url = detail['images'][0]['url']
        status, original, _ = self.request('GET', image_url)
        self.assertEqual(status, 200)
        store = Store(self.data_dir)
        store.register_image(FRAME, b'new fixture image', '/fixture/replaced.jpg')
        store.close()
        self.freeze_input('Changed task after grading.', revision=contract['revision'])
        status, detail, _ = self.request('GET', '/api/scoring-review/' + selected)
        self.assertEqual(status, 200, detail)
        self.assertEqual(detail['eligibility']['status'], 'stale')
        self.assertEqual(self.request('GET', image_url)[1], original)
        for path in ('/api/scoring-review/unknown', '/api/scoring-review/' + selected + '/image/unknown',
                     '/api/scoring-review/prediction-image/bad/shape', '/api/scoring-review?group_id=bad'):
            self.assertEqual(self.request('GET', path)[0], 404, path)
        self.assertEqual(self.request('POST', '/api/scoring-review/' + selected, {'score': 1})[0], 404)

    def test_scoring_review_unjudged_prediction_route_does_not_use_current_gt(self):
        self.approve()
        run = self.import_run()
        status, detail, _ = self.request('GET', '/api/scoring-review/prediction?run_id=' + run['id'] + '&frame_id=' + FRAME)
        self.assertEqual(status, 200, detail)
        self.assertIsNone(detail['judgment'])
        self.assertIsNone(detail['ground_truth'])
        self.assertIsNone(detail['input'])
        self.assertIsNone(detail['scores'])
        self.assertEqual(detail['images'], [])
        self.assertEqual(self.request('GET', '/api/scoring-review/prediction-image/' + run['id'] + '/' + FRAME + '/canonical')[0], 404)


if __name__ == "__main__":
    unittest.main()
