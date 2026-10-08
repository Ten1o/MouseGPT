import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from mousebench.store import Store, Conflict, FIELDS, digest

FRAME='240420_C10M1_00003972'

class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.store=Store(self.temp.name)
        self.store.import_catalog({'frames':[{'id':FRAME,'recording_id':'240420_C10M1'}],'assets':[],'ratings':[]})
        self.store.register_image(FRAME,b'fixture-image-not-a-real-scientific-input','fixture')
        self.contract=self.store.save_input(FRAME,{'prompt':'Describe the six fields.','reviewer':'test','expected_revision':0})
        self.content={'reference_description':'test fixture only','facts':[{'id':'f1','text':'Fixture fact','field':'Overall','evidence':'image','required':True}],
                      'uncertainties':[],'sections':{f:'Fixture only' for f in FIELDS}}
    def tearDown(self):
        self.store.close();self.temp.cleanup()
    def save(self,status,revision=0,content=None):
        return self.store.save_gt(FRAME,{'status':status,'human_confirmed':status=='approved','expected_revision':revision,'reviewer':'test','content':content or self.content})
    def approved(self):
        self.save('reviewed');return self.save('approved',1)
    def direct_approval(self, revision=0, content=None, **overrides):
        payload = {'status': 'approved', 'actor_type': 'human', 'human_confirmed': True,
                   'approval_action': 'approve_reference', 'expected_revision': revision,
                   'expected_input_hash': self.contract['input_hash'],
                   'content': self.content if content is None else content}
        payload.update(overrides)
        return self.store.save_gt(FRAME, payload)
    def test_bulk_latest_and_prediction_headers_preserve_version_and_hash_semantics(self):
        self.save('draft')
        self.save('draft',1)
        self.assertEqual(self.store.latest_versions('gt'),{FRAME:self.store.latest('gt',FRAME)})
        self.assertEqual(self.store.latest_versions('input'),{FRAME:self.store.latest('input',FRAME)})
        self.assertEqual(self.store.latest_versions('absent'),{})
        examples=[{'input_hash':'legacy'}, {'input_sha256':None,'input_hash':'legacy'},
                  {'input_sha256':'current','input_hash':'legacy'}, {}]
        for i,example in enumerate(examples):
            raw=dict(example,status='succeeded',frame_id=FRAME,request_log={'large':'unused'})
            self.store.db.execute('INSERT INTO predictions VALUES(?,?,?)',(str(i),FRAME,json.dumps(raw)))
            self.assertEqual(self.store.prediction_headers(str(i)),[{
                'frame_id':FRAME,'status':'succeeded','input_sha256':raw.get('input_sha256',raw.get('input_hash'))}])

    def test_one_click_approval_is_one_revision_without_an_invented_reviewer(self):
        draft = self.store.save_gt(FRAME, {'status': 'draft', 'actor_type': 'assistant',
            'content': self.content, 'provenance': {'job_id': 'fixture-job'}})
        approved = self.direct_approval(draft['revision'])
        self.assertEqual(approved['status'], 'approved')
        self.assertEqual(approved['revision'], draft['revision'] + 1)
        self.assertEqual(approved['reviewer'], '')
        self.assertEqual(approved['notes'], '')
        self.assertEqual([v['status'] for v in self.store.versions('gt', FRAME)], ['approved', 'draft'])
        self.assertEqual(approved['content_sha256'], draft['content_sha256'])
        self.assertEqual(approved['input_hash'], self.contract['input_hash'])
        self.assertEqual(approved['provenance']['approval_action'], 'approve_reference')
        self.assertEqual(approved['provenance']['actor_type'], 'human')
        self.assertTrue(approved['provenance']['human_confirmed'])
        self.assertTrue(approved['provenance']['assistant_generated'])
        self.assertEqual(approved['provenance']['job_id'], 'fixture-job')
        later_draft = self.store.save_gt(FRAME, {'status': 'draft', 'expected_revision': approved['revision'],
                                                'content': self.content})
        self.assertNotIn('approval_action', later_draft['provenance'])
        self.assertFalse(later_draft['provenance']['human_confirmed'])
    def test_one_click_approval_accepts_displayed_content_edits(self):
        self.save('draft')
        edited = copy.deepcopy(self.content)
        edited['sections']['Head'] = 'Corrected fixture description.'
        approved = self.direct_approval(1, edited, reviewer='  Optional name  ', notes='Optional note')
        self.assertEqual(approved['content'], edited)
        self.assertEqual(approved['content_sha256'], digest(edited))
        self.assertEqual(approved['reviewer'], 'Optional name')
        self.assertEqual(approved['notes'], 'Optional note')
        self.assertEqual(len(self.store.versions('gt', FRAME)), 2)
    def test_one_click_approval_rejects_stale_revision_and_input_binding(self):
        draft = self.save('draft')
        with self.assertRaises(Conflict): self.direct_approval(0)
        with self.assertRaises(Conflict): self.direct_approval(1, expected_input_hash=None)
        self.store.save_input(FRAME, {'prompt': 'Changed fixture', 'reviewer': 'test', 'expected_revision': 1})
        with self.assertRaises(Conflict): self.direct_approval(1)
        self.assertEqual(self.store.latest('gt', FRAME), draft)
        self.assertFalse(self.store.db.in_transaction)
    def test_one_click_approval_requires_explicit_human_action(self):
        self.save('draft')
        for change in ({'actor_type': 'assistant'}, {'actor_type': None},
                       {'human_confirmed': False}, {'approval_action': 'save'},
                       {'status': 'draft'}, {'approval_action': None}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.direct_approval(1, **change)
        self.assertEqual(len(self.store.versions('gt', FRAME)), 1)
    def test_one_click_approval_requires_a_displayed_reference_revision(self):
        with self.assertRaises(Conflict):
            self.store.save_gt(FRAME, {'status': 'approved', 'actor_type': 'human', 'human_confirmed': True,
                'approval_action': 'approve_reference', 'expected_input_hash': self.contract['input_hash'],
                'content': self.content})
        self.assertIsNone(self.store.latest('gt', FRAME))
    def test_one_click_approval_keeps_content_and_image_gates(self):
        self.save('draft')
        for change in ({'facts': []}, {'sections': {}},
                       {'facts': [dict(self.content['facts'][0], evidence='velocity')]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.direct_approval(1, dict(self.content, **change))
        image = self.store.image(FRAME)
        (self.store.root / image['path']).write_bytes(b'changed fixture bytes')
        with self.assertRaisesRegex(ValueError, 'checksum'): self.direct_approval(1)
        self.assertEqual(len(self.store.versions('gt', FRAME)), 1)
    def test_one_click_approval_cannot_edit_away_an_open_critical_issue(self):
        from mousebench.workflow import Workflow
        workflow = Workflow(self.store)
        draft = self.save('draft')
        issue = workflow._issue(FRAME, {'severity': 'critical', 'category': 'contradiction',
            'message': 'Fixture claim needs resolution.', 'fact_ids': ['f1']}, 'fixture', draft, self.contract['input_hash'])
        with self.store.transaction(): workflow.put('issue', issue)
        edited = copy.deepcopy(self.content)
        edited['facts'][0]['text'] = 'Changed fixture fact'
        for content in (self.content, edited):
            with self.subTest(content=content), self.assertRaisesRegex(ValueError, 'critical'):
                self.direct_approval(1, content)
        self.assertEqual(self.store.latest('gt', FRAME), draft)
        workflow.resolve_issue({'id': issue['id'], 'status': 'resolved', 'reviewer': 'Fixture',
                                'resolution': 'The fixture contradiction was resolved.'})
        approved = self.direct_approval(1, edited)
        self.assertEqual(approved['status'], 'approved')
    def test_one_click_approval_checks_issues_on_submitted_content(self):
        from mousebench.workflow import Workflow
        workflow = Workflow(self.store)
        draft = self.save('draft')
        edited = copy.deepcopy(self.content)
        edited['facts'][0]['text'] = 'Changed fixture fact'
        submitted = dict(draft, content=edited, content_sha256=digest(edited))
        issue = workflow._issue(FRAME, {'severity': 'critical', 'message': 'Fixture submitted claim needs resolution.'},
                                'fixture', submitted, self.contract['input_hash'])
        with self.store.transaction(): workflow.put('issue', issue)
        with self.assertRaisesRegex(ValueError, 'critical'): self.direct_approval(1, edited)
        self.assertEqual(self.store.latest('gt', FRAME), draft)
    def test_ground_truth_name_is_optional_without_changing_status(self):
        for reviewer in ('', '  ', None):
            previous = self.store.latest('gt', FRAME)
            saved = self.store.save_gt(FRAME, {'status': 'draft', 'reviewer': reviewer, 'content': self.content,
                'expected_revision': previous['revision'] if previous else 0})
            self.assertEqual(saved['reviewer'], '')
            self.assertEqual(saved['status'], 'draft')
            self.assertFalse(saved['provenance']['human_confirmed'])
    def test_approval_requires_reviewed_exact_content(self):
        with self.assertRaises(ValueError):self.save('approved')
        self.save('reviewed')
        altered=copy.deepcopy(self.content);altered['facts'][0]['text']='Changed'
        with self.assertRaises(ValueError):self.save('approved',1,altered)
        self.save('approved',1)
        self.save('draft',2,altered)
        history=self.store.versions('gt',FRAME)
        self.assertEqual(history[1]['status'],'approved')
        self.assertEqual(history[1]['content']['facts'][0]['text'],'Fixture fact')
    def test_concurrent_write_conflict(self):
        self.save('draft')
        with self.assertRaises(Conflict):self.save('draft',0)
        self.assertEqual(len(self.store.versions('gt',FRAME)),1)
    def test_temporal_facts_rejected_on_image_only(self):
        self.content['facts'][0]['evidence']='velocity'
        with self.assertRaises(ValueError):self.save('reviewed')
    def test_input_change_requires_new_review(self):
        self.save('reviewed')
        self.store.save_input(FRAME,{'prompt':'Changed','reviewer':'test','expected_revision':1})
        with self.assertRaises(ValueError):self.save('approved',1)
    def test_input_hash_stable_for_same_request(self):
        c=self.store.save_input(FRAME,{'prompt':'Describe the six fields.','reviewer':'other','expected_revision':1})
        self.assertEqual(c['input_hash'],self.contract['input_hash'])
    def test_atomic_run_import_and_duplicate_rejection(self):
        p={'name':'Fixture','model':'fixture','records':[{'frame_id':FRAME,'output':'a'},{'frame_id':FRAME,'output':'b'}]}
        with self.assertRaises(ValueError):self.store.import_run(p)
        self.assertEqual(self.store.list_runs(),[])
    def test_controlled_run_requires_real_metadata_and_matching_input(self):
        p={'name':'Fixture','model':'fixture','track':'controlled','records':[{'frame_id':FRAME,'output':'a','input_hash':self.contract['input_hash']}]}
        with self.assertRaises(ValueError):self.store.import_run(p)
        p['records'][0].update(model_metadata={'model_id':'fixture','generation_settings':{}},request_log={'kind':'native_execution','request':'fixture','input_sha256':self.contract['input_hash']})
        r=self.store.import_run(p);self.assertEqual(r['track'],'controlled')
    def test_import_judgment_hashes_and_stale_reference(self):
        gt=self.approved()
        run=self.store.import_run({'name':'Fixture','model':'fixture','records':[{'frame_id':FRAME,'output':'A fixture answer'}]})
        prediction=self.store.predictions(run['id'])[0]
        j={'schema_version':'mousebench.judgment.v1','frame_id':FRAME,'run_id':run['id'],'protocol_version':'fixture-v1',
           'judge':{'provider':'fixture','model':'fixture','version':'1'},'rubric_sha256':'a'*64,'input_sha256':self.contract['input_hash'],
           'ground_truth_sha256':gt['content_sha256'],'prediction_sha256':prediction['prediction_sha256'],
           'claims':[{'text':'Fixture fact','label':'supported'}],'supported_fact_ids':['f1'],'structure_score':1,'instruction_score':1}
        bad=dict(j,prediction_sha256='b'*64)
        with self.assertRaises(ValueError):self.store.import_judgments({'records':[bad]})
        self.assertEqual(self.store.import_judgments({'records':[j]})['imported'],1)
        self.assertEqual(self.store.scorecards()['groups'][0]['input_track'],'image_only')
        self.save('draft',2)
        self.assertEqual(self.store.scorecards()['stale_judgments_excluded'],1)
    def test_dedup_and_catalog_reimport_preserve_reviews(self):
        self.save('draft')
        self.store.import_catalog({'frames':[{'id':FRAME,'recording_id':'240420_C10M1'}],'assets':[],'ratings':[]})
        self.store.register_image(FRAME,b'fixture-image-not-a-real-scientific-input','fixture-copy')
        self.assertEqual(len(list((Path(self.temp.name)/'objects').iterdir())),1)
        self.assertEqual(self.store.latest('gt',FRAME)['revision'],1)
    def test_frame_output_views_preserve_stored_text_and_database_bytes(self):
        text='  {"Overall": "Fixture", "Head": "first", "Head": "second"}\n'
        asset={'id':'output-fixture','frame_id':FRAME,'nas_path':'/fixture/output.txt'}
        sha=hashlib.sha256(text.encode()).hexdigest()
        self.store.register_text(asset,text,sha)
        before=list(self.store.db.iterdump())
        original=self.store.get_records('descriptions',FRAME)[0]
        output=self.store.frame_detail(FRAME)['descriptions'][0]
        self.assertEqual(output['text'],text)
        self.assertEqual(output['sha256'],sha)
        self.assertTrue(output['output_view']['json_valid'])
        self.assertFalse(output['output_view']['contract_valid'])
        head=next(field for field in output['output_view']['fields'] if field['name']=='Head')
        self.assertEqual(head,{'name':'Head','value':None,'status':'duplicate'})
        self.assertNotIn('output_view',original)
        self.assertEqual(self.store.get_records('descriptions',FRAME)[0],original)
        self.assertEqual(list(self.store.db.iterdump()),before)
    def test_frame_detail_exposes_only_verified_historical_input_bindings(self):
        from mousebench.workflow import Workflow
        workflow=Workflow(self.store)
        self.store.import_catalog({'frames':[{'id':FRAME}], 'ratings':[{'id':'rating-fixture','frame_id':FRAME}]})
        run=self.store.import_run({'name':'Historical fixture','model':'unverified','records':[{'frame_id':FRAME,'output':'Fixture answer'}]})
        payload={'frame_id':FRAME,'rating_ids':['rating-fixture'],'run_id':run['id'],'reviewer':'Fixture human',
                 'reason':'Fixture exact correspondence only','human_confirmed':True}
        workflow.verify_evidence_link(payload)
        self.assertEqual(self.store.frame_detail(FRAME)['input_evidence_links'],[])
        workflow.verify_evidence_link(dict(payload,input_verified=True))
        before=list(self.store.db.iterdump())
        links=self.store.frame_detail(FRAME)['input_evidence_links']
        self.assertEqual(len(links),1)
        self.assertEqual(links[0]['input_sha256'],self.contract['input_hash'])
        self.assertEqual(links[0]['prediction_hashes'],[self.store.predictions(run['id'])[0]['prediction_sha256']])
        self.assertNotIn('reviewer',links[0])
        self.assertEqual(list(self.store.db.iterdump()),before)

if __name__=='__main__':unittest.main()
