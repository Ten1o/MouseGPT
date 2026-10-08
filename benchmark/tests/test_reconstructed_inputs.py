"""Derived requests stay distinct from archives and transport exact verified inputs."""
import base64
import copy
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch

from mousebench.api_providers import _payload, validate_config
from mousebench.evaluation import analyze_output, check_output_structure
from mousebench.output_format import expected_output_fields
from mousebench.providers import build_job_preview
from mousebench.recorded_inputs import RECONSTRUCTED_PROTOCOL, input_semantics, verify_input_images
from mousebench.store import Store, dumps, digest, FIELDS
from mousebench.workflow import Workflow
from test_recorded_inputs import PROMPT

TEMPLATE = '231117_G4_30min_00002061'
TARGET = '231121_C3M3-2_00038137'
SYSTEM = 'Fixture original system: output exactly Description and Keywords.'


class ReconstructedInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(self.temp.name)
        self.format_patch = patch('mousebench.output_format.TWO_FIELD_SYSTEM_SHA256', hashlib.sha256(SYSTEM.encode()).hexdigest())
        self.format_patch.start()
        self.store.import_catalog({'frames':[{'id':TEMPLATE},{'id':TARGET}]})
        for fid in (TEMPLATE,TARGET):
            self.store.register_image(fid, b'preview-'+fid.encode(), '/fixture/'+fid+'.jpg', kind='source_preview')
        self.images = [b'\xff\xd8\xfffirst fixture view\xff\xd9',b'\xff\xd8\xffsecond fixture view\xff\xd9']
        self.prompt = PROMPT.replace('left and right', 'right and left')
        self.request = {'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':[
            {'type':'text','text':self.prompt}, *[{'type':'image_url','image_url':{
                'url':'data:image/jpeg;base64,'+base64.b64encode(raw).decode(),'detail':'high'}} for raw in self.images]]}],
            'max_tokens':2000,'temperature':0,'seed':1}
        self.add_record('raw_records', TEMPLATE, 'original-log', 'inference_log', self.request)
        self.template = self.store.save_input(TEMPLATE, {'source_record_id':'original-log','reviewer':'Fixture'})
        self.manifest = {'frame_id':TARGET,'request':copy.deepcopy(self.request), 'output_fields':['Description','Keywords'],
            'reconstruction':{'method_id':'fixture-derived-v1','template_frame_ids':[TEMPLATE],
                'velocity_method':{'source':'PRIVATE fixture velocity','method':'documented backward difference'},
                'image_method':{'source':'PRIVATE fixture images','method':'verified synchronized views'},
                'limitations':['Reconstruction; original request unavailable.']}}
        self.manifest['request']['messages'][1]['content'][0]['text'] = self.prompt.replace('9.77','2.34').replace('22.47, 9.76','4.64, 3.53')
        self.update_manifest()

    def tearDown(self):
        self.format_patch.stop()
        self.store.close()
        self.temp.cleanup()

    def add_record(self, kind, fid, record_id, role, content):
        raw = json.dumps(content).encode()
        sha = hashlib.sha256(raw).hexdigest()
        path = 'objects/'+sha+'.json'
        (self.store.root/path).write_bytes(raw)
        record = {'id':record_id,'frame_id':fid,'path':path,'sha256':sha,'role':role,
                  'source_path':'/fixture/'+record_id+'.json','frame_membership_verified':True}
        with self.store.transaction():
            self.store.db.execute('INSERT OR REPLACE INTO records VALUES(?,?,?,?)',(kind,record_id,fid,dumps(record)))
        return record

    def update_manifest(self):
        self.record = self.add_record('assets',TARGET,'derived-manifest','reconstruction_manifest',self.manifest)

    def save(self, **extra):
        return self.store.save_input(TARGET, {'reconstruction_manifest_id':'derived-manifest','reviewer':'Fixture reconstruction', **extra})

    def test_contract_preserves_exact_text_images_and_explicit_reconstruction_provenance(self):
        saved = self.save()
        self.assertEqual(saved['protocol_version'],RECONSTRUCTED_PROTOCOL)
        self.assertNotIn('source_request',saved)
        self.assertFalse(saved['reconstruction']['historical_request_recovered'])
        self.assertEqual(saved['reconstruction']['manifest_sha256'],self.record['sha256'])
        self.assertEqual(saved['reconstruction']['template_input_hashes'],{TEMPLATE:self.template['input_hash']})
        self.assertEqual(saved['output_fields'],['Description','Keywords'])
        self.assertEqual(saved['prompt'],self.manifest['request']['messages'][1]['content'][0]['text'])
        self.assertEqual(saved['velocity']['value']['right_forelimb'],4.64)
        self.assertEqual(saved['velocity']['value']['left_forelimb'],3.53)
        self.assertEqual(saved['input_hash'],digest(input_semantics(saved)))
        self.assertEqual(verify_input_images(self.store,saved),saved['input_images'])
        for image,raw in zip(saved['input_images'],self.images):
            self.assertEqual((self.store.root/image['path']).read_bytes(),raw)
        self.assertEqual(self.store.get_records('raw_records',TARGET),[])
        self.assertEqual(self.store.latest('input',TEMPLATE),self.template)
        logged_shape = dict(saved,protocol_version='mousebench.input.v3.logged')
        self.assertNotEqual(digest(input_semantics(logged_shape)),saved['input_hash'])

    def test_reconstruction_cannot_replace_a_logged_input(self):
        with self.assertRaisesRegex(ValueError,'cannot replace'):
            self.store.save_input(TEMPLATE, {'reconstruction_manifest_id':'derived-manifest','reviewer':'Fixture','expected_revision':1})
        self.assertEqual(self.store.versions('input',TEMPLATE),[self.template])

    def test_template_edits_missing_provenance_and_fake_log_identity_fail_closed(self):
        original = copy.deepcopy(self.manifest)
        for change in ('system','prompt','missing_method','missing_templates','output_fields'):
            with self.subTest(change=change):
                self.manifest=copy.deepcopy(original)
                if change=='system':self.manifest['request']['messages'][0]['content']+=' changed'
                elif change=='prompt':self.manifest['request']['messages'][1]['content'][0]['text']+=' changed'
                elif change=='missing_method':self.manifest['reconstruction'].pop('velocity_method')
                elif change=='missing_templates':self.manifest['reconstruction']['template_frame_ids']=[]
                else:self.manifest['output_fields']=list(FIELDS)
                self.update_manifest()
                with self.assertRaises(ValueError):self.save()
        self.manifest=original;self.update_manifest()
        with self.assertRaisesRegex(ValueError,'archived request provenance'):
            self.save(source_record_id='original-log')
        with self.assertRaisesRegex(ValueError,'differs'):
            self.save(prompt='Edited prompt')
        self.assertIsNone(self.store.latest('input',TARGET))

    def test_bad_manifest_membership_or_checksum_cannot_create_input(self):
        with self.store.transaction():
            bad=dict(self.record,frame_membership_verified=False)
            self.store.db.execute('UPDATE records SET data=? WHERE kind=? AND id=?',(dumps(bad),'assets',bad['id']))
        with self.assertRaisesRegex(ValueError,'membership'):self.save()
        self.update_manifest()
        (self.store.root/self.record['path']).write_bytes(b'changed manifest')
        with self.assertRaisesRegex(ValueError,'checksum'):self.save()
        self.assertIsNone(self.store.latest('input',TARGET))

    def test_exact_candidate_transport_and_blind_gt_exclude_reconstruction_evidence(self):
        self.save()
        entry=self.store.frame_detail(TARGET)
        config=validate_config({'provider':'openai','model':'fixture','api_mode':'chat_completions'})
        job=build_job_preview('inference',[entry])['jobs'][0]
        request,_,images=_payload(job,config,self.store.root)
        self.assertEqual(request['messages'],self.manifest['request']['messages'])
        self.assertEqual(len(images),2)
        self.assertNotIn('PRIVATE',json.dumps(request))
        canonical=json.loads(job['payload_template']['messages'][1]['content'])['canonical_input']
        self.assertEqual(canonical['output_fields'],['Description','Keywords'])
        self.assertNotIn('reconstruction',canonical)
        gt=build_job_preview('ground_truth',[entry],evidence_mode='sensory_only')['jobs'][0]
        self.assertNotIn('PRIVATE',json.dumps(gt))
        self.assertEqual(len(gt['attachment_manifest']),2)

    def test_gt_review_and_workflow_use_reconstructed_inputs_with_integrity_gates(self):
        contract=self.save()
        content={'sections':{field:'Fixture GT' for field in FIELDS}, 'facts':[
            {'id':'f1','text':'Fixture fact','field':'Overall','evidence':'image','required':True}]}
        reviewed=self.store.save_gt(TARGET,{'content':content,'status':'reviewed','reviewer':'Fixture'})
        self.store.save_gt(TARGET,{'content':content,'status':'approved','reviewer':'Fixture','human_confirmed':True,'expected_revision':reviewed['revision']})
        workflow=Workflow(self.store)
        batch=workflow.create_batch({'task':'inference','executor':'codex','model':'fixture','frame_ids':[TARGET]})
        self.assertEqual(len(workflow.job(batch['jobs'][0]['id'])['packet']['attachment_manifest']),2)
        (self.store.root/contract['input_images'][1]['path']).write_bytes(b'tampered view')
        with self.assertRaisesRegex(ValueError,'checksum'):
            verify_input_images(self.store,contract)

    def test_original_two_field_request_overrides_old_metadata_without_version_changes(self):
        before=list(self.store.db.iterdump())
        self.assertEqual(self.template['output_fields'],list(FIELDS))
        fields=expected_output_fields(self.template)
        self.assertEqual(fields,('Description','Keywords'))
        text='{"Description":"Fixture behavior.","Keywords":"fixture"}'
        view=analyze_output(text,fields)
        self.assertTrue(view['contract_valid'])
        self.assertEqual(view['expected_fields'],['Description','Keywords'])
        self.assertTrue(check_output_structure(text,fields)['passed'])
        self.assertFalse(check_output_structure(text)['passed'])
        self.assertEqual(before,list(self.store.db.iterdump()))
        run=self.store.import_run({'name':'Fixture original output','model':'fixture','records':[{'frame_id':TEMPLATE,'output':text}]})
        self.assertTrue(self.store.predictions(run['id'])[0]['mechanical_checks']['passed'])
        self.assertEqual(self.store.latest('input',TEMPLATE),self.template)


if __name__=='__main__':unittest.main()
