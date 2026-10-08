"""Durable local workflows. Creating jobs never invokes a model.

Codex and API workers share immutable job packets and normalized result ingestion.
A database transaction couples each receipt to its scientific records. Ambiguous
in-flight calls are never automatically repeated after an interruption.
"""
import copy
import hashlib
import json
import statistics
import time
import uuid
from pathlib import Path
from .store import Store, Conflict, FIELDS, digest, dumps, now
from .providers import build_job_preview
from .protocol import default_rubric, validate_rubric, rubric_hash, benchmark_index, prepare_benchmark_index, calibration_report

TASKS = ('ground_truth', 'gt_check', 'inference', 'judge')
SEVERITIES = ('critical', 'warning', 'info')


def identifier(prefix):
    return prefix + '_' + uuid.uuid4().hex[:16]


def required_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name + ' is required')
    return value.strip()


def result_schema(task, rubric=None):
    fact = {'type':'object','required':['id','text','field','evidence','required'],
            'properties':{'id':{'type':'string'},'text':{'type':'string'},'field':{'enum':list(FIELDS)},
                          'evidence':{'enum':['image','velocity','unknown']},'required':{'type':'boolean'},
                          'evidence_note':{'type':'string'},'accepted_alternatives':{'type':'array','items':{'type':'string'}}}}
    issue = {'type':'object','required':['severity','category','message'],
             'properties':{'severity':{'enum':list(SEVERITIES)},'category':{'type':'string'},'message':{'type':'string'},
                           'target':{'enum':['reference','candidate'],'description':'Judge issues default to candidate; use reference only for suspected GT/evidence defects. GT draft/check issues concern the reference.'},
                           'fact_ids':{'type':'array','items':{'type':'string'}},'evidence':{'type':'string'},'suggested_fix':{'type':'string'}}}
    issues={'type':'array','items':issue}
    if task == 'ground_truth':
        return {'type':'object','required':['content','issues'],'properties':{
            'content':{'type':'object','required':['reference_description','sections','facts','uncertainties'],
                       'properties':{'reference_description':{'type':'string'},'sections':{'type':'object','required':list(FIELDS),'properties':{f:{'type':'string'} for f in FIELDS}},
                                     'facts':{'type':'array','items':fact},'uncertainties':{'type':'array','items':{'type':'string'}}}},'issues':issues}}
    if task == 'gt_check':
        return {'type':'object','required':['issues'],'properties':{'issues':issues}}
    if task == 'judge':
        result={'type':'object','required':['claims','supported_fact_ids','structure_score','instruction_score'],
                'properties':{'claims':{'type':'array','items':{'type':'object','required':['text','label','evidence','fact_ids'],
                              'properties':{'text':{'type':'string'},'label':{'enum':['supported','contradicted','unverifiable']},
                                            'evidence':{'type':'string'},'fact_ids':{'type':'array','items':{'type':'string'}}}}},
                              'supported_fact_ids':{'type':'array','items':{'type':'string'}},
                              'structure_score':{'type':'number','minimum':0,'maximum':1},'instruction_score':{'type':'number','minimum':0,'maximum':1},
                              'notes':{'type':'string'},'issues':issues}}
        if (rubric or {}).get('scoring',{}).get('version')=='mousebench.graded-factual-f1.v2':
            quarter={'type':'number','enum':[0,0.25,0.5,0.75,1]}
            strings={'type':'array','items':{'type':'string'},'uniqueItems':True}
            result['required']=['claims','fact_coverage','structure_score','instruction_score']
            result['properties'].pop('supported_fact_ids')
            result['properties'].update({
                'claims':{'type':'array','items':{'type':'object','additionalProperties':False,
                    'required':['id','text','field','correctness','evidence','error_type','severity','fact_ids'],
                    'properties':{'id':{'type':'string'},'text':{'type':'string'},
                        'field':{'enum':['Overall','Head','Limb','Torso','Others']},
                        'correctness':copy.deepcopy(quarter),'evidence':{'type':'string'},
                        'error_type':{'enum':['none','contradicted','unverifiable','mixed']},
                        'severity':{'enum':['none','minor','major']},'fact_ids':copy.deepcopy(strings)}}},
                'fact_coverage':{'type':'array','description':'Exactly one row for every required approved fact, including omissions.',
                    'items':{'type':'object','additionalProperties':False,
                        'required':['fact_id','coverage','claim_ids','evidence'],
                        'properties':{'fact_id':{'type':'string'},'coverage':copy.deepcopy(quarter),
                            'claim_ids':copy.deepcopy(strings),'evidence':{'type':'string'}}}},
                'structure_score':copy.deepcopy(quarter),'instruction_score':copy.deepcopy(quarter)})
        criteria=(rubric or {}).get('analytic_criteria',[])
        if criteria:
            result['required'].append('criterion_assessments')
            result['properties']['criterion_assessments']={'type':'object','additionalProperties':False,
                'required':[c['id'] for c in criteria],'properties':{c['id']:{'type':'object','required':['verdict','evidence'],
                    'additionalProperties':False,'properties':{'verdict':{'enum':['met','unmet']+(['not_assessable'] if c['allow_not_assessable'] else [])},
                    'evidence':{'type':'string','description':'Concise evidence for this criterion; explain any not-assessable verdict.'}}} for c in criteria}}
        return result
    return {'type':'object','required':['output_text'],'properties':{
        'output_text':{'type':'string','description':'Exact candidate model output; no repair, summarization or added commentary.'}},
        'description':'Local submission envelope only. The candidate model receives the frozen input without this schema or wrapper.'}


class Workflow:
    def __init__(self, store):
        self.store=store; self.db=store.db
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS wf_records(kind TEXT, id TEXT, data TEXT NOT NULL, PRIMARY KEY(kind,id));
        CREATE TABLE IF NOT EXISTS wf_jobs(id TEXT PRIMARY KEY, batch_id TEXT NOT NULL, frame_id TEXT NOT NULL,
            status TEXT NOT NULL, lease_token TEXT, leased_at REAL, data TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS wf_batch_status ON wf_jobs(batch_id,status);
        ''')

    def records(self, kind):
        rows=[json.loads(r[0]) for r in self.db.execute('SELECT data FROM wf_records WHERE kind=?',(kind,))]
        return sorted(rows,key=lambda r:(r.get('created_at',''),r['id']))

    def record(self, kind, item_id):
        row=self.db.execute('SELECT data FROM wf_records WHERE kind=? AND id=?',(kind,item_id)).fetchone()
        if row is None: raise KeyError('Unknown '+kind)
        return json.loads(row[0])

    def put(self, kind, value):
        self.db.execute('INSERT OR REPLACE INTO wf_records VALUES(?,?,?)',(kind,value['id'],dumps(value)))

    def event(self, action, data):
        self.put('event',{'id':identifier('event'),'action':action,'created_at':now(),'data':data})

    def frame_ids(self, ids):
        catalog_ids = [f['id'] for f in self.store.get_records('frames')]
        if ids is None or ids==[]: ids=catalog_ids
        if (not isinstance(ids,list) or not ids or len(ids)>len(catalog_ids)
            or any(not isinstance(fid, str) for fid in ids) or len(ids)!=len(set(ids))):
            raise ValueError(f'Choose 1–{len(catalog_ids)} distinct frame IDs')
        for fid in ids: self.store.frame(fid)
        return sorted(ids)

    def rubric(self, item_id=None):
        if item_id: return self.record('rubric',item_id)
        frozen=[r for r in self.records('rubric') if r['status']=='frozen']
        if not frozen: raise ValueError('Save and freeze a rubric before creating judge jobs or a release')
        return sorted(frozen,key=lambda r:r['created_at'])[-1]

    def save_rubric(self, payload):
        content=validate_rubric(payload.get('content'))
        author=required_text(payload.get('author'),'Author')
        sha=rubric_hash(content)
        # An identical draft is an idempotent save, independent of display label.
        for old in self.records('rubric'):
            if old['sha256']==sha:return old
        entry={'id':identifier('rubric'),'content':content,'sha256':sha,'status':'draft','author':author,
               'label':payload.get('label') or content['id'],'created_at':now(),'parent_id':payload.get('parent_id')}
        if entry['parent_id']:self.record('rubric',entry['parent_id'])
        with self.store.transaction():self.put('rubric',entry);self.event('save_rubric',{'id':entry['id'],'sha256':sha})
        return entry

    def freeze_rubric(self, payload):
        entry=self.record('rubric',payload.get('id'))
        reviewer=required_text(payload.get('reviewer'),'Reviewer')
        if entry['status']=='frozen':return entry
        # Content and hash stay immutable. Freeze is a release decision, not calibration.
        entry.update(status='frozen',frozen_by=reviewer,frozen_at=now(),calibration_status='not_established')
        with self.store.transaction():self.put('rubric',entry);self.event('freeze_rubric',{'id':entry['id'],'reviewer':reviewer})
        return entry

    def freeze_inputs(self, payload):
        ids=self.frame_ids(payload.get('frame_ids'))
        prompt=required_text(payload.get('prompt'),'Prompt')
        reviewer=required_text(payload.get('reviewer'),'Reviewer')
        velocity=payload.get('velocity')
        # One shared velocity across different frames is generally a provenance error.
        if velocity is not None and len(ids)>1:raise ValueError('Enter verified velocity per frame; bulk input setup is image-only')
        saved=[]; skipped=[]
        with self.store.transaction():
            for fid in ids:
                old=self.store.latest('input',fid)
                if old and not payload.get('overwrite',False): skipped.append(fid);continue
                value=self.store.save_input(fid,dict(payload,prompt=prompt,reviewer=reviewer,expected_revision=old['revision'] if old else 0))
                saved.append({'frame_id':fid,'revision':value['revision'],'input_sha256':value['input_hash']})
            self.event('freeze_inputs',{'reviewer':reviewer,'saved':saved,'skipped':skipped})
        return {'saved':len(saved),'skipped':len(skipped),'records':saved}

    def summary(self):
        summary=self.store.summary()
        batches=[self.batch(b['id'],include_jobs=False)['batch'] for b in self.records('batch')]
        issues=self.issues(current_only=True)
        providers=[]
        try:
            from .api_providers import provider_status
            for cfg in self.records('provider'):
                providers.append(dict(provider_status(cfg['config']),id=cfg['id']))
        except ImportError:pass
        return {'default_prompt':(Path(__file__).resolve().parents[1]/'protocol'/'prompt.txt').read_text(),
                'input_baseline':'recorded_per_frame_request',
                'default_rubric':default_rubric(),'rubrics':self.records('rubric'),'batches':batches,'issues':issues,
                'calibrations':[dict(c,records_count=len(c['records']),records=None) for c in self.records('calibration')],
                'evidence_links':self.records('evidence_link'),
                'releases':[dict({k:v for k,v in r.items() if k not in ('samples','rubric')},frame_count=len(r['samples']),rubric_id=r['rubric']['id']) for r in self.records('release')],
                'providers':providers,
                'readiness':dict({k:summary[k] for k in ('frames','input_contracts','gt_approved','gt_drafts','gt_reviewed')},open_issues=sum(i['status']=='open' for i in issues)),
                'open_issues':sum(i['status']=='open' for i in issues),'live_execution':'cli_only'}

    def freeze_recorded_inputs(self, payload):
        """Import each frame's own archived request; never broadcast one speed/prompt."""
        ids=self.frame_ids(payload.get('frame_ids'))
        reviewer=required_text(payload.get('reviewer'),'Importer')
        saved=[]; skipped=[]
        from .recorded_inputs import recorded_input
        with self.store.transaction():
            for fid in ids:
                old=self.store.latest('input',fid)
                if old and not payload.get('overwrite',False):
                    skipped.append(fid);continue
                source=recorded_input(self.store,fid)
                if source['status']!='ready':
                    raise ValueError('Recorded input unavailable for '+fid+': '+source.get('error',source['status']))
                entry=self.store.save_input(fid,{'source_record_id':source['source_record_id'],
                    'reviewer':reviewer,'expected_revision':old['revision'] if old else 0,
                    'notes':'Imported from the archived per-frame request. Source availability does not verify every historical model execution.'})
                saved.append({'frame_id':fid,'revision':entry['revision'],'input_sha256':entry['input_hash'],
                              'source_record_id':source['source_record_id']})
            self.event('freeze_recorded_inputs',{'reviewer':reviewer,'saved':saved,'skipped':skipped})
        return {'saved':len(saved),'skipped':len(skipped),'records':saved}

    def job(self, item_id):
        row=self.db.execute('SELECT * FROM wf_jobs WHERE id=?',(item_id,)).fetchone()
        if row is None:raise KeyError('Unknown job')
        d=json.loads(row['data']);d.update(id=row['id'],batch_id=row['batch_id'],frame_id=row['frame_id'],status=row['status'],lease_token=row['lease_token'],leased_at=row['leased_at'])
        return d

    def save_job(self, value):
        data={k:v for k,v in value.items() if k not in ('id','batch_id','frame_id','status','lease_token','leased_at')}
        self.db.execute('INSERT OR REPLACE INTO wf_jobs VALUES(?,?,?,?,?,?,?)',
            (value['id'],value['batch_id'],value['frame_id'],value['status'],value.get('lease_token'),value.get('leased_at'),dumps(data)))

    def batch(self, item_id, include_jobs=True):
        b=self.record('batch',item_id)
        counts={k:0 for k in ('pending','leased','succeeded','failed','uncertain')}
        counts.update(dict(self.db.execute('SELECT status,count(*) FROM wf_jobs WHERE batch_id=? GROUP BY status',(item_id,))))
        b['counts']=counts;b['total']=sum(counts.values())
        result={'batch':b}
        if include_jobs:
            result['jobs']=[{k:v for k,v in self.job(r[0]).items() if k not in ('packet','raw_result','lease_token')}
                            for r in self.db.execute('SELECT id FROM wf_jobs WHERE batch_id=? ORDER BY frame_id',(item_id,))]
        return result

    def create_batch(self, payload):
        task=payload.get('task');executor=payload.get('executor','codex')
        if task not in TASKS or executor not in ('codex','openai','azure'):raise ValueError('Unsupported task or executor')
        evidence_mode=payload.get('evidence_mode','expert_informed')
        if evidence_mode not in ('expert_informed','sensory_only'):raise ValueError('Unsupported GT evidence mode')
        if task!='ground_truth' and evidence_mode!='expert_informed':raise ValueError('GT evidence mode applies only to ground truth drafting')
        ids=self.frame_ids(payload.get('frame_ids'))
        model=required_text(payload.get('model'),'Model or Codex model label')
        name=payload.get('name') or task+' / '+model
        rubric=self.rubric(payload.get('rubric_id')) if task=='judge' else (self.record('rubric',payload['rubric_id']) if payload.get('rubric_id') and task!='inference' else None)
        if task=='judge' and rubric['status']!='frozen':raise ValueError('Freeze this rubric before judging')
        config=None
        if executor!='codex':
            from .api_providers import validate_config
            config=validate_config(dict(payload.get('provider_config') or {},provider=executor,model=model))
        run=None; preds={}
        if task=='judge':
            run=next((r for r in self.store.list_runs() if r['id']==payload.get('run_id')),None)
            if run is None:raise ValueError('Select a prediction run to judge')
            preds={p['frame_id']:p for p in self.store.predictions(run['id'])}
        batch_id=identifier('batch'); jobs=[]; tracks=set()
        for fid in ids:
            entry=self.store.frame_detail(fid)
            if not entry.get('input_contract'):raise ValueError('Freeze input for '+fid+' first')
            tracks.add(entry['input_contract']['track'])
            entry['expert_evidence']=entry['ratings'];entry['candidates']=entry['descriptions']
            if task=='ground_truth':
                links=[r for r in self.records('evidence_link') if r['frame_id']==fid]
                if links:
                    entry['expert_evidence']=dict(source_rows=entry['ratings'],verified_output_links=links)
            if task=='judge':
                entry['prediction']=preds.get(fid)
                if not entry['prediction'] or entry['prediction']['status']!='succeeded':
                    raise ValueError('Selected frame has no successful prediction: '+fid)
                gt=entry.get('ground_truth') or self.store.latest('gt',fid)
                if gt and self.store.has_critical_issue(fid,gt['content_sha256'],entry['input_contract']['input_hash']):
                    raise ValueError('Resolve critical reference issues before judging '+fid)
            packet=build_job_preview(task,[entry],run=run,rubric=rubric['content'] if rubric else None,evidence_mode=evidence_mode)['jobs'][0]
            if rubric and task in ('ground_truth','gt_check'):
                payload_data=json.loads(packet['payload_template']['messages'][1]['content'])
                payload_data['rubric']=rubric['content']
                packet['payload_template']['messages'][1]['content']=dumps(payload_data)
                packet['rubric_sha256']=rubric['sha256']
            packet['expected_result_schema']=result_schema(task,rubric['content'] if rubric else None)
            packet['packet_schema_version']='mousebench.work.v1'
            if task!='inference':
                packet['payload_template']['messages'][0]['content']+=' Return a JSON object matching this schema: '+dumps(packet['expected_result_schema'])
                packet['payload_sha256']=digest(packet['payload_template'])
            expected_gt=entry['ground_truth']
            packet['job_id']=identifier('job')
            packet['expected_gt_revision']=expected_gt['revision'] if expected_gt else 0
            if task=='gt_check':packet['reference_sha256']=expected_gt['content_sha256']
            jobs.append({'id':packet['job_id'],'batch_id':batch_id,'frame_id':fid,'status':'pending','packet':packet,
                         'attempts':0,'created_at':now(),'history':[]})
        if len(tracks)>1:raise ValueError('Use separate batches for image-only and image-plus-velocity conditions')
        release_id=payload.get('release_id')
        if release_id:
            release=self.record('release',release_id)
            samples={s['frame_id']:s for s in release['samples']}
            for j in jobs:
                sample=samples.get(j['frame_id'])
                if not sample or sample['input']['input_hash']!=j['packet']['input_contract_sha256']:
                    raise ValueError('Selected frames do not match this release')
                if task=='judge' and sample['ground_truth']['content_sha256']!=j['packet']['bindings']['ground_truth_sha256']:
                    raise ValueError('Reference changed since release')
            if rubric and release['rubric']['sha256']!=rubric['sha256']:raise ValueError('Rubric differs from release')
        b={'id':batch_id,'name':name,'task':task,'executor':executor,'model':model,'provider_config':config,
           'evidence_mode':evidence_mode if task=='ground_truth' else None,
           'run_id':run['id'] if run else (identifier('run') if task=='inference' else None),
           'rubric_id':rubric['id'] if rubric else None,'rubric_sha256':rubric['sha256'] if rubric else None,
           'created_at':now(),'frame_ids':ids,'release_id':release_id,'input_track':next(iter(tracks)),
           'packet_manifest_sha256':digest([j['packet'] for j in jobs])}
        b['execution_config_sha256']=digest({'executor':executor,'model':model,'config':config,
            'system_instruction':jobs[0]['packet']['payload_template']['messages'][0]['content'],
            'schema':jobs[0]['packet']['expected_result_schema'],'rubric_sha256':b['rubric_sha256']})
        with self.store.transaction():
            self.put('batch',b)
            if config:self.put('provider',{'id':digest(config)[:20],'config':config})
            if task=='inference':
                initial={'id':b['run_id'],'name':name,'model':model,'track':'controlled','created_at':now(),
                         'predictions_count':0,'coverage':0,'protocol_tracks':sorted(tracks),'eligibility':'request_contract_matched',
                         'manifest_sha256':digest([]),'model_identity_status':'declared_by_executor','batch_id':batch_id,'planned_frame_ids':ids}
                self.db.execute('INSERT INTO runs VALUES(?,?)',(initial['id'],dumps(initial)))
            for j in jobs:self.save_job(j)
            self.event('create_batch',{'id':batch_id,'task':task,'jobs':len(jobs),'model_calls':0})
        return self.batch(batch_id)

    def claim(self, batch_id, worker, limit=1, *, job_id=None):
        worker=required_text(worker,'Worker');limit=int(limit)
        if job_id is not None:
            job_id=required_text(job_id,'Job ID')
            if limit!=1:raise ValueError('An exact job claim requires limit=1')
        # Keep older workers' 395-item ceiling requests valid for small batches;
        # expand the bound with the actual catalog when a collection is added.
        maximum=max(395,len(self.store.get_records('frames')))
        if not 1<=limit<=maximum:raise ValueError(f'Limit must be between 1 and {maximum}')
        b=self.record('batch',batch_id); claimed=[]
        with self.store.transaction():
            if job_id is not None:
                if self.job(job_id)['batch_id']!=batch_id:raise ValueError('Exact job is outside the requested batch')
                rows=self.db.execute("SELECT id FROM wf_jobs WHERE batch_id=? AND id=? AND status='pending'",(batch_id,job_id)).fetchall()
            else:
                rows=self.db.execute("SELECT id FROM wf_jobs WHERE batch_id=? AND status='pending' ORDER BY frame_id LIMIT ?",(batch_id,limit)).fetchall()
            for row in rows:
                j=self.job(row[0])
                # Reject obsolete work before a worker can spend an API call.
                # Completion repeats the check because input/reference state can
                # also change after the lease has been granted.
                self._current(j,b['task'])
                j.update(status='leased',lease_token=uuid.uuid4().hex,leased_at=time.time(),worker=worker,attempts=j['attempts']+1)
                j['history'].append({'event':'claimed','worker':worker,'at':now(),'attempt':j['attempts']})
                self.save_job(j);claimed.append(j)
        return {'batch':b,'jobs':claimed,'notice':'Return result with job_id and lease_token. Approval is never performed by job completion.'}

    def export_jobs(self, batch_id):
        b=self.record('batch',batch_id)
        jobs=[]
        for row in self.db.execute("SELECT id FROM wf_jobs WHERE batch_id=? AND status!='succeeded' ORDER BY frame_id",(batch_id,)):
            j=self.job(row[0]);p=copy.deepcopy(j['packet']);p.update(status=j['status'],image_base_directory=str(self.store.root))
            jobs.append(p)
        return {'batch':b,'jobs':jobs,'notice':'Read-only export, no leases or calls. Inspect images using attachment_manifest. Submit via Codex CLI or workflow result import. Do not approve AI drafts.'}

    def _current(self, j, task):
        contract=self.store.latest('input',j['frame_id'])
        if not contract or contract['input_hash']!=j['packet']['input_contract_sha256']:
            raise Conflict('Input changed after this job was created; create a new batch')
        gt=self.store.latest('gt',j['frame_id'])
        if task in ('ground_truth','gt_check') and (gt['revision'] if gt else 0)!=j['packet']['expected_gt_revision']:
            raise Conflict('Reference changed after this job was created; preserve the result and create a new batch')
        return contract,gt

    def _parsed_result(self, value):
        if not isinstance(value,dict):raise ValueError('Result must be an object')
        if 'parsed_output' in value:
            parsed=value['parsed_output']
            if not isinstance(parsed,dict):raise ValueError('Structured model output was not a JSON object; inspect the saved response')
            return parsed
        return value

    def _issue(self, frame_id, issue, source, gt, input_hash, target='reference'):
        if not isinstance(issue,dict):raise ValueError('Issue must be an object')
        severity=issue.get('severity','warning')
        if severity not in SEVERITIES:raise ValueError('Invalid issue severity')
        if target not in ('reference','candidate'):raise ValueError('Invalid issue target')
        message=required_text(issue.get('message'),'Issue message')
        content_hash=(gt or {}).get('content_sha256')
        iid='issue_'+digest([frame_id,content_hash,input_hash,source,target,message])[:24]
        existing=self.db.execute("SELECT data FROM wf_records WHERE kind='issue' AND id=?",(iid,)).fetchone()
        if existing:
            old=json.loads(existing[0])
            if SEVERITIES.index(severity)<SEVERITIES.index(old['severity']):
                old.update(severity=severity,category=issue.get('category',old['category']),status='open',
                           evidence=issue.get('evidence',old.get('evidence','')),suggested_fix=issue.get('suggested_fix',old.get('suggested_fix','')),
                           fact_ids=issue.get('fact_ids',old.get('fact_ids',[])))
                old['history'].append({'event':'severity_promoted','at':now(),'severity':severity});self.put('issue',old)
            return old
        entry={'id':iid,'frame_id':frame_id,'severity':severity,'target':target,'category':issue.get('category','review'),
               'message':message,'source':source,'status':'open','created_at':now(),'gt_revision':(gt or {}).get('revision'),
               'content_sha256':content_hash,'input_sha256':input_hash,'fact_ids':issue.get('fact_ids',[]),
               'evidence':issue.get('evidence',''),'suggested_fix':issue.get('suggested_fix',''),'history':[]}
        self.put('issue',entry);return entry

    def _import_prediction(self, batch, j, result, worker, model_version):
        output=result.get('output_text',result.get('output'))
        if not isinstance(output,str):raise ValueError('Inference result needs exact output_text')
        native=result.get('request_log')
        if batch['executor']!='codex' and (not isinstance(native,dict) or native.get('kind')!='native_execution'):
            raise ValueError('API inference requires its native request log')
        if batch['executor']!='codex':
            config=batch['provider_config']
            for key,expected in [('provider',batch['executor']),('model',batch['model']),('job_payload_sha256',j['packet']['payload_sha256'])]:
                if native.get(key)!=expected:raise ValueError('Execution log '+key+' does not match this job')
            if native.get('generation_settings',native.get('settings'))!=config['settings']:
                raise ValueError('Execution settings do not match the frozen batch configuration')
        if native is None:
            native={'kind':'native_execution','execution_source':'codex_submission','worker':worker,'input_sha256':j['packet']['input_contract_sha256'],
                    'job_payload_sha256':j['packet']['payload_sha256'],'observed_model':model_version,'provenance_status':'declared_by_submitter'}
        metadata={'model_id':batch['model'],'observed_model':native.get('observed_model',model_version),
                  'generation_settings':(batch.get('provider_config') or {}).get('settings',{'execution_source':'codex'})}
        # Reuse all controlled-import validation, then attach the validated row to the durable batch run.
        temporary=self.store.import_run({'name':batch['name'],'model':batch['model'],'track':'controlled','records':[
            {'frame_id':j['frame_id'],'output_text':output,'status':'succeeded','input_sha256':j['packet']['input_contract_sha256'],
             'model_metadata':metadata,'request_log':native,'logs':[{'kind':'native_execution','job_id':j['id']}]}]})
        pred=self.store.predictions(temporary['id'])[0];pred['run_id']=batch['run_id']
        self.db.execute('DELETE FROM predictions WHERE run_id=?',(temporary['id'],));self.db.execute('DELETE FROM runs WHERE id=?',(temporary['id'],))
        self.db.execute('INSERT INTO predictions VALUES(?,?,?)',(batch['run_id'],j['frame_id'],dumps(pred)))
        run=json.loads(self.db.execute('SELECT data FROM runs WHERE id=?',(batch['run_id'],)).fetchone()[0])
        predictions=self.store.predictions(batch['run_id']);run.update(predictions_count=len(predictions),coverage=len(predictions)/len(self.store.get_records('frames')),manifest_sha256=digest(predictions))
        self.db.execute('UPDATE runs SET data=? WHERE id=?',(dumps(run),run['id']))
        return {'run_id':run['id'],'prediction_sha256':pred['prediction_sha256']}

    def complete(self, payload):
        item_id=payload.get('job_id');worker=required_text(payload.get('worker'),'Worker')
        model_version=required_text(payload.get('model_version'),'Actual model/version used')
        result=payload.get('result');result_hash=digest(result)
        with self.store.transaction():
            j=self.job(item_id);b=self.record('batch',j['batch_id'])
            if j['status']=='succeeded':
                if j.get('result_sha256')!=result_hash:raise Conflict('A different result is already recorded for this job')
                return {'job_id':item_id,'status':'succeeded','duplicate':True,'imported':j.get('imported')}
            if j['status']=='pending' and b['executor']=='codex':
                # UI import claims a still-pending external Codex result atomically.
                j.update(status='leased',lease_token=uuid.uuid4().hex,leased_at=time.time(),worker=worker,attempts=j['attempts']+1)
            elif j['status']!='leased' or j.get('lease_token')!=payload.get('lease_token') or j.get('worker')!=worker:
                raise Conflict('Claim this job first, using its current worker and lease token')
            contract,gt=self._current(j,b['task'])
            if b['executor']!='codex':
                if not isinstance(result,dict) or not isinstance(result.get('request_log'),dict):raise ValueError('API results require a native receipt')
                native=result['request_log'];config=b['provider_config']
                for key,expected in [('kind','native_execution'),('provider',b['executor']),('model',b['model']),
                    ('job_payload_sha256',j['packet']['payload_sha256']),('input_sha256',contract['input_hash']),('settings',config['settings']),('api_mode',config['api_mode'])]:
                    if native.get(key)!=expected:raise ValueError('Native receipt '+key+' does not match this job/configuration')
                endpoint=config['base_url']+('/responses' if config['api_mode']=='responses' else '/chat/completions')
                if native.get('endpoint')!=endpoint:raise ValueError('Native receipt endpoint differs from this batch')
                observed=native.get('observed_model') or b['model']+' (snapshot not exposed by provider)'
                if model_version!=observed:raise ValueError('Reported model version differs from the provider receipt')
            provenance={'executor':b['executor'],'model':b['model'],'model_version':model_version,'job_id':item_id,'batch_id':b['id'],
                        'payload_sha256':j['packet']['payload_sha256'],'result_sha256':result_hash,'worker':worker,'rubric_sha256':b.get('rubric_sha256'),
                        'evidence_mode':b.get('evidence_mode')}
            if b.get('observed_model_version') and b['observed_model_version']!=model_version:
                raise Conflict('Actual model version differs from earlier results in this batch; preserve the receipt and use a separate run')
            b['observed_model_version']=model_version;self.put('batch',b)
            imported={};parsed=None
            if b['task']=='inference':imported=self._import_prediction(b,j,result,worker,model_version)
            else:
                parsed=copy.deepcopy(self._parsed_result(result))
                if b['task']=='ground_truth':
                    content=parsed.get('content')
                    if not isinstance(content,dict):raise ValueError('GT result requires content object')
                    gt=self.store.save_gt(j['frame_id'],{'status':'draft','actor_type':'assistant','reviewer':worker,'content':content,
                        'expected_revision':j['packet']['expected_gt_revision'],'provenance':provenance,'notes':'Assisted draft; explicit human review required.'})
                    imported={'gt_revision':gt['revision'],'content_sha256':gt['content_sha256']}
                    for uncertainty in content.get('uncertainties',[]):
                        if not isinstance(uncertainty,str):raise ValueError('Uncertainties must be text')
                        self._issue(j['frame_id'],{'severity':'warning','category':'uncertainty','message':uncertainty},item_id,gt,contract['input_hash'])
                elif b['task']=='judge':
                    bindings=j['packet']['bindings']
                    if not isinstance(parsed.get('claims'),list) or any(not isinstance(c,dict) or not isinstance(c.get('fact_ids'),list) for c in parsed['claims']):
                        raise ValueError('Every new judge claim must include fact_ids (empty for unsupported claims)')
                    rubric_content=self.record('rubric',b['rubric_id'])['content']
                    if rubric_content.get('analytic_criteria'):
                        from .criteria import validate_assessments
                        parsed['criterion_assessments']=validate_assessments(parsed.get('criterion_assessments'),rubric_content['analytic_criteria'])
                    graded=rubric_content.get('scoring',{}).get('version')=='mousebench.graded-factual-f1.v2'
                    if graded and 'supported_fact_ids' in parsed:
                        raise ValueError('Graded judge results must use fact_coverage, not legacy supported_fact_ids')
                    if not graded and 'fact_coverage' in parsed:
                        raise ValueError('Legacy judge results cannot silently import graded fact_coverage')
                    fields=('claims','fact_coverage' if graded else 'supported_fact_ids','structure_score','instruction_score','notes','criterion_assessments')
                    judgment={k:parsed[k] for k in fields if k in parsed}
                    judgment.update(schema_version='mousebench.judgment.v2' if graded else 'mousebench.judgment.v1',frame_id=j['frame_id'],run_id=b['run_id'],
                        protocol_version=bindings['protocol_version'],rubric_sha256=bindings['rubric_sha256'],
                        ground_truth_sha256=bindings['ground_truth_sha256'],prediction_sha256=bindings['prediction_sha256'],
                        input_sha256=contract['input_hash'],judge={'provider':b['executor'],'model':b['model'],
                            'version':model_version+'|config:'+b['execution_config_sha256']})
                    imported=self.store.import_judgments({'records':[judgment]})
                issues=parsed.get('issues',[])
                if not isinstance(issues,list):raise ValueError('Issues must be a list')
                if b['task']=='gt_check' and 'issues' not in parsed:raise ValueError('GT check must explicitly return issues, even when empty')
                for issue in issues:
                    target=issue.get('target','candidate') if b['task']=='judge' and isinstance(issue,dict) else 'reference'
                    self._issue(j['frame_id'],issue,item_id,gt,contract['input_hash'],target=target)
                imported['issues']=len(issues)
            j.update(status='succeeded',result_sha256=result_hash,raw_result=result,provenance=provenance,imported=imported,completed_at=now())
            j['history'].append({'event':'completed','at':now(),'result_sha256':result_hash});self.save_job(j)
            self.event('complete_job',{'id':item_id,'task':b['task'],'imported':imported})
        return {'job_id':item_id,'status':'succeeded','imported':imported}

    def record_receipt(self, job_id, result, model_version, worker):
        """Persist provider response before parsing/importing; re-ingestion needs no call."""
        j=self.job(job_id)
        receipt={'id':job_id+'_'+str(j['attempts']),'job_id':job_id,'result':result,'model_version':model_version,
                 'worker':worker,'created_at':now(),'result_sha256':digest(result),'attempt':j['attempts']}
        with self.store.transaction():
            existing=self.db.execute("SELECT data FROM wf_records WHERE kind='receipt' AND id=?",(receipt['id'],)).fetchone()
            if existing:
                old=json.loads(existing[0])
                if old['result_sha256']!=receipt['result_sha256']:raise Conflict('Receipt already exists with different output')
                return old
            self.put('receipt',receipt)
        return receipt

    def ingest_receipts(self, batch_id):
        b=self.record('batch',batch_id);imported=[];errors=[]
        for row in self.db.execute("SELECT id FROM wf_jobs WHERE batch_id=? AND status!='succeeded'",(b['id'],)).fetchall():
            j=self.job(row[0]);rid=j['id']+'_'+str(j['attempts'])
            try:receipt=self.record('receipt',rid)
            except KeyError:continue
            try:
                with self.store.transaction():
                    j.update(status='leased',worker=receipt['worker'],lease_token=j.get('lease_token') or uuid.uuid4().hex)
                    self.save_job(j)
                    imported.append(self.complete({'job_id':j['id'],'lease_token':j['lease_token'],'worker':receipt['worker'],
                        'model_version':receipt['model_version'],'result':receipt['result']}))
            except (ValueError,KeyError,TypeError) as exc:errors.append({'job_id':j['id'],'error':str(exc)})
        return {'imported':len(imported),'errors':errors,'model_calls':0}

    def fail(self, payload):
        with self.store.transaction():
            j=self.job(payload.get('job_id'))
            if j['status']!='leased' or j['lease_token']!=payload.get('lease_token'):
                raise Conflict('Failure receipt needs the active lease')
            j.update(status='uncertain' if payload.get('uncertain') else 'failed',error={'code':payload.get('code','worker_error'),
                'retryable':bool(payload.get('retryable')),'uncertain':bool(payload.get('uncertain'))},last_failure_at=now())
            if 'result' in payload:j['raw_result']=payload['result']
            j['history'].append({'event':'failed','at':now(),'error':j['error']});self.save_job(j)
        return {'job_id':j['id'],'status':j['status'],'error':j['error']}

    def retry_jobs(self, payload):
        reviewer=required_text(payload.get('reviewer'),'Reviewer')
        b=self.record('batch',payload.get('batch_id'));ids=payload.get('job_ids')
        if ids is None:ids=[r[0] for r in self.db.execute("SELECT id FROM wf_jobs WHERE batch_id=? AND status='failed'",(b['id'],))]
        reset=[]
        with self.store.transaction():
            for iid in ids:
                j=self.job(iid)
                if j['batch_id']!=b['id']:raise ValueError('Job is outside this batch')
                if j['status'] not in ('failed','uncertain','leased'):raise ValueError('Only failed or interrupted jobs can be retried')
                if j['status'] in ('uncertain','leased') and payload.get('acknowledge_uncertain') is not True:
                    raise ValueError('An interrupted call may have completed remotely; explicitly acknowledge before retrying')
                j['history'].append({'event':'retry_authorized','reviewer':reviewer,'at':now(),'previous_status':j['status'],'previous_error':j.get('error')})
                j.update(status='pending',lease_token=None,leased_at=None);self.save_job(j);reset.append(iid)
            self.event('retry_jobs',{'ids':reset,'reviewer':reviewer})
        return {'pending':len(reset),'job_ids':reset}

    def submit_results(self, payload):
        records=payload.get('records')
        if not isinstance(records,list) or not records:raise ValueError('Submit a nonempty records array')
        result=[]
        # Whole batch import is atomic; an invalid result leaves every job unchanged.
        with self.store.transaction():
            for record in records:
                if not isinstance(record,dict):raise ValueError('Each result must be an object')
                j=self.job(record.get('job_id'))
                if payload.get('batch_id') and j['batch_id']!=payload['batch_id']:raise ValueError('Result is outside this batch')
                result.append(self.complete(record))
        return {'imported':len(result),'records':result}

    def issues(self, current_only=False):
        result=[]
        references=self.store.latest_versions('gt');inputs=self.store.latest_versions('input')
        for issue in self.records('issue'):
            issue.setdefault('target','reference')
            gt=references.get(issue['frame_id']);contract=inputs.get(issue['frame_id'])
            current=bool(gt and contract and gt['content_sha256']==issue.get('content_sha256') and contract['input_hash']==issue.get('input_sha256'))
            issue['current']=current
            if not current_only or current:result.append(issue)
        return sorted(result,key=lambda i:(i['status']!='open',SEVERITIES.index(i['severity']),i['frame_id'],i['created_at']))

    def resolve_issue(self, payload):
        entry=self.record('issue',payload.get('id'));status=payload.get('status')
        if status not in ('resolved','dismissed'):raise ValueError('Choose resolved or dismissed')
        reviewer=required_text(payload.get('reviewer'),'Reviewer');reason=required_text(payload.get('resolution'),'Resolution or dismissal reason')
        entry['history'].append({'status':status,'reviewer':reviewer,'resolution':reason,'at':now()})
        entry.update(status=status,reviewer=reviewer,resolution=reason,resolved_at=now())
        with self.store.transaction():self.put('issue',entry)
        return entry

    def import_calibration(self, payload):
        rubric=self.rubric(payload.get('rubric_id'));records=payload.get('records')
        if rubric['status']!='frozen':raise ValueError('Freeze the rubric before calibration')
        if not isinstance(records,list):raise ValueError('Calibration records must be an array')
        # Verify actual stored output hashes and recording IDs; do not trust caller identities.
        clean=[];managed_templates={}
        for record in records:
            if not isinstance(record,dict):raise ValueError('Each calibration record must be an object')
            r=copy.deepcopy(record)
            managed='managed_binding' in r or str(r.get('id','')).startswith('managed_calibration_')
            if managed:
                binding=r.get('managed_binding')
                if not isinstance(binding,dict) or not isinstance(binding.get('judgment_id'),str):
                    raise ValueError('Managed calibration requires its original judgment binding')
                row=self.db.execute('SELECT data FROM judgments WHERE id=?',(binding['judgment_id'],)).fetchone()
                if row is None:raise ValueError('Managed calibration judgment no longer exists')
                judgment=json.loads(row[0]);run_id=judgment['run_id']
                if run_id not in managed_templates:
                    managed_templates[run_id]={item['id']:item for item in self.calibration_template(run_id)['records']}
                expected=managed_templates[run_id].get(r.get('id'))
                if expected is None:raise ValueError('Managed calibration is stale or has an invalid criterion binding')
                editable={'split','human_label','verified_by','human_verified'}
                for key,value in expected.items():
                    if key not in editable and r.get(key)!=value:
                        raise ValueError('Managed calibration '+key+' differs from the authoritative stored record')
                for key in ('claim_id','claim_text','fact_id','fact_text','dimension','other_prediction_sha256'):
                    if r.get(key)!=expected.get(key):
                        raise ValueError('Managed calibration criterion differs from the authoritative stored record')
                criterion=binding['criterion_id']
                if criterion.startswith('correctness:'):
                    claim_id=criterion.split(':',1)[1]
                    label=next(c['correctness'] for c in judgment['claims'] if c['id']==claim_id)
                elif criterion.startswith('coverage:'):
                    fact_id=criterion.split(':',1)[1]
                    label=next(c['coverage'] for c in judgment['fact_coverage'] if c['fact_id']==fact_id)
                elif expected['kind']=='claim':
                    label=judgment['claims'][int(criterion[1:])-1]['label']
                else:label=judgment[criterion]
                for key,value in (('judge',judgment['judge']),('judge_label',label)):
                    if key in r and r[key]!=value:
                        raise ValueError('Managed calibration '+key+' cannot override the authoritative stored judgment')
                    r[key]=copy.deepcopy(value)
            fid=r.get('frame_id');frame=self.store.frame(fid)
            contract=self.store.latest('input',fid)
            if not contract or contract['input_hash']!=r.get('input_sha256'):raise ValueError('Calibration input is not the frozen input')
            outputs=[json.loads(p[0]) for p in self.db.execute('SELECT data FROM predictions WHERE frame_id=?',(fid,))]
            output_hashes={p['prediction_sha256'] for p in outputs if p.get('status')=='succeeded' and p.get('input_sha256',p.get('input_hash'))==r['input_sha256']}
            for link in self.records('evidence_link'):
                if link['frame_id']==fid and link.get('input_verified') and link.get('input_sha256')==r['input_sha256']:
                    output_hashes.update(link['prediction_hashes'])
            if r.get('prediction_sha256') not in output_hashes or (r.get('kind')=='pairwise' and r.get('other_prediction_sha256') not in output_hashes):
                raise ValueError('Calibration must bind to existing exact candidate outputs under the same verified input; historical input must be reconciled first')
            r['recording_id']=frame.get('recording_id') or fid.rsplit('_',1)[0];clean.append(r)
        report=calibration_report(clean,rubric['content'])
        entry={'id':'calibration_'+digest([rubric['sha256'],clean])[:20],'rubric_id':rubric['id'],'rubric_sha256':rubric['sha256'],
               'created_at':now(),'records':clean,'report':report}
        with self.store.transaction():self.put('calibration',entry)
        return entry

    def verify_evidence_link(self,payload):
        fid=payload.get('frame_id');self.store.frame(fid)
        reviewer=required_text(payload.get('reviewer'),'Reviewer');reason=required_text(payload.get('reason'),'Evidence for this exact match')
        if payload.get('human_confirmed') is not True:raise ValueError('Confirm that you verified the exact output, not only its frame or folder label')
        ids=payload.get('rating_ids')
        if not isinstance(ids,list) or not ids or len(ids)!=len(set(ids)):raise ValueError('Choose distinct source rating IDs')
        ratings={r['id']:r for r in self.store.get_records('ratings',fid)}
        if any(i not in ratings for i in ids):raise ValueError('Rating is not attached to this frame')
        if any(ratings[i].get('kind')=='pairwise' for i in ids) and not payload.get('other_run_id'):
            raise ValueError('Pairwise expert evidence needs both candidate runs, preserving A/B order')
        runs=[payload.get('run_id')]
        if payload.get('other_run_id'):runs.append(payload['other_run_id'])
        if len(set(runs))!=len(runs):raise ValueError('Choose different A/B runs')
        predictions=[]
        for run_id in runs:
            match=next((p for p in self.store.predictions(run_id) if p['frame_id']==fid and p['status']=='succeeded'),None)
            if match is None:raise ValueError('Run has no successful output for this frame')
            predictions.append(match)
        contract=self.store.latest('input',fid)
        if payload.get('input_verified') and not contract:raise ValueError('Freeze the exact input before verifying historical input correspondence')
        if payload.get('input_verified') and any(p.get('track')=='controlled' and p.get('input_sha256',p.get('input_hash'))!=contract['input_hash'] for p in predictions):
            raise ValueError('A controlled output cannot be relabeled under a different input')
        if payload.get('input_verified'):
            from .recorded_inputs import ADAPTED_PROTOCOL
            for prediction in predictions:
                source_hash = prediction.get('input_sha256') or prediction.get('input_hash')
                if ((source_hash and source_hash != contract['input_hash'])
                        or (contract.get('protocol_version') == ADAPTED_PROTOCOL and source_hash != contract['input_hash'])):
                    raise ValueError('An output from an older or unknown request cannot be verified as the adapted current input.')
        entry={'id':identifier('evidence'),'frame_id':fid,'rating_ids':ids,'source_rating_hashes':{i:digest(ratings[i]) for i in ids},
               'run_id':runs[0],'other_run_id':runs[1] if len(runs)>1 else None,'prediction_hashes':[p['prediction_sha256'] for p in predictions],
               'reviewer':reviewer,'reason':reason,'human_confirmed':True,'input_verified':payload.get('input_verified') is True,
               'input_sha256':contract['input_hash'] if payload.get('input_verified') else None,'created_at':now(),
               'note':'Human-declared provenance decision. A/B order follows run_id/other_run_id; repeated exports remain copies.'}
        with self.store.transaction():self.put('evidence_link',entry)
        return entry

    def calibration_template(self,run_id):
        """Export sensory context for human annotation without exposing judge verdicts."""
        rows=[json.loads(r[0]) for r in self.db.execute('SELECT data FROM judgments WHERE run_id=?',(run_id,))]
        predictions={p['frame_id']:p for p in self.store.predictions(run_id)};records=[]
        for j in rows:
            gt=self.store.latest('gt',j['frame_id']);contract=self.store.latest('input',j['frame_id'])
            if not gt or gt['status']!='approved' or not contract or j['input_sha256']!=contract['input_hash'] or j['ground_truth_sha256']!=gt['content_sha256']:continue
            if self.store.has_critical_issue(j['frame_id'],gt['content_sha256'],contract['input_hash']):continue
            prediction=predictions.get(j['frame_id']);image=self.store.image(j['frame_id'])
            if not prediction or prediction.get('status')!='succeeded' or prediction['prediction_sha256']!=j['prediction_sha256']:continue
            if not image or image['sha256']!=contract['image_sha256']:continue
            from .recorded_inputs import verify_input_images
            verify_input_images(self.store,contract)
            # Keep only candidate-visible sensory fields, never source paths,
            # reviewer notes, model metadata, reference content or judge scores.
            from .providers import _sensory_contract
            sensory=_sensory_contract(contract)
            context={'canonical_input':sensory,'image':{'path':image['path'],'sha256':image['sha256']},
                     'candidate_text':prediction.get('output_text',prediction.get('output'))}
            if contract.get('input_images'):
                context['input_images']=[{k:v for k,v in img.items() if k in ('attachment_id','path','sha256','image_detail','media_type')} for img in contract['input_images']]
            base={'schema_version':'mousebench.calibration.v1','frame_id':j['frame_id'],'recording_id':j['recording_id'],
                  'split':None,'input_sha256':j['input_sha256'],'prediction_sha256':j['prediction_sha256'],
                  'ground_truth_sha256':j['ground_truth_sha256'],'rubric_sha256':j['rubric_sha256'],
                  'human_verified':False,'verified_by':'','human_label':None,'context':context}
            def managed(kind,criterion,**fields):
                binding={'judgment_id':j['id'],'judgment_sha256':digest(j),'criterion_id':criterion}
                return dict(copy.deepcopy(base),id='managed_calibration_'+digest([j['id'],kind,criterion]),
                            kind=kind,managed_binding=binding,**fields)
            if j['schema_version']=='mousebench.judgment.v2':
                # Export the assessed point, never the judge's grade, severity,
                # supporting rationale or claim-to-reference credit assignment.
                # Each unique point is calibrated once, matching scoring dedup.
                from .evaluation import _normalized_claim
                seen=set()
                for claim in j['claims']:
                    key=_normalized_claim(claim['text'])
                    if key in seen:continue
                    seen.add(key)
                    records.append(managed('dimension','correctness:'+claim['id'],
                        dimension='factual_correctness',claim_id=claim['id'],claim_text=claim['text']))
                required={f['id']:f for f in gt['content']['facts'] if f['required']}
                for point in j['fact_coverage']:
                    fact=required[point['fact_id']]
                    entry=managed('dimension','coverage:'+fact['id'],dimension='required_fact_coverage',
                        fact_id=fact['id'],fact_text=fact['text'])
                    entry['context']['required_fact']={k:v for k,v in fact.items()
                        if k in ('id','text','field','required','evidence','evidence_note','accepted_alternatives')}
                    records.append(entry)
            else:
                for index,claim in enumerate(j['claims']):
                    criterion='c'+str(index+1)
                    records.append(managed('claim',criterion,claim_id=criterion,claim_text=claim['text']))
            for dimension in ('structure_score','instruction_score'):
                records.append(managed('dimension',dimension,dimension=dimension))
        return {'schema_version':'mousebench.calibration-template.v2','blinded':True,'records':records,
                'editable_fields':['split','human_label','verified_by','human_verified'],
                'notice':'Judge identity and verdicts are withheld. Inspect the canonical image, task and exact candidate independently; fill human_label, verified_by and human_verified yourself. Explicitly choose development or heldout with recording-level separation. Preserve IDs, bindings and context. Import joins the original stored judge verdict only after validating these bindings; filter to one rubric/judge configuration.'}

    def create_release(self, payload):
        reviewer=required_text(payload.get('reviewer'),'Reviewer');name=required_text(payload.get('name'),'Release name')
        rubric=self.rubric(payload.get('rubric_id'))
        if rubric['status']!='frozen':raise ValueError('Freeze the rubric first')
        ids=self.frame_ids(payload.get('frame_ids'));samples=[];missing=[]
        for fid in ids:
            gt=self.store.latest('gt',fid);contract=self.store.latest('input',fid);image=self.store.image(fid)
            if not gt or gt['status']!='approved' or not contract or gt['input_hash']!=contract['input_hash'] or not image:
                missing.append(fid);continue
            from .recorded_inputs import verify_input_images
            verify_input_images(self.store,contract)
            if self.store.has_critical_issue(fid,gt['content_sha256'],contract['input_hash']):
                raise ValueError('Release has an unresolved critical issue for '+fid)
            samples.append({'frame_id':fid,'recording_id':self.store.frame(fid).get('recording_id'),'input':contract,'ground_truth':gt,'image':image})
        if payload.get('require_complete',True) and missing:raise ValueError('Approve all selected references before releasing; missing '+str(len(missing)))
        if not samples:raise ValueError('A release needs at least one approved reference')
        release={'id':identifier('release'),'name':name,'reviewer':reviewer,'created_at':now(),'rubric':rubric,'samples':samples,
                 'selected_frame_ids':ids,'missing_frame_ids':missing,'complete':not missing,'scope':'manually_selected_frame_description',
                 'calibration_ids':[c['id'] for c in self.records('calibration') if c['rubric_sha256']==rubric['sha256']],
                 'scientific_validation':'provisional','training_independence':'unverified'}
        release['sha256']=digest(release)
        with self.store.transaction():self.put('release',release)
        return release

    def scorecards(self):
        snapshot=self.store.scorecard_snapshot()
        cards=self.store.scorecards(snapshot=snapshot)
        critical_bindings=snapshot['critical_bindings']
        rubrics={r['sha256']:r for r in self.records('rubric')}
        indexers={sha:prepare_benchmark_index(r['content']) for sha,r in rubrics.items()}
        rows_by_run={}
        for row in snapshot['entries']:
            rows_by_run.setdefault(row['run_id'],[]).append(row)
        calibrations=self.records('calibration')
        for card in cards['models']:
            group=next(g for g in cards['groups'] if g['group_id']==card['group_id']);rubric=rubrics.get(group['rubric_sha256'])
            indices=[];index_by_frame={};criterion_rows=[]
            if rubric:
                for j in rows_by_run.get(card['run_id'],[]):
                    if (j['run_id']!=card['run_id'] or j['rubric_sha256']!=rubric['sha256'] or j['judge']!=group['judge']
                        or j['protocol_version']!=group['protocol_version'] or j.get('input_track')!=group.get('input_track')):continue
                    gt=snapshot['references'].get(j['frame_id']);contract=snapshot['inputs'].get(j['frame_id'])
                    if not gt or not contract or gt['status']!='approved' or gt['content_sha256']!=j['ground_truth_sha256'] or gt['input_hash']!=contract['input_hash'] or contract['input_hash']!=j['input_sha256']:continue
                    if (j['frame_id'],gt['content_sha256'],contract['input_hash']) in critical_bindings:continue
                    criterion_rows.append(j.get('criterion_assessments'))
                    result=indexers[rubric['sha256']](j['scores'])
                    if result['value'] is not None:
                        indices.append(result['value']);index_by_frame[j['frame_id']]=result['value']
            card['benchmark_index']=statistics.mean(indices) if indices else None
            card['index_count']=len(indices)
            recording_indices={}
            for fid,index_value in index_by_frame.items():
                recording_indices.setdefault(snapshot['frames'][fid].get('recording_id') or fid.rsplit('_',1)[0],[]).append(index_value)
            card['recording_balanced_benchmark_index']=statistics.mean(
                statistics.mean(values) for values in recording_indices.values()) if recording_indices else None
            card['index_recording_count']=len(recording_indices)
            if rubric and rubric['content'].get('analytic_criteria'):
                from .criteria import summarize_assessments
                card['criterion_results']=summarize_assessments(criterion_rows,rubric['content']['analytic_criteria'])
            common=[index_by_frame[f] for f in group['common_frame_ids'] if f in index_by_frame]
            card['common_benchmark_index']=statistics.mean(common) if common else None
            card['common_index_count']=len(common)
            card['index_formula']=rubric['content']['scoring']['index']['formula'] if rubric else None
            card['index_weights']={k:rubric['content']['scoring']['index'][k] for k in ('base','structure_weight','instruction_weight')} if rubric else None
            card['completion_adjusted_index']=card['benchmark_index']*card['benchmark_coverage'] if card['benchmark_index'] is not None else None
            matching=[c for c in calibrations if c['rubric_sha256']==group['rubric_sha256'] and c['report'].get('judge')==group['judge']]
            card['calibration_reports']=[c['id'] for c in matching]
            run=snapshot['runs'][card['run_id']]
            if run.get('batch_id'):
                b=self.batch(run['batch_id'],False)
                attempts=[row[0] for row in self.db.execute("SELECT json_extract(data,'$.attempts') FROM wf_jobs WHERE batch_id=?",(run['batch_id'],))]
                card.update(failed_count=b['batch']['counts']['failed'],uncertain_count=b['batch']['counts']['uncertain'],
                    planned_count=len(b['batch']['frame_ids']),attempted_count=sum(value>0 for value in attempts),
                    execution_attempts=sum(attempts),pending_count=b['batch']['counts']['pending'],
                    leased_count=b['batch']['counts']['leased'])
        cards['batches']=[self.batch(b['id'],False)['batch'] for b in self.records('batch')]
        cards['index_note']='Proposed composite. Report factual F1 and every component, coverage and failure rates. Completion-adjusted index treats unevaluated catalog frames as zero utility, not as semantic judgments.'
        return cards

    def dispatch(self, payload):
        action=payload.get('action')
        actions={'save_rubric':self.save_rubric,'freeze_rubric':self.freeze_rubric,'freeze_inputs':self.freeze_inputs,
                 'freeze_recorded_inputs':self.freeze_recorded_inputs,
                 'create_batch':self.create_batch,'submit_results':self.submit_results,'resolve_issue':self.resolve_issue,
                 'retry_jobs':self.retry_jobs,'import_calibration':self.import_calibration,'create_release':self.create_release,
                 'verify_evidence_link':self.verify_evidence_link,'ingest_receipts':lambda p:self.ingest_receipts(p.get('batch_id'))}
        if action not in actions:raise ValueError('Unknown workflow action')
        return actions[action](payload)
