"""Local persistence and scientific eligibility gates. No model API calls."""
import hashlib
import json
import sqlite3
import uuid
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

FIELDS = ('Overall', 'Head', 'Limb', 'Torso', 'Others', 'Keywords')

def now():
    return datetime.now(timezone.utc).isoformat()

def dumps(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)

def digest(obj):
    return hashlib.sha256(dumps(obj).encode()).hexdigest()

def normalize_tags(tags):
    """Collection membership is explicit metadata, independent of GT/input versions."""
    if tags is None:
        return []
    if not isinstance(tags, list) or any(not isinstance(tag, str) or not tag.strip() for tag in tags):
        raise ValueError('Frame tags must be a list of non-empty strings')
    return sorted({tag.strip() for tag in tags})

def frame_metadata(frame):
    return dict(frame, tags=normalize_tags(frame.get('tags')))

def recorded_timestamp(value):
    """Comparable UTC time for display metadata; absent/invalid dates stay unknown."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)

def reasoning_effort_metadata(settings_records):
    """Do not confuse an omitted provider parameter with an explicit 'none'."""
    states = set()
    for settings in settings_records:
        if not isinstance(settings, dict):
            states.add(('unrecorded', None))
        elif 'reasoning_effort' not in settings:
            states.add(('not_requested', None))
        elif isinstance(settings['reasoning_effort'], str) and settings['reasoning_effort'].strip():
            states.add(('explicit', settings['reasoning_effort']))
        else:
            states.add(('unrecorded', None))
    if not states:
        states.add(('unrecorded', None))
    status, effort = next(iter(states)) if len(states) == 1 else ('mixed', None)
    return {'reasoning_effort': effort, 'reasoning_effort_status': status}

class Conflict(ValueError):
    pass

class Store:
    def __init__(self, data_dir):
        self.root = Path(data_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.root / 'benchmark.sqlite3'), timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS records(kind TEXT, id TEXT, frame_id TEXT, data TEXT NOT NULL, PRIMARY KEY(kind,id));
        CREATE INDEX IF NOT EXISTS frame_records ON records(kind,frame_id);
        CREATE TABLE IF NOT EXISTS versions(kind TEXT, frame_id TEXT, revision INTEGER, data TEXT NOT NULL, PRIMARY KEY(kind,frame_id,revision));
        CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, data TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS predictions(run_id TEXT, frame_id TEXT, data TEXT NOT NULL, PRIMARY KEY(run_id,frame_id));
        CREATE TABLE IF NOT EXISTS judgments(id TEXT PRIMARY KEY, run_id TEXT, frame_id TEXT, data TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, data TEXT NOT NULL);
        ''')
    def close(self):
        self.db.close()
    @contextmanager
    def transaction(self):
        """Nested writes stay atomic with a workflow receipt and its scientific record."""
        nested = self.db.in_transaction
        savepoint = 'sp_' + uuid.uuid4().hex
        self.db.execute('SAVEPOINT '+savepoint if nested else 'BEGIN IMMEDIATE')
        try:
            yield
            if nested: self.db.execute('RELEASE SAVEPOINT '+savepoint)
            else: self.db.commit()
        except Exception:
            if nested:
                self.db.execute('ROLLBACK TO SAVEPOINT '+savepoint)
                self.db.execute('RELEASE SAVEPOINT '+savepoint)
            else: self.db.rollback()
            raise
    def get_records(self, kind, frame_id=None):
        if frame_id is None:
            rows = self.db.execute('SELECT data FROM records WHERE kind=? ORDER BY id', (kind,))
        else:
            rows = self.db.execute('SELECT data FROM records WHERE kind=? AND frame_id=? ORDER BY id', (kind, frame_id))
        values = [json.loads(r[0]) for r in rows]
        return [frame_metadata(value) for value in values] if kind == 'frames' else values
    def frame(self, frame_id):
        r = self.db.execute("SELECT data FROM records WHERE kind='frames' AND id=?", (frame_id,)).fetchone()
        if not r:
            raise KeyError('Unknown frame')
        return frame_metadata(json.loads(r[0]))
    def import_catalog(self, catalog):
        frames = catalog['frames']
        ids = {f['id'] for f in frames}
        if len(ids) != len(frames):
            raise ValueError('Duplicate frame IDs')
        with self.transaction():
            for kind in ('frames', 'assets', 'ratings'):
                for item in catalog.get(kind, []):
                    if kind != 'frames' and item['frame_id'] not in ids:
                        raise ValueError('Record references unknown frame')
                    if kind == 'frames':
                        old = self.db.execute("SELECT data FROM records WHERE kind='frames' AND id=?", (item['id'],)).fetchone()
                        previous = json.loads(old[0]) if old else {}
                        # Partial/repeated source imports may refresh metadata, but must not
                        # remove collection membership or unrelated locally recorded provenance.
                        tags = normalize_tags(previous.get('tags')) + normalize_tags(item.get('tags'))
                        item = dict(previous, **item)
                        item['tags'] = normalize_tags(tags)
                    # Re-import refreshes source metadata; locally downloaded objects remain separate.
                    self.db.execute('INSERT OR REPLACE INTO records VALUES(?,?,?,?)',
                        (kind,item['id'],item.get('frame_id') if kind!='frames' else item['id'],dumps(item)))
            self.db.execute('INSERT OR REPLACE INTO meta VALUES(?,?)', ('catalog_summary',dumps(catalog.get('summary',{}))))
        return self.summary()
    def latest(self, kind, frame_id):
        row = self.db.execute('SELECT data FROM versions WHERE kind=? AND frame_id=? ORDER BY revision DESC LIMIT 1', (kind, frame_id)).fetchone()
        return json.loads(row[0]) if row else None
    def latest_versions(self, kind):
        """Read current revisions once for one request, without persistent caching."""
        rows = self.db.execute('SELECT v.frame_id,v.data FROM versions v JOIN (SELECT frame_id,max(revision) revision FROM versions WHERE kind=? GROUP BY frame_id) latest ON v.frame_id=latest.frame_id AND v.revision=latest.revision WHERE v.kind=?', (kind, kind))
        return {row[0]: json.loads(row[1]) for row in rows}

    def versions(self, kind, frame_id):
        return [json.loads(r[0]) for r in self.db.execute('SELECT data FROM versions WHERE kind=? AND frame_id=? ORDER BY revision DESC', (kind,frame_id))]
    def image(self, frame_id):
        self.frame(frame_id)
        items = self.get_records('local_images', frame_id)
        if not items:
            return None
        entry = items[0]
        path = (self.root / entry['path']).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            return None
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError('Canonical image checksum mismatch; restore the stored source bytes before review.')
        return entry
    def register_image(self, frame_id, image_bytes, source_path, extension='.jpg', *, kind='source_concat', image_layout=None):
        self.frame(frame_id)
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError('Image kind must be non-empty text')
        if image_layout is None:
            image_layout = 'original_concat_unmodified' if kind == 'source_concat' else 'source_image_unmodified'
        if not isinstance(image_layout, str) or not image_layout.strip():
            raise ValueError('Image layout must be non-empty text')
        sha = hashlib.sha256(image_bytes).hexdigest()
        path = Path('objects') / (sha + extension)
        (self.root / 'objects').mkdir(exist_ok=True)
        if not (self.root/path).exists():
            (self.root/path).write_bytes(image_bytes)
        entry = {'id':frame_id,'frame_id':frame_id,'path':str(path),'sha256':sha,'size':len(image_bytes),'source_path':source_path,'kind':kind,'image_layout':image_layout,'retrieved_at':now()}
        with self.transaction():
            self.db.execute('INSERT OR REPLACE INTO records VALUES(?,?,?,?)', ('local_images',frame_id,frame_id,dumps(entry)))
        return entry
    def register_text(self, asset, text, raw_bytes_sha256):
        entry = dict(asset, text=text, source_path=asset.get('nas_path'), sha256=raw_bytes_sha256, retrieved_at=now())
        with self.transaction():
            self.db.execute('INSERT OR REPLACE INTO records VALUES(?,?,?,?)', ('descriptions',asset['id'],asset['frame_id'],dumps(entry)))
    def frame_detail(self, frame_id):
        from .evaluation import analyze_output
        from .rating_references import resolve_rating_references
        from .recorded_inputs import recorded_input_preview
        from .output_format import output_format_evidence
        from .adapted_inputs import archived_output_fields
        frame = self.frame(frame_id)
        contract = self.latest('input', frame_id)
        expected_fields = archived_output_fields(contract)
        ratings = self.get_records('ratings', frame_id)
        descriptions = [dict(record, output_view=analyze_output(record.get('text'), expected_fields))
                        for record in self.get_records('descriptions', frame_id)]
        input_links, evidence_links = [], []
        if self.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='wf_records'").fetchone():
            for row in self.db.execute("SELECT data FROM wf_records WHERE kind='evidence_link'"):
                link = json.loads(row[0])
                evidence_links.append(link)
                if link.get('frame_id') == frame_id and link.get('input_verified') is True and link.get('human_confirmed') is True:
                    input_links.append({key:link.get(key) for key in ('frame_id','input_sha256','prediction_hashes','input_verified','human_confirmed')})
        predictions = [json.loads(row[0]) for row in self.db.execute('SELECT data FROM predictions WHERE frame_id=?',(frame_id,))]
        for prediction in predictions:
            prediction['expected_output_fields'] = list(self.prediction_output_fields(prediction))
        references = resolve_rating_references(frame_id, ratings, descriptions, evidence_links,
            predictions, self.list_runs(), expected_fields)
        return {'frame':frame,'assets':self.get_records('assets',frame_id),'ratings':ratings,
                'ground_truth':self.latest('gt',frame_id),'history':self.versions('gt',frame_id),'descriptions':descriptions,
                'input_evidence_links':input_links, **references,
                'input_contract':contract,'output_format':output_format_evidence(contract),'image':self.image(frame_id),'raw_records':self.get_records('raw_records',frame_id),
                'recorded_input':recorded_input_preview(self,frame_id)}
    def frame_list(self, q='', status='', tag=''):
        rating_counts = dict(self.db.execute("SELECT frame_id,count(*) FROM records WHERE kind='ratings' GROUP BY frame_id"))
        asset_counts = dict(self.db.execute("SELECT frame_id,count(*) FROM records WHERE kind='assets' GROUP BY frame_id"))
        image_ids = {r[0] for r in self.db.execute("SELECT frame_id FROM records WHERE kind='local_images'")}
        result=[]
        references = self.latest_versions('gt')
        inputs = self.latest_versions('input')
        for f in self.get_records('frames'):
            if tag and tag != 'all' and tag not in f['tags']:
                continue
            gt = references.get(f['id'])
            state = gt['status'] if gt else 'not_started'
            contract=inputs.get(f['id'])
            if gt and gt['status'] in ('reviewed','approved') and (not contract or gt.get('input_hash')!=contract['input_hash']):
                state='needs_review'
            if q.lower() not in (f['id']+' '+f.get('recording_id','')).lower():
                continue
            if status and status != 'all' and state != status:
                continue
            result.append(dict(f,gt_status=state,ratings_count=rating_counts.get(f['id'],0),assets_count=asset_counts.get(f['id'],0),has_image=f['id'] in image_ids))
        return result
    def collection_tags(self):
        counts = Counter(tag for frame in self.get_records('frames') for tag in frame['tags'])
        return [{'tag': tag, 'frames': counts[tag]} for tag in sorted(counts)]
    def summary(self):
        counts = dict(self.db.execute('SELECT kind,count(*) FROM records GROUP BY kind'))
        statuses = [f['gt_status'] for f in self.frame_list()]
        row = self.db.execute("SELECT data FROM meta WHERE key='catalog_summary'").fetchone()
        return dict(frames=counts.get('frames',0),assets=counts.get('assets',0),ratings=counts.get('ratings',0),collection_tags=self.collection_tags(),
            local_images=counts.get('local_images',0),descriptions=counts.get('descriptions',0),
            gt_drafts=statuses.count('draft'),gt_reviewed=statuses.count('reviewed'),gt_approved=statuses.count('approved'),gt_needs_review=statuses.count('needs_review'),
            runs=self.db.execute('SELECT count(*) FROM runs').fetchone()[0],judgments=self.db.execute('SELECT count(*) FROM judgments').fetchone()[0],
            input_contracts=self.db.execute("SELECT count(DISTINCT frame_id) FROM versions WHERE kind='input'").fetchone()[0],
            provider={'enabled':False,'label':'Codex packets and explicit CLI API execution','mode':'cli_only'},catalog_summary=json.loads(row[0]) if row else {})
    def _append_version(self, kind, frame_id, payload, expected):
        # Acquire the write lock before checking version to avoid lost updates across requests.
        with self.transaction():
            previous = self.latest(kind,frame_id)
            revision = previous['revision'] if previous else 0
            if expected != revision:
                raise Conflict('This record changed. Reload before saving.')
            entry = dict(payload, frame_id=frame_id,revision=revision+1,created_at=now())
            entry['sha256']=digest(entry)
            self.db.execute('INSERT INTO versions VALUES(?,?,?,?)',(kind,frame_id,revision+1,dumps(entry)))
            return entry
    def save_input(self, frame_id, payload):
        self.frame(frame_id)
        image = self.image(frame_id)
        if not image:
            raise ValueError('Download the canonical image before freezing an input.')
        if 'adaptation_source' in payload:
            from .adapted_inputs import adapted_input_contract
            from .recorded_inputs import input_semantics
            reviewer = payload.get('reviewer', '').strip()
            if not reviewer:
                raise ValueError('An importer or reviewer name is required for the audit trail.')
            with self.transaction():
                request = adapted_input_contract(self, frame_id, payload, image)
                return self._append_version('input', frame_id,
                    dict(request, input_hash=digest(input_semantics(request)), reviewer=reviewer, notes=payload.get('notes', '')),
                    payload.get('expected_revision', 0))
        if 'reconstruction_manifest_id' in payload:
            from .reconstructed_inputs import reconstructed_input_contract
            from .recorded_inputs import input_semantics
            reviewer = payload.get('reviewer', '').strip()
            if not reviewer:
                raise ValueError('An importer or reviewer name is required for the audit trail.')
            request = reconstructed_input_contract(self, frame_id, payload, image)
            return self._append_version('input', frame_id,
                dict(request, input_hash=digest(input_semantics(request)), reviewer=reviewer, notes=payload.get('notes', '')),
                payload.get('expected_revision', 0))
        if 'source_record_id' in payload:
            from .recorded_inputs import recorded_input_contract, input_semantics
            reviewer = payload.get('reviewer', '').strip()
            if not reviewer:
                raise ValueError('An importer or reviewer name is required for the audit trail.')
            request = recorded_input_contract(self, frame_id, payload, image, FIELDS)
            return self._append_version('input', frame_id,
                dict(request, input_hash=digest(input_semantics(request)), reviewer=reviewer, notes=payload.get('notes', '')),
                payload.get('expected_revision', 0))
        prompt = payload.get('prompt','').strip()
        reviewer = payload.get('reviewer','').strip()
        if not prompt or not reviewer:
            raise ValueError('Prompt and reviewer are required.')
        velocity = payload.get('velocity')
        if velocity is not None and (not isinstance(velocity,dict) or not velocity.get('units') or not velocity.get('source') or 'value' not in velocity):
            raise ValueError('Velocity needs value, units and source; leave null until verified.')
        track = 'image_velocity' if velocity is not None else 'image_only'
        system_prompt=payload.get('system_prompt','')
        image_detail=payload.get('image_detail','high')
        if not isinstance(system_prompt,str) or image_detail not in ('high','low','auto'):
            raise ValueError('Invalid system prompt or image detail')
        request = {'protocol_version':'mousebench.input.v2','image_sha256':image['sha256'],'prompt':prompt,'system_prompt':system_prompt,'image_detail':image_detail,'velocity':velocity,'track':track,
                   'output_fields':list(FIELDS),'image_layout':image.get('image_layout') or ('original_concat_unmodified' if image.get('kind')=='source_concat' else 'source_image_unmodified')}
        # Hash excludes reviewer/time: same exact request has a stable identity.
        return self._append_version('input',frame_id,dict(request,input_hash=digest(request),reviewer=reviewer,notes=payload.get('notes','')),payload.get('expected_revision',0))
    def save_gt(self, frame_id, payload):
        # Keep the displayed version/input checks, issue gate, and saved approval
        # under one write lock so concurrent changes cannot alter what was approved.
        with self.transaction():
            return self._save_gt(frame_id, payload)

    def _save_gt(self, frame_id, payload):
        self.frame(frame_id)
        status = payload.get('status','draft')
        if status not in ('draft','reviewed','approved'):
            raise ValueError('Invalid GT status')
        actor=payload.get('actor_type','human')
        if actor not in ('human','assistant') or (actor=='assistant' and status!='draft'):
            raise ValueError('Assistant submissions may create drafts only')
        if status=='approved' and payload.get('human_confirmed') is not True:
            raise ValueError('Approval requires explicit human confirmation after review')
        approval_action = payload.get('approval_action')
        if approval_action is not None and (approval_action != 'approve_reference' or status != 'approved'
                                            or payload.get('actor_type') != 'human'):
            raise ValueError('Approve reference requires an explicit human approval action')
        reviewer = payload.get('reviewer', '')
        if reviewer is None:
            reviewer = ''
        if not isinstance(reviewer, str):
            raise ValueError('Reviewer must be text when supplied')
        reviewer = reviewer.strip()
        content = payload.get('content')
        if not isinstance(content,dict):
            raise ValueError('GT content must be an object')
        facts = content.get('facts',[])
        ids=set()
        if not isinstance(facts,list) or not isinstance(content.get('uncertainties',[]),list):
            raise ValueError('Facts and uncertainties must be lists')
        for fact in facts:
            if (not isinstance(fact,dict) or not isinstance(fact.get('id'),str) or not fact['id'].strip()
                or fact['id']!=fact['id'].strip() or not isinstance(fact.get('text'),str) or not fact['text'].strip() or fact['id'] in ids):
                raise ValueError('Each fact requires a unique ID and text')
            ids.add(fact['id'])
            if fact.get('field') not in FIELDS or fact.get('evidence') not in ('image','velocity','unknown') or not isinstance(fact.get('required'),bool):
                raise ValueError('Facts need a valid field, evidence scope and required flag')
            if fact.get('required') and fact.get('evidence') == 'unknown':
                raise ValueError('Unknown facts cannot be required')
            if 'uncertain' in fact and not isinstance(fact['uncertain'],bool):
                raise ValueError('Fact uncertainty must be boolean')
            if fact.get('required') and fact.get('uncertain'):
                raise ValueError('Uncertain facts cannot be required')
        contract=self.latest('input',frame_id)
        previous=self.latest('gt',frame_id)
        if approval_action and 'expected_revision' not in payload:
            raise Conflict('Identify the displayed reference revision before approving.')
        if payload.get('expected_revision',0) != (previous['revision'] if previous else 0):
            raise Conflict('This record changed. Reload before saving.')
        if approval_action and (not contract or payload.get('expected_input_hash') != contract['input_hash']):
            raise Conflict('The input changed or was not identified. Reload before approving.')
        if status in ('reviewed','approved'):
            if not contract or not facts or not any(f.get('required') for f in facts):
                raise ValueError('Review requires a frozen input and at least one required fact.')
            from .recorded_inputs import PROTOCOL, verify_input_images
            verify_input_images(self, contract)
            if approval_action and contract.get('protocol_version') != PROTOCOL:
                image = self.image(frame_id)
                if not image or image['sha256'] != contract.get('image_sha256'):
                    raise ValueError('Restore the frozen input image before approval.')
            if any(f.get('evidence')=='velocity' for f in facts) and contract['velocity'] is None:
                raise ValueError('Velocity facts require a verified velocity input.')
            sections=content.get('sections',{})
            if not all(isinstance(sections.get(f),str) and sections[f].strip() for f in FIELDS):
                raise ValueError('Complete all six sections; use an explicit uncertainty statement when needed.')
        if status=='approved':
            if not approval_action:
                if not previous or previous['status']!='reviewed' or digest(previous['content'])!=digest(content):
                    raise ValueError('Save and review the exact content before approval.')
                if previous.get('input_hash') != contract['input_hash']:
                    raise ValueError('Input changed since review; review the reference against the new input first.')
            blocked_hashes = {digest(content)}
            if approval_action and previous and previous.get('input_hash') == contract['input_hash']:
                # Editing text in the approval request must not silently retire
                # an unresolved critical finding on the currently saved draft.
                blocked_hashes.add(digest(previous['content']))
            if any(self.has_critical_issue(frame_id,sha,contract['input_hash']) for sha in blocked_hashes):
                raise ValueError('Resolve critical issues on this reference before approval')
        provenance=payload.get('provenance')
        if provenance is not None and not isinstance(provenance,dict):
            raise ValueError('Provenance must be an object')
        provenance=dict(provenance or (previous or {}).get('provenance',{}))
        provenance.update(origin='assistant_draft' if actor=='assistant' else 'human_review',
                          assistant_generated=True if actor=='assistant' else provenance.get('assistant_generated',False),
                          actor_type=actor,human_confirmed=payload.get('human_confirmed') is True)
        provenance.pop('approval_action', None)
        if approval_action:
            provenance['approval_action'] = approval_action
        entry=dict(status=status,reviewer=reviewer,notes=payload.get('notes',''),content=content,
                   input_hash=contract['input_hash'] if contract else None,content_sha256=digest(content),
                   provenance=provenance)
        return self._append_version('gt',frame_id,entry,payload.get('expected_revision',0))
    def list_runs(self):
        return [json.loads(r[0]) for r in self.db.execute('SELECT data FROM runs ORDER BY id')]
    def prediction_output_fields(self, record):
        """Use a prediction's original frozen input, never impose a later format."""
        from .output_format import expected_output_fields
        from .adapted_inputs import archived_output_fields
        input_hash = record.get('input_sha256') or record.get('input_hash')
        current = self.latest('input', record['frame_id'])
        if input_hash:
            source = next((item for item in self.versions('input', record['frame_id'])
                           if item.get('input_hash') == input_hash), None)
            if source:
                return expected_output_fields(source)
        if record.get('track', 'historical') == 'historical':
            return archived_output_fields(current)
        return expected_output_fields(current)

    def predictions(self, run_id, *, field_cache=None):
        from .evaluation import check_output_structure
        records = [json.loads(r[0]) for r in self.db.execute('SELECT data FROM predictions WHERE run_id=? ORDER BY frame_id',(run_id,))]
        # Refresh display diagnostics against the actual request without rewriting
        # immutable output/import provenance or established contract hashes.
        if field_cache is None:
            field_cache = {}
        result = []
        for record in records:
            key = (record['frame_id'], record.get('input_sha256') or record.get('input_hash'), record.get('track', 'historical'))
            if key not in field_cache:
                field_cache[key] = self.prediction_output_fields(record)
            result.append(dict(record, mechanical_checks=check_output_structure(record.get('output',''), field_cache[key])))
        return result

    def prediction_headers(self, run_id):
        """Coverage metadata only; do not decode provider logs or model answers."""
        rows = self.db.execute("""SELECT frame_id,json_extract(data,'$.status'),
            CASE WHEN json_type(data,'$.input_sha256') IS NULL THEN json_extract(data,'$.input_hash')
                 ELSE json_extract(data,'$.input_sha256') END FROM predictions WHERE run_id=? ORDER BY frame_id""", (run_id,))
        return [{'frame_id': row[0], 'status': row[1], 'input_sha256': row[2]} for row in rows]
    def import_run(self, payload):
        from .evaluation import validate_prediction, check_output_structure
        name=payload.get('name','').strip(); model=payload.get('model','').strip(); track=payload.get('track','historical')
        if not name or not model or track not in ('historical','controlled'):
            raise ValueError('Name, model and a valid track are required')
        records=payload.get('records',[])
        if not records or not isinstance(records,list):
            raise ValueError('A nonempty predictions list is required')
        known={f['id'] for f in self.get_records('frames')}
        run_id='run_'+uuid.uuid4().hex[:12]
        clean=[]; seen=set(); protocols=set()
        for raw in records:
            record=validate_prediction(dict(raw,run_id=run_id),known,track)
            fid=record['frame_id']
            if fid in seen:
                raise ValueError('Duplicate frame in one run; preserve separate attempts in separate runs')
            seen.add(fid)
            record['output']=record.get('output_text',record.get('output','')) or ''
            record['prediction_sha256']=hashlib.sha256(record['output'].encode()).hexdigest()
            record['input_hash']=record.get('input_hash') or record.get('input_sha256')
            record['input_sha256']=record['input_hash']
            record['mechanical_checks']=check_output_structure(record['output'],self.prediction_output_fields(record))
            record['recording_id']=self.frame(fid).get('recording_id')
            if track=='controlled':
                contract=self.latest('input',fid)
                if not contract or record['input_hash']!=contract['input_hash']:
                    raise ValueError('Controlled prediction does not match the frozen input for '+fid)
                meta=record.get('model_metadata',{})
                if not meta.get('model_id') or not isinstance(meta.get('generation_settings'),dict) or not record.get('request_log'):
                    raise ValueError('Controlled runs need model_id, generation_settings and an actual request_log per prediction')
                log=record['request_log']
                if not isinstance(log,dict) or log.get('kind')!='native_execution' or log.get('input_sha256')!=contract['input_hash']:
                    raise ValueError('Controlled request_log must be native_execution and bind to the same input_sha256')
                protocols.add(contract['track'])
            clean.append(record)
        if len(protocols)>1:
            raise ValueError('Separate image-only and image-plus-velocity runs')
        run={'id':run_id,'name':name,'model':model,'track':track,'created_at':now(),'predictions_count':len(clean),
             'coverage':len(clean)/len(known) if known else 0,'protocol_tracks':sorted(protocols),
             'eligibility':'request_contract_matched' if track=='controlled' else 'historical_unverified',
             'manifest_sha256':digest(clean),'model_identity_status':'declared_by_importer',
             'note':'Imported records are not evidence of native execution logs unless supplied.'}
        with self.transaction():
            self.db.execute('INSERT INTO runs VALUES(?,?)',(run_id,dumps(run)))
            self.db.executemany('INSERT INTO predictions VALUES(?,?,?)',[(run_id,r['frame_id'],dumps(r)) for r in clean])
        return run
    def has_critical_issue(self,frame_id,content_hash,input_hash):
        if not self.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='wf_records'").fetchone():
            return False
        for row in self.db.execute("SELECT data FROM wf_records WHERE kind='issue'"):
            issue=json.loads(row[0])
            if (issue.get('frame_id')==frame_id and issue.get('status')=='open' and issue.get('severity')=='critical'
                and issue.get('target','reference')=='reference'
                and issue.get('content_sha256')==content_hash and issue.get('input_sha256')==input_hash):
                return True
        return False

    def critical_issue_bindings(self):
        """Read the blocking reference bindings once for one aggregation pass.

        The returned set is local to its caller, never a cross-request cache.
        Include content and input hashes so stale issues cannot block new data.
        """
        if not self.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='wf_records'").fetchone():
            return set()
        bindings=set()
        for row in self.db.execute("SELECT data FROM wf_records WHERE kind='issue'"):
            issue=json.loads(row[0])
            if (issue.get('status')=='open' and issue.get('severity')=='critical'
                and issue.get('target','reference')=='reference'):
                bindings.add((issue.get('frame_id'),issue.get('content_sha256'),issue.get('input_sha256')))
        return bindings

    def import_judgments(self,payload):
        from .evaluation import score_judgment
        records=payload.get('records',[])
        if not isinstance(records,list) or not records:
            raise ValueError('A nonempty judgments list is required')
        staged=[]; batch_signatures={}
        for j in records:
            fid=j.get('frame_id'); run_id=j.get('run_id')
            row=self.db.execute('SELECT data FROM predictions WHERE run_id=? AND frame_id=?',(run_id,fid)).fetchone()
            if not row:
                raise ValueError('Judgment has no matching prediction')
            pred=json.loads(row[0]); gt=self.latest('gt',fid); contract=self.latest('input',fid)
            if pred['status']!='succeeded':
                raise ValueError('Failed, missing or cancelled predictions cannot receive a semantic judgment.')
            if not gt or gt['status']!='approved' or not contract or gt['input_hash']!=contract['input_hash']:
                raise ValueError('Judge import requires current approved GT bound to the current input')
            if self.has_critical_issue(fid,gt['content_sha256'],contract['input_hash']):
                raise ValueError('Resolve critical reference issues before importing judgments')
            for key,expected in [('ground_truth_sha256',gt['content_sha256']),('prediction_sha256',pred['prediction_sha256']),('input_sha256',contract['input_hash'])]:
                if j.get(key)!=expected:
                    raise ValueError('Judgment '+key+' does not match the stored version')
            run=json.loads(self.db.execute('SELECT data FROM runs WHERE id=?',(run_id,)).fetchone()[0])
            if run['track']=='controlled' and pred['input_hash']!=contract['input_hash']:
                raise ValueError('Prediction input is stale')
            core=dict(j); assessments=core.pop('criterion_assessments',None)
            rubric_rows=self.db.execute("SELECT data FROM wf_records WHERE kind='rubric'").fetchall() if self.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='wf_records'").fetchone() else []
            rubric_record=next((r for row in rubric_rows for r in [json.loads(row[0])] if r['sha256']==j['rubric_sha256']),None)
            rubric=rubric_record['content'] if rubric_record else None
            graded=(rubric or {}).get('scoring',{}).get('version')=='mousebench.graded-factual-f1.v2'
            if graded and rubric_record['status']!='frozen':
                raise ValueError('Freeze the graded rubric before importing judgments')
            if graded and j.get('schema_version')!='mousebench.judgment.v2':
                raise ValueError('The graded rubric requires a v2 judgment with correctness and coverage credits')
            if graded and rubric.get('analytic_criteria') and assessments is None:
                raise ValueError('The graded rubric requires every analytic criterion assessment')
            if assessments is not None:
                from .criteria import validate_assessments
                if not rubric or not rubric.get('analytic_criteria'):raise ValueError('Criterion assessments need a known rubric with analytic criteria')
                validate_assessments(assessments,rubric['analytic_criteria'])
            scored=score_judgment(dict(core,track=run['track'],recording_id=self.frame(fid).get('recording_id')),dict(gt['content'],status='approved'),rubric=rubric)
            entry=dict(j,scores=scored,input_track=contract['track'],track=run['track'],model=run['model'],run_name=run['name'],recording_id=self.frame(fid).get('recording_id'),
                       gt_revision=gt['revision'],created_at=now(),calibration_status='not_established')
            entry['id']=digest(j)
            batch_key=digest([run_id,fid,j.get('protocol_version'),j.get('rubric_sha256'),j.get('judge'),j.get('ground_truth_sha256')])
            if batch_key in batch_signatures and batch_signatures[batch_key]!=entry['id']:
                raise ValueError('Conflicting judgments inside the same import batch')
            batch_signatures[batch_key]=entry['id']
            # One judgment per frame/config/run; duplicates with different verdicts need a new rubric/judge version.
            for old in self.db.execute('SELECT data FROM judgments WHERE run_id=? AND frame_id=?',(run_id,fid)):
                previous=json.loads(old[0])
                signature=lambda x:(x.get('protocol_version'),x.get('rubric_sha256'),dumps(x.get('judge')),x.get('ground_truth_sha256'))
                if signature(previous)==signature(entry) and previous['id']!=entry['id']:
                    raise ValueError('Conflicting judgment for the same configuration; use a distinct version')
            staged.append(entry)
        with self.transaction():
            for e in staged:
                self.db.execute('INSERT OR IGNORE INTO judgments VALUES(?,?,?,?)',(e['id'],e['run_id'],e['frame_id'],dumps(e)))
        return {'imported':len(staged),'note':'Scores are provisional until judge calibration is established.'}
    def scorecard_snapshot(self):
        """One request's source rows, shared by aggregation and index enrichment."""
        return {'entries': [json.loads(row[0]) for row in self.db.execute('SELECT data FROM judgments')],
                'references': self.latest_versions('gt'), 'inputs': self.latest_versions('input'),
                'frames': {f['id']: f for f in self.get_records('frames')},
                'runs': {r['id']: r for r in self.list_runs()},
                'critical_bindings': self.critical_issue_bindings()}

    def scorecards(self, *, snapshot=None):
        from .evaluation import aggregate_scorecards, prepare_judgment_scorer
        snapshot = snapshot if snapshot is not None else self.scorecard_snapshot()
        entries=snapshot['entries']
        rubric_rows=self.db.execute("SELECT data FROM wf_records WHERE kind='rubric'").fetchall() if self.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='wf_records'").fetchone() else []
        rubrics={r['sha256']:r['content'] for row in rubric_rows for r in [json.loads(row[0])]}
        critical_bindings=snapshot['critical_bindings']
        scorers={sha:prepare_judgment_scorer(content) for sha,content in rubrics.items()}
        fallback_scorer=prepare_judgment_scorer(None)
        eligible=[]; stale=0; blocked=0
        for j in entries:
            gt=snapshot['references'].get(j['frame_id']); contract=snapshot['inputs'].get(j['frame_id'])
            if not gt or gt['status']!='approved' or gt['content_sha256']!=j['ground_truth_sha256'] or not contract or contract['input_hash']!=j['input_sha256'] or gt.get('input_hash')!=contract['input_hash']:
                stale+=1;continue
            if (j['frame_id'],gt['content_sha256'],contract['input_hash']) in critical_bindings:
                blocked+=1;continue
            # Refresh additive diagnostics from immutable raw verdicts; no stored
            # judgment or historical scientific score is rewritten.
            allowed=('schema_version','frame_id','run_id','protocol_version','judge','rubric_sha256',
                     'ground_truth_sha256','prediction_sha256','input_sha256','claims','supported_fact_ids','fact_coverage',
                     'structure_score','instruction_score','notes','track','recording_id')
            raw={k:j[k] for k in allowed if k in j}
            refreshed=scorers.get(j['rubric_sha256'],fallback_scorer)(raw,dict(gt['content'],status='approved'))
            j=dict(j,scores=refreshed)
            eligible.append(j)
        def group_key(j):
            return digest([j['protocol_version'],j['rubric_sha256'],j['judge'],j['track'],j.get('input_track'),j['scores']['scoring_version']])
        # Number groups by their first recorded judgment, including groups that
        # are temporarily ineligible. A reference edit must not renumber history.
        recorded_groups={}
        for j in entries:
            key=group_key(j)
            dates=recorded_groups.setdefault(key,[])
            stamp=recorded_timestamp(j.get('created_at'))
            if stamp is not None: dates.append(stamp)
        chronology=sorted(recorded_groups,key=lambda key: min(recorded_groups[key]) if recorded_groups[key] else datetime.min.replace(tzinfo=timezone.utc))
        comparison_numbers={key:index+1 for index,key in enumerate(chronology)}
        groups={}
        for j in eligible:
            key=group_key(j)
            groups.setdefault(key,[]).append(j)
        result={'models':[],'groups':[],'provisional':True,'calibration_status':'not_established',
                'stale_judgments_excluded':stale,'critical_issue_judgments_excluded':blocked,'total_frames':len(snapshot['frames'])}
        run_lookup=snapshot['runs']
        prediction_stats={}; field_cache={}; judge_settings={}
        if self.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='wf_records'").fetchone():
            for row in self.db.execute("""SELECT json_extract(data,'$.execution_config_sha256'),
                json_extract(data,'$.model'),json_extract(data,'$.executor'),
                json_extract(data,'$.provider_config.settings') FROM wf_records
                WHERE kind='batch' AND json_extract(data,'$.task')='judge'"""):
                judge_settings.setdefault((row[0],row[1],row[2]),[]).append(json.loads(row[3]) if row[3] else None)
        for key in sorted(groups,key=lambda key:comparison_numbers[key],reverse=True):
            rows=groups[key]
            aggregate=aggregate_scorecards([r['scores'] for r in rows])
            dates=[stamp for row in rows for stamp in [recorded_timestamp(row.get('created_at'))] if stamp is not None]
            judge=rows[0]['judge']; config_sha=judge.get('version','').partition('|config:')[2]
            effort=reasoning_effort_metadata(judge_settings.get((config_sha,judge.get('model'),judge.get('provider')),[]))
            aggregate.update(group_id=key,judge=judge,rubric_sha256=rows[0]['rubric_sha256'],input_track=rows[0].get('input_track'),
                comparison_number=comparison_numbers[key],created_at=min(dates).isoformat() if dates else None,
                updated_at=max(dates).isoformat() if dates else None,
                judge_reasoning_effort=effort['reasoning_effort'],judge_reasoning_effort_status=effort['reasoning_effort_status'])
            result['groups'].append(aggregate)
            for run_id,card in aggregate['runs'].items():
                run=run_lookup[run_id]
                if run_id not in prediction_stats:
                    predictions=self.predictions(run_id, field_cache=field_cache)
                    prediction_stats[run_id]={
                        'attempted_count':sum(p['status']!='missing' for p in predictions),
                        'failed_count':sum(p['status'] in ('failed','timeout','cancelled') for p in predictions),
                        'mechanical_section_pass_rate':sum(p.get('mechanical_checks',{}).get('passed',False) for p in predictions)/run['predictions_count'],
                        **reasoning_effort_metadata((p.get('model_metadata') or {}).get('generation_settings') for p in predictions)}
                result['models'].append(dict(card,run_id=run_id,model=run['model'],name=run['name'],track=run['track'],
                    group_id=key,scoring_version=aggregate['scoring_version'],input_track=rows[0].get('input_track'),benchmark_coverage=card['evaluable_count']/result['total_frames'],
                    **prediction_stats[run_id]))
        return result
    def export(self,kind):
        if kind=='catalog':
            return [dict(r,record_kind=k) for k in ('frames','assets','ratings','local_images','descriptions','raw_records') for r in self.get_records(k)]
        if kind=='ground-truth':
            return [dict(json.loads(r['data']),record_kind=r['kind']) for r in self.db.execute('SELECT kind,data FROM versions ORDER BY kind,frame_id,revision')]
        if kind=='runs':
            return [dict(r,record_kind='run') for r in self.list_runs()]+[dict(json.loads(r[0]),record_kind='prediction') for r in self.db.execute('SELECT data FROM predictions')]
        if kind=='judgments':
            return [json.loads(r[0]) for r in self.db.execute('SELECT data FROM judgments')]
        raise KeyError('Unknown export')
