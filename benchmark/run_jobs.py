#!/usr/bin/env python3
"""Provider-neutral future execution harness; dry-run unless explicitly enabled.

An adapter is a locally installed module exposing execute(job) -> JSON object.
No adapter, API endpoint, credential or model identifier is supplied here.
"""
import argparse
import importlib
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from mousebench.store import dumps, digest


def read_jobs(path):
    text=Path(path).read_text()
    try:
        obj=json.loads(text)
        jobs=obj.get('jobs') if isinstance(obj,dict) else obj
    except json.JSONDecodeError:
        jobs=[json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(jobs,list) or not jobs: raise ValueError('Expected a nonempty job list or preview export')
    for j in jobs:
        if not isinstance(j,dict) or j.get('task') not in ('ground_truth','inference','judge') or not j.get('frame_id') or not j.get('payload_sha256'):
            raise ValueError('Expected versioned MouseBench job previews')
        if digest(j.get('payload_template')) != j['payload_sha256']:
            raise ValueError('Job payload checksum mismatch')
    return jobs


def validate_attachments(jobs, data_dir):
    root=Path(data_dir).resolve()
    for job in jobs:
        declared=job['payload_template'].get('attachments',[])
        manifest=job.get('attachment_manifest',[])
        if len(manifest)!=len(declared) or not declared:
            raise ValueError('Every job needs its canonical image attachment manifest')
        for spec,record in zip(declared,manifest):
            path=(root/record.get('path','')).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError('Attachment must be an existing file inside the benchmark data directory')
            sha=hashlib.sha256(path.read_bytes()).hexdigest()
            if record.get('sha256')!=sha or spec.get('sha256')!=sha or record.get('attachment_id')!=spec.get('attachment_id'):
                raise ValueError('Attachment checksum or identity mismatch')
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('jobs',type=Path); p.add_argument('--adapter',help='Verified local Python module exposing execute(job)')
    p.add_argument('--data',type=Path,default=Path(__file__).resolve().parent/'data',help='Allowed local attachment directory')
    p.add_argument('--execute',action='store_true',help='Explicitly permit calls through the supplied adapter')
    p.add_argument('--output',type=Path,help='New JSONL execution log; existing files are never overwritten')
    args=p.parse_args(); jobs=read_jobs(args.jobs)
    if not args.execute:
        print(dumps({'mode':'dry_run','jobs':len(jobs),'tasks':sorted({j['task'] for j in jobs}),'model_calls':0,'notice':'No adapter was loaded.'}));return
    if not args.adapter or not args.output: p.error('--execute requires --adapter and a new --output log')
    if args.adapter.startswith('mousebench.providers'): p.error('The bundled future provider is disabled')
    validate_attachments(jobs,args.data)
    # Exclusive log creation precedes adapter import, preventing accidental overwrite/retry.
    succeeded=failed=0
    with args.output.open('x',encoding='utf-8') as stream:
        adapter=importlib.import_module(args.adapter)
        if not callable(getattr(adapter,'execute',None)): raise ValueError('Adapter must expose execute(job)')
        for job in jobs:
            started=time.monotonic(); timestamp=datetime.now(timezone.utc).isoformat()
            base={'log_kind':'native_execution','job_payload_sha256':job['payload_sha256'],'frame_id':job['frame_id'],
                  'task':job['task'],'adapter':args.adapter,'started_at':timestamp,'input_sha256':job.get('input_contract_sha256')}
            # Persist attempt before a side effect. An interrupted attempt remains visible and is never retried automatically.
            stream.write(dumps(dict(base,event='attempt'))+'\n');stream.flush()
            try:
                execution_job=dict(job,execution_context={'data_directory':str(args.data.resolve())})
                result=adapter.execute(execution_job)
                if not isinstance(result,dict): raise ValueError('Adapter response must be a JSON object')
                record=dict(base,event='result',status='succeeded',elapsed_seconds=time.monotonic()-started,result=result)
                encoded=dumps(record)
                succeeded+=1
            except Exception as exc:
                failed+=1
                # Avoid writing arbitrary exception text that might include provider credentials.
                encoded=dumps(dict(base,event='result',status='failed',elapsed_seconds=time.monotonic()-started,error_type=type(exc).__name__))
            stream.write(encoded+'\n');stream.flush()
    print(dumps({'mode':'executed','attempted_jobs':len(jobs),'log':str(args.output),'automatic_retries':0,'succeeded':succeeded,'failed':failed}))
    if failed: raise SystemExit(1)

if __name__=='__main__':main()
