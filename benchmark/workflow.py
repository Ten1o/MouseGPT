#!/usr/bin/env python3
"""MouseGPT workflow CLI. Network calls occur only with `run --execute`.

Codex can claim small packets, inspect each original image, save JSON results and
submit them. Review/approval is a distinct explicit human action in the UI.
"""
import argparse
import json
import sys
import time
import zipfile
from pathlib import Path
from mousebench.store import Store, dumps
from mousebench.workflow import Workflow

ROOT=Path(__file__).resolve().parent


def read_json(path):
    text=Path(path).read_text()
    try:return json.loads(text)
    except json.JSONDecodeError:return [json.loads(line) for line in text.splitlines() if line.strip()]


def emit(value, output=None):
    text=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if output:
        with Path(output).open('x',encoding='utf-8') as stream:stream.write(text)
        print(json.dumps({'written':str(Path(output).resolve())}))
    else:print(text,end='')


def run_batch(wf, batch_id, *, execute=False, limit=None, worker='api-worker', interval=1.0, transport=None):
    """Resumable one-at-a-time worker. Injection is for isolated offline tests."""
    from mousebench.api_providers import provider_status, execute_job, ProviderExecutionError
    b=wf.record('batch',batch_id)
    if limit is None: limit=max(1,len(b['frame_ids']))
    if b['executor']=='codex':raise ValueError('Use work/submit for Codex batches')
    status=provider_status(b['provider_config'])
    pending=wf.batch(batch_id,False)['batch']['counts']['pending']
    if not execute:
        return {'mode':'dry_run','pending':pending,'limit':limit,'provider_status':status,'model_calls':0}
    if not status['credential_configured'] and transport is None:raise ValueError('Set the configured credential environment variable before execution')
    if not isinstance(limit,int) or isinstance(limit,bool) or not 1<=limit<=max(395,len(wf.store.get_records('frames'))) or interval<0 or interval>60:raise ValueError('Invalid limit or interval')
    results=[];attempted=0
    for _ in range(limit):
        claim=wf.claim(batch_id,worker,1)
        if not claim['jobs']:break
        j=claim['jobs'][0];attempted+=1
        try:
            result=execute_job(j['packet'],b['provider_config'],wf.store.root,transport=transport)
        except ProviderExecutionError as exc:
            results.append(wf.fail({'job_id':j['id'],'lease_token':j['lease_token'],'code':exc.code,'retryable':exc.retryable,'uncertain':exc.uncertain}))
            # Stop on provider-wide rejection, rate limiting or uncertain delivery.
            # Resume leaves succeeded jobs untouched; failures need explicit review.
            if exc.uncertain or exc.code in ('rate_limited','request_rejected','missing_credential','invalid_credential','unsafe_endpoint'):break
        except (ValueError,KeyError,TypeError):
            results.append(wf.fail({'job_id':j['id'],'lease_token':j['lease_token'],'code':'local_request_validation','retryable':False}));break
        else:
            observed=result['request_log'].get('observed_model') or b['model']+' (snapshot not exposed by provider)'
            wf.record_receipt(j['id'],result,observed,worker)
            try:
                results.append(wf.complete({'job_id':j['id'],'lease_token':j['lease_token'],'worker':worker,'model_version':observed,'result':result}))
            except (ValueError,KeyError,TypeError) as exc:
                results.append(wf.fail({'job_id':j['id'],'lease_token':j['lease_token'],'code':'result_requires_review','result':result,'retryable':False}))
                # A provider response is already saved: never call again merely to repair ingestion.
                results[-1]['review_reason']=str(exc);break
        if interval and _+1<limit:time.sleep(interval)
    return {'mode':'executed','attempted':attempted,'results':results,'batch':wf.batch(batch_id,False)['batch'],
            'notice':'Interrupted leases and uncertain calls require explicit reconciliation. Saved receipts can be re-ingested without another model call.'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data')
    subs=p.add_subparsers(dest='command',required=True)
    subs.add_parser('status')
    action=subs.add_parser('action',help='Apply a JSON workflow action; never invokes a model')
    action.add_argument('file',type=Path)
    work=subs.add_parser('work',help='Claim image packets for an external Codex session')
    work.add_argument('batch_id');work.add_argument('--worker',required=True);work.add_argument('--limit',type=int,default=10);work.add_argument('--output',type=Path)
    show=subs.add_parser('export',help='Read-only packet export, no claim or model call');show.add_argument('batch_id');show.add_argument('--output',type=Path)
    submit=subs.add_parser('submit');submit.add_argument('file',type=Path);submit.add_argument('--batch-id')
    run=subs.add_parser('run');run.add_argument('batch_id');run.add_argument('--execute',action='store_true');run.add_argument('--limit',type=int)
    run.add_argument('--worker',default='api-worker');run.add_argument('--interval',type=float,default=1.0)
    ingest=subs.add_parser('ingest',help='Retry saved response ingestion without model calls');ingest.add_argument('batch_id')
    retry=subs.add_parser('retry');retry.add_argument('batch_id');retry.add_argument('--job-id',action='append',dest='job_ids')
    retry.add_argument('--reviewer',required=True);retry.add_argument('--acknowledge-uncertain',action='store_true')
    release=subs.add_parser('export-release');release.add_argument('release_id');release.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv);store=Store(args.data);wf=Workflow(store)
    try:
        if args.command=='status':emit(wf.summary())
        elif args.command=='action':emit(wf.dispatch(read_json(args.file)))
        elif args.command=='work':
            b=wf.record('batch',args.batch_id)
            if b['executor']!='codex':raise ValueError('API batches are claimed by run --execute')
            if args.output and args.output.exists():raise ValueError('Output already exists; choose a new path before claiming jobs')
            result=wf.claim(args.batch_id,args.worker,args.limit)
            for job in result['jobs']:job['packet']['image_base_directory']=str(store.root)
            emit(result,args.output)
        elif args.command=='export':emit(wf.export_jobs(args.batch_id),args.output)
        elif args.command=='submit':
            records=read_json(args.file)
            if isinstance(records,dict):records=records.get('records',[records])
            emit(wf.submit_results({'batch_id':args.batch_id,'records':records}))
        elif args.command=='run':
            result=run_batch(wf,args.batch_id,execute=args.execute,limit=args.limit,worker=args.worker,interval=args.interval)
            emit(result)
            if any(r['status']!='succeeded' for r in result.get('results',[])):return 1
        elif args.command=='ingest':
            result=wf.ingest_receipts(args.batch_id);emit(result)
            if result['errors']:return 1
        elif args.command=='retry':emit(wf.retry_jobs({'batch_id':args.batch_id,'reviewer':args.reviewer,'job_ids':args.job_ids,'acknowledge_uncertain':args.acknowledge_uncertain}))
        elif args.command=='export-release':
            record=wf.record('release',args.release_id)
            with zipfile.ZipFile(args.output,'x',zipfile.ZIP_DEFLATED) as z:
                z.writestr('release.json',dumps(record)+'\n')
                z.writestr('README.txt','MouseGPT frozen reference release. See release.json for full input/GT/rubric hashes and provenance. Provisional scientific validation; this is a manually selected historical frame collection.\n')
                seen=set()
                import hashlib
                for sample in record['samples']:
                    for image in [sample['image'], *sample['input'].get('input_images',[])]:
                        path=(store.root/image['path']).resolve()
                        if not path.is_relative_to(store.root) or hashlib.sha256(path.read_bytes()).hexdigest()!=image['sha256']:
                            raise ValueError('Release image checksum mismatch')
                        if image['path'] not in seen:z.write(path,image['path']);seen.add(image['path'])
            emit({'release_id':record['id'],'archive':str(args.output.resolve()),'frames':len(record['samples'])})
        return 0
    finally:store.close()

if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,KeyError) as exc:
        print(json.dumps({'error':str(exc)},ensure_ascii=False),file=sys.stderr);raise SystemExit(2)
