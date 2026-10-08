#!/usr/bin/env python3
"""Run: python3 benchmark/app.py [--port 8765] [--data benchmark/data]."""
import argparse
import gzip
import hashlib
import json
import mimetypes
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote
from mousebench.store import Store, Conflict, dumps
from mousebench.workflow import Workflow
from mousebench.read_cache import ReadResponseCache

ROOT=Path(__file__).resolve().parent

class Handler(BaseHTTPRequestHandler):
    server_version='MouseBench/0.3'
    def log_message(self,fmt,*args):
        if self.path.startswith('/api/image/'):
            return
        super().log_message(fmt,*args)
    def respond(self,status,data,content_type='application/json; charset=utf-8',headers=None):
        raw=data if isinstance(data,bytes) else dumps(data).encode()
        compress=False
        if content_type.startswith('application/json') and len(raw)>=2048:
            for encoding in self.headers.get('Accept-Encoding','').split(','):
                parts=[part.strip() for part in encoding.split(';')]
                if parts[0].lower()!='gzip': continue
                try: quality=float(next((part[2:] for part in parts[1:] if part.startswith('q=')),'1'))
                except ValueError: quality=0
                compress=quality>0
            if compress: raw=gzip.compress(raw,compresslevel=1,mtime=0)
        self.send_response(status)
        self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(raw)))
        self.send_header('Cache-Control','no-store')
        if content_type.startswith('application/json'): self.send_header('Vary','Accept-Encoding')
        if compress: self.send_header('Content-Encoding','gzip')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Security-Policy',"default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        for key,value in (headers or {}).items(): self.send_header(key,value)
        self.end_headers(); self.wfile.write(raw)
    def permitted_host(self):
        host=self.headers.get('Host','')
        return host in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}')
    def read_response(self, compute):
        cache=getattr(self.server,'read_cache',None)
        if cache is None: return self.respond(200,compute())
        return self.respond(200,cache.get(self.path,lambda:dumps(compute()).encode()))
    def do_GET(self):
        if not self.permitted_host():
            return self.respond(403,{'error':'Use the local address displayed by the server.'})
        store=Store(self.server.data_dir)
        try:
            u=urlparse(self.path); path=unquote(u.path); query=parse_qs(u.query)
            if path=='/api/session': return self.respond(200,{'token':self.server.csrf_token})
            if path=='/api/summary': return self.read_response(store.summary)
            if path=='/api/overview':
                from mousebench.overview import build_overview
                return self.read_response(lambda:build_overview(store,Workflow(store)))
            if path=='/api/frames': return self.respond(200,store.frame_list(query.get('q',[''])[0],query.get('status',[''])[0],query.get('tag',[''])[0]))
            if path.startswith('/api/frames/'): return self.respond(200,store.frame_detail(path.split('/')[3]))
            if path.startswith('/api/assets/'):
                asset_id=path.split('/')[3]
                row=store.db.execute("SELECT data FROM records WHERE kind='raw_records' AND id=?",(asset_id,)).fetchone()
                if not row:
                    row=store.db.execute("SELECT data FROM records WHERE kind='assets' AND id=?",(asset_id,)).fetchone()
                if not row: raise KeyError('This asset is indexed but not materialized locally')
                record=json.loads(row[0])
                if not record.get('path'): raise KeyError('This asset is indexed but not materialized locally')
                asset_path=(store.root/record['path']).resolve()
                if not asset_path.is_relative_to(store.root) or not asset_path.is_file(): raise KeyError('Asset missing')
                mime={'.jpg':'image/jpeg','.jpeg':'image/jpeg','.png':'image/png',
                      '.txt':'text/plain; charset=utf-8','.json':'application/json; charset=utf-8',
                      '.jsonl':'application/x-ndjson; charset=utf-8'}.get(asset_path.suffix.lower())
                if not mime: raise ValueError('This asset type cannot be previewed')
                raw=asset_path.read_bytes()
                if not record.get('sha256') or hashlib.sha256(raw).hexdigest()!=record['sha256']:
                    raise ValueError('Asset checksum mismatch; restore the stored source bytes before review.')
                return self.respond(200,raw,mime)
            if path=='/api/runs': return self.read_response(store.list_runs)
            if path.startswith('/api/runs/'): return self.respond(200,store.predictions(path.split('/')[3]))
            if path=='/api/scorecards': return self.read_response(Workflow(store).scorecards)
            if path=='/api/scoring-review':
                from mousebench.scoring_review import build_review
                return self.read_response(lambda:build_review(store,query.get('run_id',[''])[0],query.get('group_id',[''])[0]))
            if path=='/api/scoring-review/prediction':
                from mousebench.scoring_review import prediction_detail
                return self.respond(200,prediction_detail(store,query.get('run_id',[''])[0],query.get('frame_id',[''])[0]))
            if path.startswith('/api/scoring-review/prediction-image/'):
                from mousebench.scoring_review import prediction_image
                parts=path.split('/')
                if len(parts)!=7: raise KeyError('Unknown prediction image')
                raw,mime=prediction_image(store,parts[4],parts[5],parts[6])
                return self.respond(200,raw,mime)
            if path.startswith('/api/scoring-review/'):
                from mousebench.scoring_review import review_detail, review_image
                parts=path.split('/')
                if len(parts)==6 and parts[4]=='image':
                    raw,mime=review_image(store,parts[3],parts[5])
                    return self.respond(200,raw,mime)
                if len(parts)!=4: raise KeyError('Unknown score review endpoint')
                return self.respond(200,review_detail(store,parts[3]))
            if path=='/api/workflow': return self.read_response(Workflow(store).summary)
            if path=='/api/workflow/calibration-template':
                return self.respond(200,Workflow(store).calibration_template(query.get('run_id',[''])[0]),headers={'Content-Disposition':'attachment; filename="human-judge-calibration.json"'})
            if path.startswith('/api/workflow/batches/'):
                return self.respond(200,Workflow(store).batch(path.split('/')[4]))
            if path.startswith('/api/workflow/export/'):
                bid=path.split('/')[4]
                return self.respond(200,Workflow(store).export_jobs(bid),headers={'Content-Disposition':f'attachment; filename="{bid}-packets.json"'})
            if path.startswith('/api/workflow/releases/'):
                return self.respond(200,Workflow(store).record('release',path.split('/')[4]))
            if path=='/api/workflow/issues': return self.respond(200,Workflow(store).issues())
            if path=='/api/providers':
                return self.respond(200,{'enabled':False,'mode':'explicit_cli_execution','providers':[
                    {'id':'codex','label':'Codex-assisted work packets','enabled':True},
                    {'id':'openai','label':'OpenAI Responses / Chat Completions','enabled':False},
                    {'id':'azure','label':'Azure OpenAI v1 Responses','enabled':False}],
                    'message':'Codex works through local packets. API connectors run only through the explicit CLI worker with credentials; this page never launches model calls.'})
            if path.startswith('/api/export/'):
                kind=path.split('/')[3]
                payload=''.join(dumps(r)+'\n' for r in store.export(kind)).encode()
                return self.respond(200,payload,'application/x-ndjson',{'Content-Disposition':f'attachment; filename="mousebench-{kind}.jsonl"'})
            if path.startswith('/api/image/'):
                entry=store.image(path.split('/')[3])
                if not entry: raise KeyError('Canonical image is not downloaded yet')
                return self.respond(200,(store.root/entry['path']).read_bytes(),'image/jpeg')
            if path.startswith('/api/input-image/'):
                parts=path.split('/')
                if len(parts)!=5: raise KeyError('Unknown input image')
                contract=store.latest('input',parts[3]) or {}
                entry=next((img for img in contract.get('input_images',[]) if img['attachment_id']==parts[4]),None)
                if not entry: raise KeyError('Recorded input image not imported')
                source=(store.root/entry['path']).resolve()
                if not source.is_relative_to(store.root) or not source.is_file(): raise KeyError('Input image missing')
                raw=source.read_bytes()
                if hashlib.sha256(raw).hexdigest()!=entry['sha256']: raise ValueError('Input image checksum mismatch')
                return self.respond(200,raw,entry['media_type'])
            rel='index.html' if path=='/' else path.lstrip('/')
            static=(ROOT/'web'/rel).resolve()
            if not static.is_relative_to(ROOT/'web') or not static.is_file(): raise KeyError('Not found')
            self.respond(200,static.read_bytes(),mimetypes.guess_type(str(static))[0] or 'application/octet-stream')
        except KeyError as exc: self.respond(404,{'error':str(exc)})
        except (ValueError,TypeError) as exc: self.respond(400,{'error':str(exc)})
        except Exception as exc:
            self.log_error('%s',str(exc)); self.respond(500,{'error':'Unexpected server error; see local terminal.'})
        finally: store.close()
    def do_POST(self):
        if not self.permitted_host(): return self.respond(403,{'error':'Invalid host'})
        origin=self.headers.get('Origin')
        allowed=(f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}')
        if origin and origin not in allowed: return self.respond(403,{'error':'Cross-origin writes are disabled'})
        if self.headers.get('X-Mousebench-Token')!=self.server.csrf_token:
            return self.respond(403,{'error':'Reload the workbench before saving.', 'code':'session_token_expired'})
        store=Store(self.server.data_dir)
        try:
            n=int(self.headers.get('Content-Length','0'))
            if n<1 or n>25_000_000: raise ValueError('Request size must be between 1 byte and 25 MB')
            body=json.loads(self.rfile.read(n)); path=urlparse(self.path).path
            if not isinstance(body,dict): raise ValueError('Expected a JSON object')
            if path=='/api/workflow/action': return self.respond(200,Workflow(store).dispatch(body))
            if path.startswith('/api/frames/'):
                parts=path.split('/'); fid=parts[3]
                if len(parts)!=5: raise KeyError('Unknown operation')
                if parts[4]=='ground-truth': return self.respond(200,store.save_gt(fid,body))
                if parts[4]=='input-contract': return self.respond(200,store.save_input(fid,body))
            if path=='/api/runs/import': return self.respond(200,store.import_run(body))
            if path=='/api/judgments/import': return self.respond(200,store.import_judgments(body))
            if path=='/api/jobs/preview':
                from mousebench.providers import build_job_preview
                ids=body.get('frame_ids',[])
                total=len(store.get_records('frames'))
                if not isinstance(ids,list) or not ids or len(ids)>total: raise ValueError(f'Select between 1 and {total} frames')
                entries=[store.frame_detail(fid) for fid in ids]
                for entry in entries:
                    entry['expert_evidence']=entry['ratings']
                    entry['candidates']=entry['descriptions']
                run_id=body.get('run_id'); run=None
                if run_id:
                    run=next((r for r in store.list_runs() if r['id']==run_id),None)
                    if not run: raise ValueError('Unknown run')
                    preds={p['frame_id']:p for p in store.predictions(run_id)}
                    for entry in entries: entry['prediction']=preds.get(entry['frame']['id'])
                return self.respond(200,build_job_preview(body.get('task'),entries,run=run))
            raise KeyError('Unknown operation')
        except Conflict as exc: self.respond(409,{'error':str(exc)})
        except KeyError as exc: self.respond(404,{'error':str(exc)})
        except (ValueError,TypeError) as exc: self.respond(400,{'error':str(exc)})
        except Exception as exc:
            self.log_error('%s',str(exc)); self.respond(500,{'error':'Unexpected server error; see local terminal.'})
        finally: store.close()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--data',type=Path,default=ROOT/'data')
    parser.add_argument('--import-catalog',type=Path)
    args=parser.parse_args()
    store=Store(args.data)
    catalog=args.import_catalog or (args.data/'catalog.json')
    if args.import_catalog or (not store.summary()['frames'] and catalog.exists()):
        store.import_catalog(json.loads(catalog.read_text()))
    store.close()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    server.data_dir=args.data.resolve(); server.csrf_token=secrets.token_urlsafe(32)
    server.read_cache=ReadResponseCache(server.data_dir)
    print(f'MouseGPT benchmark: http://127.0.0.1:{args.port}\nBrowser model calls: disabled; API execution requires the explicit CLI worker. Data: {server.data_dir}',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        server.server_close()
        server.read_cache.close()

if __name__=='__main__': main()
