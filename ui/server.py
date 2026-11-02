"""Local-only MouseGPT UI server. Static assets are allowlisted; secrets stay server-side."""
import argparse
import base64
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import mimetypes
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit, parse_qs

ROOT=Path(__file__).resolve().parents[1]
UI=ROOT/'ui'
sys.path.insert(0,str(ROOT))
from mousegpt_config import load_config, read_env_file, ROLES, DEFAULT_MODELS
from local_paths import APP_HOME, SETTINGS_FILE, RUNTIME_DIR, backend_id
from environment_manager import EnvironmentManager,PROFILES
from environment_access import access as check_environment_access

RUNTIME=RUNTIME_DIR
SETTINGS=SETTINGS_FILE
ENVIRONMENTS=EnvironmentManager()
TOKEN=secrets.token_urlsafe(32)
BACKEND_ID=backend_id()
JOBS={}
LOCK=threading.Lock()
BUSY=threading.Lock()
CONFIG_LOCK=threading.Lock()
PORT=8878
AGENT=None


class LocalUIServer(ThreadingHTTPServer):
    """Only one UI process may own a port, including on Windows."""
    allow_reuse_address=False

    def server_bind(self):
        if hasattr(socket,'SO_EXCLUSIVEADDRUSE'):
            self.socket.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
        super().server_bind()

ENVROOT=Path.home()/'anaconda3/envs'
DEFAULTS={'analysis':str(ENVROOT/'swift-langchain/python.exe'),
          'pose':str(ENVROOT/'mmpose310/python.exe'),
          'preprocess':str(ENVROOT/'mmpose310/python.exe'),
          'projection':str(ENVROOT/'mgpt_windows/python.exe'),
          'agent':str(ENVROOT/'swift-langchain/python.exe')}


def read_settings():
    source=SETTINGS
    if not source.exists() and SETTINGS==SETTINGS_FILE and (UI/'.api.local.json').is_file():source=UI/'.api.local.json'
    return json.loads(source.read_text(encoding='utf-8')) if source.exists() else {'roles':{},'interpreters':DEFAULTS}


def environment(settings=None):
    settings=settings or read_settings()
    env=dict(os.environ)
    for name,value in settings.get('shared',{}).items():
        if value:env[f'MOUSEGPT_{name.upper()}']=str(value)
    if settings.get('shared',{}).get('model'):
        env.setdefault('MOUSEGPT_EMBEDDING_MODEL',DEFAULT_MODELS['EMBEDDING'])
    for role,fields in settings.get('roles',{}).items():
        for name,value in fields.items():
            if value:env[f'MOUSEGPT_{role}_{name.upper()}']=str(value)
    if 'shared' in settings:
        for role in ROLES:
            values={k:v for k,v in settings['shared'].items() if v}
            if role=='EMBEDDING':values['model']=DEFAULT_MODELS['EMBEDDING']
            custom={k:v for k,v in settings.get('roles',{}).get(role,{}).items() if v}
            changed=any(custom.get(k) and custom[k].rstrip('/')!=str(values.get(k,'')).rstrip('/') for k in ('provider','endpoint'))
            if changed and not custom.get('api_key'):
                raise ValueError(f'Set a dedicated API key for {role} when using a different service.')
            values.update(custom)
            # UI defaults are authoritative over older role entries in .env.local.
            config=load_config(role,overrides=values,env={},env_file=UI/'__no_env_file__',require_key=False,validate_capabilities=False)
            for name in ('provider','endpoint','model','api_version','api_key'):
                env[f'MOUSEGPT_{role}_{name.upper()}']=getattr(config,name) or ''
    env.update(PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8',GRADIO_ANALYTICS_ENABLED='False')
    return env


def public_settings():
    cfg=read_settings();roles={}
    env=environment(cfg)
    for role in sorted(ROLES):
        try:
            roles[role]=load_config(role,env=env,require_key=False).public_summary()
        except ValueError as exc:
            try:roles[role]={**load_config(role,env=env,require_key=False,validate_capabilities=False).public_summary(),'error':str(exc)}
            except ValueError:
                roles[role]={'role':role,'provider':'openai','model':DEFAULT_MODELS[role],'endpoint':'https://api.openai.com/v1','error':str(exc),**{k:v for k,v in cfg.get('roles',{}).get(role,{}).items() if k!='api_key'}}
    shared={k:v for k,v in cfg.get('shared',{}).items() if k!='api_key'}
    shared['api_key_configured']=bool(cfg.get('shared',{}).get('api_key'))
    return {'shared':shared,'overrides':{r:bool(cfg.get('roles',{}).get(r)) for r in ROLES},'roles':roles,'interpreters':{**DEFAULTS,**cfg.get('interpreters',{})},'settings_file':str(SETTINGS)}


def export_api_settings(include_api_keys=False):
    roles={};env=environment()
    for role in sorted(ROLES):
        try:
            config=load_config(role,env=env,require_key=False)
            values={'provider':config.provider,'endpoint':config.endpoint,'model':config.model,'api_version':config.api_version or ''}
            if include_api_keys and config.api_key:values['api_key']=config.api_key
        except ValueError:
            values={k:v for k,v in read_settings().get('roles',{}).get(role,{}).items() if k!='api_key' or include_api_keys}
        roles[role]=values
    saved=read_settings()
    if 'shared' in saved:
        roles={r:{k:v for k,v in fields.items() if k!='api_key' or include_api_keys} for r,fields in saved.get('roles',{}).items()}
        shared={k:v for k,v in saved['shared'].items() if k!='api_key' or include_api_keys}
        return {'format':'mousegpt-api-settings','schemaVersion':2,'shared':shared,'roles':roles,'contains_api_keys':bool(shared.get('api_key')) or any(bool(r.get('api_key')) for r in roles.values())}
    return {'format':'mousegpt-api-settings','schemaVersion':1,'contains_api_keys':any('api_key' in r for r in roles.values()),'roles':roles}


def save_settings(body):
    with CONFIG_LOCK:
        old=read_settings()
        if 'shared' in body:
            dest=old.setdefault('shared',{})
            for key,value in body['shared'].items():
                if key not in {'provider','endpoint','model','api_version','api_key'} or not isinstance(value,str) or '\n' in value or '\r' in value:raise ValueError('Invalid default API setting.')
                if key=='api_key' and not value:continue
                dest[key]=value.strip()
        for role in body.get('inherit',[]):
            if role not in ROLES:raise ValueError('Unknown API role.')
            old.setdefault('roles',{}).pop(role,None)
        for role,fields in body.get('roles',{}).items():
            if role not in ROLES:raise ValueError('Unknown API role.')
            dest=old.setdefault('roles',{}).setdefault(role,{})
            for key,value in fields.items():
                if key not in {'provider','endpoint','model','api_version','api_key'}:raise ValueError('Unknown API field.')
                if key=='api_key' and not value:continue
                if not isinstance(value,str) or '\n' in value or '\r' in value:raise ValueError('Invalid API setting.')
                dest[key]=value.strip()
        for key,value in body.get('interpreters',{}).items():
            if key not in DEFAULTS or not Path(value).is_file():raise ValueError(f'Python executable not found: {key}')
            old.setdefault('interpreters',{})[key]=str(Path(value).expanduser().absolute())
        env=environment(old)
        for role in (ROLES if 'shared' in body else body.get('roles',{})):load_config(role,env=env,require_key=False,validate_capabilities=False)
        SETTINGS.parent.mkdir(parents=True,exist_ok=True)
        temp=SETTINGS.with_suffix('.tmp')
        temp.write_text(json.dumps(old,indent=2),encoding='utf-8');temp.replace(SETTINGS)
    return public_settings()


def redact(text):
    try:
        env=environment()
        values={v for k,v in {**read_env_file(),**env}.items() if 'KEY' in k and len(v)>5}
        for value in values:text=text.replace(value,'[REDACTED]')
    except Exception:pass
    return text


def allowed_path(value):
    if not isinstance(value,str) or not value.strip():raise ValueError('Provide a file or directory path.')
    return Path(value).expanduser().resolve()


def environment_access(op):
    interpreters={**DEFAULTS,**read_settings().get('interpreters',{})}
    return check_environment_access(op,ENVIRONMENTS.read_registry(),interpreters,PROFILES)


def environment_status():
    result=ENVIRONMENTS.status()
    interpreters={**DEFAULTS,**read_settings().get('interpreters',{})}
    for profile in result['profiles']:
        profile['suggested_python']=interpreters.get(profile['roles'][0],'')
    for record in result['environments']:
        record['limitations']=[]
        if record['profile']=='core' and not record.get('validation',{}).get('versions',{}).get('matplotlib'):
            record['limitations'].append('3D animation: Matplotlib validation required.')
    return result


def require_environment(op):
    result=environment_access(op)
    if not result['ready']:raise ValueError(result['message'])
    return result


def worker_command(op,fields,tabs,folder):
    access=require_environment(op)
    cfg=read_settings();interpreters={**DEFAULTS,**cfg.get('interpreters',{})}
    selected=access['role']
    interpreter=Path(interpreters[selected])
    if not interpreter.is_file():raise ValueError(f'Python environment missing: {interpreter}')
    fields={**fields,'projectionPython':interpreters['projection']}
    for key,value in {'neighbors':15,'clusters':8,'minCluster':3,'sampleStep':12,'maxUpdates':20,'part':'Head','format':'DLC CSV'}.items():
        if fields.get(key) in (None,''):fields[key]=value
    spec={'op':op,'fields':fields,'tabs':tabs,'result':str(folder/'result.json')}
    (folder/'job.json').write_text(json.dumps(spec,ensure_ascii=False),encoding='utf-8')
    return [str(interpreter),str(UI/'worker.py'),str(folder/'job.json')],environment(cfg)


OPS={'reprojection','filter','convert','pose','triangulate','preprocess','previews','describe','embed','package','session','review','pose-previews',
     'analysis:keywords','analysis:major','analysis:fine','analysis:novel','analysis:temporal','analysis:phenotype','chat','api-test'}


def run_job(job):
    folder=Path(job['folder'])
    try:
        command,env=worker_command(job['op'],job['fields'],job.get('tabs',{}),folder)
        job['status']='running'
        flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',creationflags=flags)
        job['process']=process
        for line in process.stdout:
            line=redact(line.rstrip())
            with LOCK:job['log']=(job['log']+[line])[-100:]
        code=process.wait()
        if job.get('cancelled'):
            job['status']='cancelled'
        elif code:
            job['status']='failed';job['error']='\n'.join(job['log'][-10:]) or f'Worker exited {code}'
        else:
            job['result']=json.loads((folder/'result.json').read_text(encoding='utf-8'))
            job['status']='completed'
    except Exception as exc:
        job['status']='failed';job['error']=redact(str(exc))
    finally:
        job.pop('process',None)
        job['finished_at']=datetime.now().isoformat()
        BUSY.release()


def start_job(body):
    op=body.get('op')
    if op not in OPS:raise ValueError('Unsupported operation.')
    require_environment(op)
    if op=='reprojection':
        from workflow_inputs import reprojection_inputs
        reprojection_inputs(body.get('fields',{}))
    if op=='triangulate':
        from workflow_inputs import triangulation_inputs
        triangulation_inputs(body.get('fields',{}))
    if ENVIRONMENTS.state['status'] in {'queued','creating','installing','validating'}:raise ValueError('Wait for environment validation to finish.')
    if not BUSY.acquire(blocking=False):raise ValueError('Another task is running. Wait or cancel it first.')
    jid=secrets.token_hex(8);folder=RUNTIME/jid
    try:
        folder.mkdir(parents=True)
        job={'id':jid,'op':op,'monitor':str(body.get('monitor') or op)[:100],'fields':{**body.get('fields',{}),'_jobId':jid},'tabs':body.get('tabs',{}),
             'folder':str(folder),'status':'queued','log':[],'created_at':datetime.now().isoformat()}
        JOBS[jid]=job
        threading.Thread(target=run_job,args=(job,),daemon=True).start()
        return {'id':jid}
    except Exception:
        BUSY.release();raise


def public_job(job):
    return {k:v for k,v in job.items() if k not in {'process','fields','folder','tabs'}}


def preview_frames(body):
    directory=allowed_path(body.get('previews'))
    if not directory.is_dir():raise ValueError('Frame image directory not found.')
    ids=sorted(int(p.stem) for p in directory.iterdir() if p.is_file() and p.suffix=='.jpg' and len(p.stem)==8 and p.stem.isascii() and p.stem.isdigit())
    return {'frame_ids':ids}


def local_preview(body):
    import tempfile
    RUNTIME.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='preview-',dir=RUNTIME) as tmp:
        folder=Path(tmp)
        command,env=worker_command('preview-image',body,{},folder)
        run=subprocess.run(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
            text=True,encoding='utf-8',errors='replace',timeout=45,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if run.returncode:raise ValueError(redact(run.stderr[-2000:]))
        return json.loads((folder/'result.json').read_text(encoding='utf-8'))


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass

    def send(self,value,status=200):
        data=json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
        self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Content-Length',str(len(data)))
        self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
        self.end_headers();self.wfile.write(data)

    def trusted(self,post=False):
        expected=f'127.0.0.1:{PORT}'
        if self.headers.get('Host') not in {expected,f'localhost:{PORT}'}:return False
        if post:
            return self.headers.get('X-MouseGPT-Token')==TOKEN and self.headers.get('Origin') in {f'http://{expected}',f'http://localhost:{PORT}'}
        # IDE iframe previews may block third-party cookies. The page already
        # receives a per-process token; accept it for same-host reads as well.
        return (self.headers.get('X-MouseGPT-Token')==TOKEN or
                f'mousegpt={TOKEN}' in self.headers.get('Cookie','').split('; '))

    def do_GET(self):
        route=urlsplit(self.path).path
        if route in {'/','/ui','/ui/','/ui/index.html'}:
            if self.headers.get('Host') not in {f'127.0.0.1:{PORT}',f'localhost:{PORT}'}:
                return self.send({'error':'Local requests only'},403)
            data=(UI/'index.html').read_text(encoding='utf-8').replace('</head>',f'<script>window.__TOKEN__="{TOKEN}";window.__BACKEND_ID__="{BACKEND_ID}";</script></head>').encode()
            self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8')
            self.send_header('Content-Length',str(len(data)))
            self.send_header('Set-Cookie',f'mousegpt={TOKEN}; HttpOnly; SameSite=Strict; Path=/')
            self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data);return
        if route.startswith('/api/'):
            if not self.trusted():return self.send({'error':'Open the UI in this browser first.'},403)
            try:
                if route=='/api/status':return self.send({'connected':True,'backend_id':BACKEND_ID,'server_pid':os.getpid(),'supported_operations':sorted(OPS),'jobs':[public_job(j) for j in JOBS.values()]})
                if route=='/api/settings':return self.send(public_settings())
                if route=='/api/environment-access':return self.send(environment_access(parse_qs(urlsplit(self.path).query).get('op',[''])[0]))
                if route=='/api/environments':return self.send(environment_status())
                if route=='/api/agent':
                    import urllib.request
                    status='stopped';error=''
                    if AGENT is not None:
                        if AGENT.poll() is not None:
                            status='failed';log=RUNTIME/'agent.log'
                            error=redact(log.read_text(encoding='utf-8',errors='replace')[-4000:]) if log.exists() else 'Agent exited.'
                        else:
                            status='starting'
                            try:
                                with urllib.request.urlopen('http://127.0.0.1:7861/',timeout=1) as r:
                                    if r.status==200:status='ready'
                            except Exception:pass
                    return self.send({'status':status,'url':'http://127.0.0.1:7861/','error':error})
                if route=='/api/prompt':
                    source=(ROOT/'2_VLM/prompt_test/data/test_data.jsonl').read_text(encoding='utf-8').splitlines()[0]
                    return self.send({'prompt':json.loads(source)['query'].split('The first two photos shot at different view at the same time.')[0].strip()})
                if route.startswith('/api/jobs/'):
                    return self.send(public_job(JOBS[route.rsplit('/',1)[1]]))
                if route=='/api/media':
                    q=parse_qs(urlsplit(self.path).query);job=JOBS[q['job'][0]]
                    if job.get('status')!='completed':raise ValueError('No completed video result.')
                    index=int(q.get('index',['0'])[0])
                    allowed={v['index'] for v in job.get('result',{}).get('videos',[])}
                    if index not in allowed:raise ValueError('Choose a generated video result.')
                    file=Path(job['result']['files'][index])
                    size=file.stat().st_size;start=0;end=size-1;partial=False
                    requested=self.headers.get('Range')
                    if requested:
                        import re
                        match=re.fullmatch(r'bytes=(\d*)-(\d*)',requested)
                        if not match or not any(match.groups()):
                            self.send_response(416);self.send_header('Content-Range',f'bytes */{size}');self.end_headers();return
                        left,right=match.groups()
                        if left:start=int(left);end=min(int(right),size-1) if right else size-1
                        else:start=max(0,size-int(right))
                        if start>end or start>=size:
                            self.send_response(416);self.send_header('Content-Range',f'bytes */{size}');self.end_headers();return
                        partial=True
                    self.send_response(206 if partial else 200)
                    self.send_header('Content-Type','video/mp4');self.send_header('Accept-Ranges','bytes')
                    self.send_header('Content-Length',str(end-start+1))
                    if partial:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
                    self.end_headers()
                    try:
                        with file.open('rb') as handle:
                            handle.seek(start);remaining=end-start+1
                            while remaining:
                                chunk=handle.read(min(1024*1024,remaining))
                                if not chunk:break
                                self.wfile.write(chunk);remaining-=len(chunk)
                    except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass
                    return
                if route=='/api/download':
                    q=parse_qs(urlsplit(self.path).query);job=JOBS[q['job'][0]]
                    file=Path(job['result']['files'][int(q.get('index',['0'])[0])])
                    if not file.is_file():raise ValueError('Choose a result file, not a directory.')
                    self.send_response(200);self.send_header('Content-Type',mimetypes.guess_type(file.name)[0] or 'application/octet-stream')
                    self.send_header('Content-Disposition',f'attachment; filename="result{file.suffix}"');self.end_headers()
                    with file.open('rb') as handle:
                        import shutil;shutil.copyfileobj(handle,self.wfile)
                    return
                return self.send({'error':'Not found'},404)
            except Exception as exc:return self.send({'error':redact(str(exc))},400)
        static={'/ui/stage4_preview.js':'stage4_preview.js','/ui/stage1_review.js':'stage1_review.js','/ui/workbench.js':'workbench.js','/ui/environment_page.js':'environment_page.js','/ui/task_views.js':'task_views.js','/ui/api_settings_files.js':'api_settings_files.js','/ui/style.css':'style.css'}
        if route not in static:return self.send({'error':'Not found'},404)
        file=UI/static[route];data=file.read_bytes();self.send_response(200)
        self.send_header('Content-Type',('text/javascript' if file.suffix=='.js' else 'text/css')+'; charset=utf-8')
        self.send_header('Content-Length',str(len(data)))
        self.send_header('Cache-Control','no-cache');self.end_headers();self.wfile.write(data)

    def do_POST(self):
        global AGENT
        if not self.trusted(post=True):return self.send({'error':'Invalid request origin or token.'},403)
        try:
            size=int(self.headers.get('Content-Length',0))
            if size>2_000_000:raise ValueError('Request too large.')
            body=json.loads(self.rfile.read(size) or b'{}')
            route=urlsplit(self.path).path
            if route=='/api/settings':return self.send(save_settings(body))
            if route=='/api/settings/export':return self.send(export_api_settings(body.get('include_api_keys') is True))
            if route=='/api/environments':
                if BUSY.locked():raise ValueError('Wait for the current task to finish.')
                action=body.get('action')
                if action=='activate':
                    if AGENT is not None and AGENT.poll() is None:raise ValueError('Close the running Agent before changing environments.')
                    ENVIRONMENTS.activate(body.get('id'),persist=lambda paths:save_settings({'interpreters':paths}))
                    return self.send(ENVIRONMENTS.status())
                if action=='cancel':
                    ENVIRONMENTS.cancel();return self.send(ENVIRONMENTS.status())
                return self.send(ENVIRONMENTS.start(action,body.get('profile'),body.get('python')))
            if route=='/api/jobs':return self.send(start_job(body),202)
            if route=='/api/preview-frames':return self.send(preview_frames(body))
            if route=='/api/preview':return self.send(local_preview(body))
            if route=='/api/agent':
                require_environment('agent-launch')
                if AGENT is not None and AGENT.poll() is None:
                    return self.send({'status':'starting','url':'http://127.0.0.1:7861/'})
                file=allowed_path(body.get('h5'));images=allowed_path(body.get('previews'));dest=allowed_path(body.get('out4'))
                if not file.is_file() or not images.is_dir():raise ValueError('Select an existing HDF5 and image directory.')
                import socket
                with socket.socket() as sock:
                    if sock.connect_ex(('127.0.0.1',7861))==0:raise ValueError('Port 7861 is already in use.')
                env=environment();env.update(MOUSEGPT_NLUI_H5=str(file),MOUSEGPT_NLUI_IMAGES=str(images),MOUSEGPT_NLUI_WORKSPACE=str(dest),MOUSEGPT_NLUI_PORT='7861')
                log=(RUNTIME/'agent.log').open('w',encoding='utf-8')
                try:
                    interpreter=read_settings().get('interpreters',{}).get('agent',DEFAULTS['agent'])
                    AGENT=subprocess.Popen([interpreter,str(ROOT/'4_NLUI/agent.py')],cwd=ROOT/'4_NLUI',env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                finally:log.close()
                return self.send({'status':'starting','url':'http://127.0.0.1:7861/'})
            if route=='/api/cancel':
                job=JOBS[body['id']];job['cancelled']=True
                if job.get('process'):
                    if os.name=='nt':subprocess.run(['taskkill','/PID',str(job['process'].pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    else:job['process'].terminate()
                return self.send({'status':'cancelling'})
            if route=='/api/reviews':
                dest=allowed_path(body['output']);dest.mkdir(parents=True,exist_ok=True)
                target=dest/f'review-{datetime.now().strftime("%Y%m%d-%H%M%S")}-{secrets.token_hex(3)}.json'
                target.write_text(json.dumps(body['reviews'],ensure_ascii=False,indent=2),encoding='utf-8')
                reviewed=dest.parent/'reviewed_descriptions'
                reviewed.mkdir(exist_ok=True)
                for frame,record in body['reviews'].items():
                    if not str(frame).isdigit():raise ValueError('Invalid review frame ID.')
                    original=record.get('original')
                    if not isinstance(original,dict):continue
                    updated={k:str(record.get(k+'-revised','')).strip() or v for k,v in original.items()}
                    file=reviewed/f'{str(frame).zfill(8)}.json'
                    file.write_text(json.dumps(updated,ensure_ascii=False,indent=2),encoding='utf-8')
                return self.send({'path':str(target),'reviewed_descriptions':str(reviewed)})
            return self.send({'error':'Not found'},404)
        except Exception as exc:self.send({'error':redact(str(exc))},400)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8878);args=p.parse_args();PORT=args.port
    try:service=LocalUIServer(('127.0.0.1',PORT),Handler)
    except OSError as exc:
        print(f'Cannot start MouseGPT UI on port {PORT}: {exc}. Another UI service may already be running. Stop it before restarting.',file=sys.stderr)
        sys.exit(1)
    RUNTIME.mkdir(parents=True,exist_ok=True)
    (RUNTIME/'server.pid').write_text(str(os.getpid()))
    print(f'MouseGPT UI: http://127.0.0.1:{PORT}/ui/',flush=True)
    with service:service.serve_forever()
