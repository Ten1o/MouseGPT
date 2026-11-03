"""Versioned, isolated environment installation with validation before use.

No executable path, package name or install command is accepted from HTTP input.
Only the checked-in profiles and their hash-locked package sets are installable.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import secrets
import shutil
import subprocess
import threading
import urllib.request
import zipfile

from local_paths import APP_HOME
from environment_validation import application_fingerprint, validation_is_current
from process_control import process_group_options, terminate_process_tree

HERE=Path(__file__).resolve().parent
PROFILE_DIR=HERE/'environments'
UV_VERSION='0.8.22'
UV_URL='https://files.pythonhosted.org/packages/38/82/94f08992eeb193dc3d5baac437d1867cd37f040f34c7b1a4b1bde2bc4b4b/uv-0.8.22-py3-none-win_amd64.whl'
UV_HASH='cda349c9ea53644d8d9ceae30db71616b733eb5330375ab4259765aef494b74e'
PROFILES={
    'pose':{'name':'Pose estimation & reconstruction','release':'custom','python':'existing',
            'roles':['pose'],'install_supported':False,'description':'Configure the Pose environment manually, then validate it here and select Use.'},
    'core':{'name':'Core workflow · CPU & API','release':'2026.10.29.2','python':'3.12.11',
            'roles':['analysis','preprocess','projection'],'description':'Image preprocessing, Vision API, embeddings, HDF5 and behavior analysis.'},
    'agent':{'name':'Tool-calling Agent','release':'2026.11.03.1','python':'3.12.11',
             'roles':['agent'],'description':'The Gradio / LangChain Agent using the shared public application requirements.'},
}


def timestamp():return datetime.now(timezone.utc).isoformat()


def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+secrets.token_hex(4)+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    temp.replace(path)


def clean_env(home):
    env={k:v for k,v in os.environ.items() if not k.startswith(('UV_','PIP_','CONDA_','MOUSEGPT_')) and k not in {'PYTHONHOME','PYTHONPATH','VIRTUAL_ENV'}}
    env.update(UV_CACHE_DIR=str(home/'cache'),UV_PYTHON_INSTALL_DIR=str(home/'python'),UV_PYTHON_BIN_DIR=str(home/'bin'),
               UV_NO_MODIFY_PATH='1',PYTHONNOUSERSITE='1',PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1',
               GRADIO_ANALYTICS_ENABLED='False',LANGCHAIN_TRACING_V2='false',
               UV_CONCURRENT_DOWNLOADS='2',UV_CONCURRENT_INSTALLS='1')
    if platform.system()=='Linux':
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):env.setdefault(key,'1')
    return env


LINUX_UV_URL='https://files.pythonhosted.org/packages/3a/a9/a83cee9b8cf63e57ce64ba27c77777cc66410e144fd178368f55af1fa18d/uv-0.8.22-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'
LINUX_UV_HASH='8efec4ef5acddc35f0867998c44e0b15fc4dace1e4c26d01443871a2fbb04bf6'


EXTRA_TOOLS={'macos-x86_64': {'url': 'https://files.pythonhosted.org/packages/28/e9/813f7eb9fb9694c4024362782c8933e37887b5195e189f80dc40f2da5958/uv-0.8.22-py3-none-macosx_10_12_x86_64.whl', 'sha256': '89944e99b04cc8542cb5931306f1c593f00c9d6f2b652fffc4d84d12b915f911'}, 'macos-arm64': {'url': 'https://files.pythonhosted.org/packages/d7/ca/bf37d86af6e16e45fa2b1a03300784ff3297aa9252a23dfbeaf6e391e72e/uv-0.8.22-py3-none-macosx_11_0_arm64.whl', 'sha256': '6706b782ad75662df794e186d16b9ffa4946d57c88f21d0eadfd43425794d1b0'}, 'linux-aarch64': {'url': 'https://files.pythonhosted.org/packages/0f/0c/71d5d5d3fca7aa788d63297a06ca26d3585270342277b52312bb693b100c/uv-0.8.22-py3-none-manylinux_2_28_aarch64.whl', 'sha256': '9eb3b4abfa25e07d7e1bb4c9bb8dbbdd51878356a37c3c4a2ece3d68d4286f28'}}
PLATFORM_LABELS={'windows-x86_64':'Windows x64','linux-x86_64':'Linux x64 (glibc 2.28+)','linux-aarch64':'Linux ARM64 (glibc 2.28+)','macos-x86_64':'macOS 14+ Intel','macos-arm64':'macOS 14+ Apple Silicon'}


def version_at_least(value,minimum):
    parts=re.findall(r'\d+',value)
    return bool(parts) and tuple((list(map(int,parts))+[0,0])[:len(minimum)])>=minimum


def platform_key():
    if platform.architecture()[0]!='64bit':return None
    machine=platform.machine().lower();system=platform.system()
    x64=machine in {'amd64','x86_64'};arm64=machine in {'arm64','aarch64'}
    if system=='Windows' and x64:return 'windows-x86_64'
    if system=='Linux':
        libc,version=platform.libc_ver()
        if libc!='glibc' or not version_at_least(version,(2,28)):return None
        if x64:return 'linux-x86_64'
        if arm64:return 'linux-aarch64'
    if system=='Darwin' and version_at_least(platform.mac_ver()[0],(14,0)):
        if x64:return 'macos-x86_64'
        if arm64:return 'macos-arm64'
    return None


def dependency_lock(profile):
    key=platform_key()
    suffix='' if key=='windows-x86_64' else '-'+(key or 'unsupported')
    return PROFILE_DIR/f'{profile}{suffix}.lock'


def profile_spec(profile):
    spec=dict(PROFILES[profile])
    if (profile,platform_key()) == ('core','macos-x86_64'):
        spec['release']='2026.11.02.1'
    return spec


def bootstrap_artifact():
    key=platform_key()
    if key=='windows-x86_64':return {'url':UV_URL,'sha256':UV_HASH}
    if key=='linux-x86_64':return {'url':LINUX_UV_URL,'sha256':LINUX_UV_HASH}
    if key in EXTRA_TOOLS:return EXTRA_TOOLS[key]
    raise ValueError('No installation tool is available for this platform.')


def system_report(home=APP_HOME):
    parent=Path(home)
    while not parent.exists() and parent!=parent.parent:parent=parent.parent
    free=shutil.disk_usage(parent).free
    key=platform_key();supported=key is not None
    reason=''
    if not supported:reason='Automatic installation requires Windows x64, Linux x64/ARM64 with glibc 2.28+, or macOS 14+ Intel/Apple Silicon. Windows ARM64, 32-bit and musl Linux require manual setup.'
    elif free<3*1024**3:reason='At least 3 GB of free space is required.'
    return {'os':platform.system(),'os_version':platform.version(),'architecture':platform.machine(),
            'platform_key':key,'platform_label':PLATFORM_LABELS.get(key,'Unsupported platform'),
            'validation_status':'profile-specific',
            'supported':supported,'free_gb':round(free/(1024**3),1),'minimum_free_gb':3,
            'install_supported':supported and free>=3*1024**3,'app_home':str(home),'reason':reason}


class EnvironmentManager:
    def __init__(self,home=APP_HOME):
        self.home=Path(home);self.registry=self.home/'environment-registry.json'
        self._mutex=threading.RLock();self.cancelled=threading.Event();self.process=None
        self.state={'status':'idle','log':[]};self._thread=None

    def read_registry(self):
        if not self.registry.exists():return {'schema':1,'environments':[],'active':{}}
        return json.loads(self.registry.read_text(encoding='utf-8'))

    def status(self):
        with self._mutex:
            state=dict(self.state);state['log']=list(state.get('log',[]))
        registry=self.read_registry()
        records=[]
        for row in registry['environments']:
            row=dict(row);row['available']=Path(row['python']).is_file()
            row['validation_current']=validation_is_current(row['profile'],row.get('validation',{}))
            row['limitations']=[] if row['validation_current'] else ['Application requirements changed. Validate this environment again before using it.']
            records.append(row)
        return {'system':system_report(self.home),'profiles':[{**profile_spec(name),'id':name,'lock_available':dependency_lock(name).is_file()} for name in PROFILES],
                'environments':records,'active':registry.get('active',{}),'operation':state}

    def log(self,line):
        with self._mutex:self.state['log']=(self.state.get('log',[])+[str(line).rstrip()])[-80:]

    def command(self,args):
        if self.cancelled.is_set():raise RuntimeError('Operation cancelled.')
        process=subprocess.Popen([str(a) for a in args],env=clean_env(self.home),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            text=True,encoding='utf-8',errors='replace',**process_group_options())
        self.process=process
        if self.cancelled.is_set():self.cancel()
        try:
            for line in process.stdout:self.log(line)
            code=process.wait()
            if self.cancelled.is_set():raise RuntimeError('Operation cancelled.')
            if code:raise RuntimeError(f'Environment command failed (exit {code}). See installation log.')
        finally:self.process=None

    @contextmanager
    def disk_lock(self):
        self.home.mkdir(parents=True,exist_ok=True)
        handle=(self.home/'environment-operation.lock').open('a+b')
        if handle.tell()==0:handle.write(b'0');handle.flush()
        handle.seek(0)
        try:
            if os.name=='nt':
                import msvcrt;msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl;fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            handle.close();raise ValueError('Another MouseGPT instance is managing environments.')
        try:yield
        finally:
            if os.name=='nt':
                import msvcrt;handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl;fcntl.flock(handle.fileno(),fcntl.LOCK_UN)
            handle.close()

    def bootstrap(self):
        key=platform_key();artifact=bootstrap_artifact()
        executable='uv.exe' if key=='windows-x86_64' else 'uv'
        url,checksum=artifact['url'],artifact['sha256']
        directory=self.home/'tools'/f'uv-{UV_VERSION}'
        if key!='windows-x86_64':directory=directory/key
        tool=directory/executable
        receipt=tool.with_suffix('.sha256')
        if tool.is_file() and receipt.is_file() and hashlib.sha256(tool.read_bytes()).hexdigest()==receipt.read_text().strip():return tool
        self.log('Downloading fixed installation tool (uv '+UV_VERSION+').')
        with urllib.request.urlopen(url,timeout=60) as response:
            content=response.read(80*1024*1024+1)
        if len(content)>80*1024*1024 or hashlib.sha256(content).hexdigest()!=checksum:
            raise ValueError('Installer tool checksum mismatch; nothing was executed.')
        if self.cancelled.is_set():raise RuntimeError('Operation cancelled.')
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            names=[n for n in archive.namelist() if n.endswith('/'+executable)]
            if len(names)!=1:raise ValueError('Unexpected installer wheel layout.')
            exe=archive.read(names[0])
        tool.parent.mkdir(parents=True,exist_ok=True)
        temporary=tool.with_suffix('.downloading');temporary.write_bytes(exe);temporary.chmod(0o755);temporary.replace(tool)
        receipt.write_text(hashlib.sha256(exe).hexdigest())
        return tool

    def start(self,action,profile,python=None):
        if profile not in PROFILES:raise ValueError('Unknown environment profile.')
        if action not in {'install','check'}:raise ValueError('Unknown environment action.')
        with self._mutex:
            if self._thread and self._thread.is_alive():raise ValueError('An environment operation is already running.')
            if action=='install' and not system_report(self.home)['install_supported']:
                raise ValueError(system_report(self.home)['reason'])
            if action=='install' and PROFILES[profile].get('install_supported') is False:
                raise ValueError('Automatic installation is not available for this profile. Validate an existing environment.')
            if action=='check' and (not python or not Path(python).is_file()):raise ValueError('Select an existing Python executable.')
            self.cancelled.clear();self.state={'status':'queued','action':action,'profile':profile,'started_at':timestamp(),'log':[]}
            self._thread=threading.Thread(target=self._run,args=(action,profile,python),daemon=True);self._thread.start()
        return self.status()

    def _run(self,action,profile,python):
        try:
            with self.disk_lock():
                spec=profile_spec(profile);lock=dependency_lock(profile);target=None
                if action=='install' and not lock.is_file():raise ValueError('This release has no dependency lock.')
                lock_hash=hashlib.sha256(lock.read_bytes()).hexdigest() if lock.is_file() else None
                if action=='install':
                    tool=self.bootstrap();token=secrets.token_hex(5)
                    target=self.home/'environments'/f'{profile}-{spec["release"]}-{token}'
                    if target.exists():raise ValueError('New environment directory already exists.')
                    self.state.update(status='creating',directory=str(target));self.log('Creating isolated Python '+spec['python']+'.')
                    self.command([tool,'--no-config','venv','--managed-python','--python',spec['python'],target])
                    python=target/('Scripts/python.exe' if platform_key()=='windows-x86_64' else 'bin/python')
                    self.state['status']='installing';self.log('Installing the complete hash-locked dependency set.')
                    self.command([tool,'--no-config','pip','sync','--python',python,'--require-hashes','--only-binary',':all:',
                                  '--index-url','https://pypi.org/simple',lock])
                    self.command([tool,'--no-config','pip','check','--python',python])
                python=Path(python).expanduser().absolute()
                reports=self.home/'validation';reports.mkdir(parents=True,exist_ok=True)
                report=reports/(profile+'-'+secrets.token_hex(6)+'.json')
                self.state['status']='validating';self.log('Running offline compatibility and file-processing checks.')
                self.command([python,'-I',HERE/'environment_probe.py',profile,'--output',report])
                validation=json.loads(report.read_text(encoding='utf-8'))
                if not validation_is_current(profile,validation):raise ValueError('Environment validation did not match the current application. Run validation again.')
                if action=='install' and validation['python']!=spec['python']:raise ValueError('Installed Python does not match the release.')
                # Ready records are written only after every command succeeds.
                record={'id':secrets.token_hex(10),'profile':profile,'release':spec['release'] if action=='install' else 'custom','python':str(python),
                    'platform_key':platform_key(),'managed':action=='install','lock_sha256':lock_hash if action=='install' else None,
                    'validated_at':timestamp(),'validation':validation,'report':str(report)}
                registry=self.read_registry();registry['environments'].append(record);atomic_json(self.registry,registry)
                self.state.update(status='ready',environment_id=record['id'],finished_at=timestamp())
                self.log('Validation passed. Select Use to switch; previous environments are unchanged.')
        except Exception as exc:
            self.state.update(status='cancelled' if self.cancelled.is_set() else 'failed',error=str(exc),finished_at=timestamp())
            self.log(str(exc))
        finally:
            try:atomic_json(self.home/'last-environment-operation.json',self.state)
            except OSError:pass

    def activate(self,identifier,persist=None):
        with self.disk_lock():
            if self._thread and self._thread.is_alive():raise ValueError('Wait for environment management to finish.')
            registry=self.read_registry()
            record=next((r for r in registry['environments'] if r['id']==identifier),None)
            if not record or not Path(record['python']).is_file():raise ValueError('Validated environment not found.')
            if not validation_is_current(record['profile'],record.get('validation',{})):raise ValueError('Application requirements changed. Validate this environment again before using it.')
            roles={role:record['python'] for role in PROFILES[record['profile']]['roles']}
            if persist:persist(roles)
            registry.setdefault('active',{})[record['profile']]=identifier
            atomic_json(self.registry,registry)
            return roles

    def cancel(self):
        self.cancelled.set();process=self.process
        if process is not None:terminate_process_tree(process)
