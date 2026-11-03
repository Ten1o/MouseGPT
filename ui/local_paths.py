"""Runtime paths independent of the source checkout / future desktop host."""
import os
import sys
from pathlib import Path


def app_home():
    configured=os.environ.get('MOUSEGPT_UI_HOME')
    if configured:return Path(configured).expanduser().resolve()
    legacy=Path.home()/'.local/share'
    if sys.platform=='darwin':base=legacy if (legacy/'MouseGPT').exists() else Path.home()/'Library/Application Support'
    elif os.name=='nt':base=Path(os.environ.get('LOCALAPPDATA',legacy))
    else:base=Path(os.environ.get('XDG_DATA_HOME',legacy))
    return base/'MouseGPT'


APP_HOME=app_home()
SETTINGS_FILE=APP_HOME/'settings.json'
RUNTIME_DIR=APP_HOME/'tasks'
ENVIRONMENT_DIR=APP_HOME/'environments'


def backend_id(home=None, machine=None):
    """Stable per deployment and machine; never expose machine names or paths."""
    import hashlib
    import platform
    import secrets
    folder=Path(home) if home is not None else APP_HOME
    folder.mkdir(parents=True,exist_ok=True)
    identity=folder/'backend-id'
    try:identifier=identity.read_text(encoding='ascii').strip()
    except FileNotFoundError:
        try:
            with identity.open('x',encoding='ascii') as stream:
                identifier=secrets.token_hex(16);stream.write(identifier)
        except FileExistsError:identifier=identity.read_text(encoding='ascii').strip()
    if len(identifier)!=32 or any(c not in '0123456789abcdef' for c in identifier):raise ValueError('Invalid backend identity file.')
    host=platform.node() if machine is None else machine
    return hashlib.sha256((identifier+'|'+host).encode()).hexdigest()
