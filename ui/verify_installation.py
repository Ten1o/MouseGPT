"""Exercise the real installer in an isolated caller-selected directory."""
import argparse
import json
from pathlib import Path
import time
from environment_manager import EnvironmentManager

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('profile',choices=['core','agent']);p.add_argument('--home',required=True);args=p.parse_args()
    manager=EnvironmentManager(Path(args.home).resolve());manager.start('install',args.profile);previous=None
    while True:
        status=manager.status()['operation']
        if status['status']!=previous:print(args.profile+': '+status['status'],flush=True);previous=status['status']
        if status['status'] in {'failed','cancelled','ready'}:
            print(json.dumps(status,indent=2),flush=True)
            if status['status']!='ready':raise SystemExit(1)
            row=next(r for r in manager.read_registry()['environments'] if r['id']==status['environment_id'])
            print(json.dumps(row['validation'],indent=2),flush=True);break
        time.sleep(1)
