"""Exercise the browser/backend activation contract using only temporary state."""
import http.cookiejar
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import re
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.request
import urllib.error

import server
from environment_manager import EnvironmentManager,atomic_json


class EnvironmentHTTP(unittest.TestCase):
    def test_ide_preview_reads_without_cookies(self):
        httpd=ThreadingHTTPServer(('127.0.0.1',0),server.Handler)
        port=httpd.server_address[1]
        thread=threading.Thread(target=httpd.serve_forever,daemon=True)
        with patch.object(server,'PORT',port):
            thread.start();base=f'http://127.0.0.1:{port}'
            try:
                html=urllib.request.urlopen(base+'/ui/').read().decode()
                token=re.search(r'window.__TOKEN__="([^"]+)"',html).group(1)
                request=urllib.request.Request(base+'/api/status',headers={'X-MouseGPT-Token':token})
                self.assertTrue(json.load(urllib.request.urlopen(request))['connected'])
                for headers in ({},{'X-MouseGPT-Token':'invalid'},
                                {'X-MouseGPT-Token':token,'Host':'invalid.example'}):
                    with self.assertRaises(urllib.error.HTTPError) as exc:
                        urllib.request.urlopen(urllib.request.Request(base+'/api/status',headers=headers))
                    self.assertEqual(exc.exception.code,403)
                request=urllib.request.Request(base+'/api/environments',data=b'{}',
                    headers={'X-MouseGPT-Token':token,'Origin':'https://invalid.example'})
                with self.assertRaises(urllib.error.HTTPError) as exc:
                    urllib.request.urlopen(request)
                self.assertEqual(exc.exception.code,403)
            finally:
                httpd.shutdown();thread.join();httpd.server_close()

    def test_activation_persists_interpreters_and_blocks_when_task_busy(self):
        with tempfile.TemporaryDirectory() as tmp:
            home=Path(tmp);manager=EnvironmentManager(home);python=home/'python.exe';python.touch()
            atomic_json(manager.registry,{'schema':1,'environments':[{'id':'checked','profile':'core','release':'custom','python':str(python),'managed':False,'validation':{'passed':True,'checks':['test'],'python':'3.12.11'}}],'active':{}})
            httpd=ThreadingHTTPServer(('127.0.0.1',0),server.Handler);port=httpd.server_address[1]
            thread=threading.Thread(target=httpd.serve_forever,daemon=True)
            with patch.object(server,'PORT',port),patch.object(server,'ENVIRONMENTS',manager),patch.object(server,'SETTINGS',home/'settings.json'),patch.object(server,'AGENT',None):
                thread.start();base=f'http://127.0.0.1:{port}'
                opener=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
                html=opener.open(base+'/ui/').read().decode();token=re.search(r'window.__TOKEN__="([^"]+)"',html).group(1)
                request=urllib.request.Request(base+'/api/environments',data=json.dumps({'action':'activate','id':'checked'}).encode(),headers={'Content-Type':'application/json','Origin':base,'X-MouseGPT-Token':token})
                result=json.load(opener.open(request));self.assertEqual(result['active']['core'],'checked')
                prefs=json.loads((home/'settings.json').read_text())
                for role in ('analysis','preprocess','projection'):self.assertEqual(prefs['interpreters'][role],str(python))
                server.BUSY.acquire()
                try:
                    with self.assertRaises(urllib.error.HTTPError) as exc:opener.open(request)
                    self.assertEqual(exc.exception.code,400)
                finally:server.BUSY.release()
                bad=urllib.request.Request(base+'/api/environments',data=b'{}',headers={'Content-Type':'application/json','Origin':'https://invalid.example'})
                with self.assertRaises(urllib.error.HTTPError) as exc:opener.open(bad)
                self.assertEqual(exc.exception.code,403)
                httpd.shutdown();thread.join();httpd.server_close()


if __name__=='__main__':unittest.main()
