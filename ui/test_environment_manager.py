"""Offline lifecycle checks; installation commands are replaced with deterministic fixtures."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import environment_manager as em
from environment_validation import application_fingerprint

def report(profile, **values):
    return {"passed":True, "application_fingerprint":application_fingerprint(profile), **values}



class EnvironmentLifecycle(unittest.TestCase):
    def manager(self,tmp):return em.EnvironmentManager(Path(tmp))

    def test_platform_matrix_and_lock_selection(self):
        cases=[('Windows','AMD64','',('',''),'windows-x86_64'),('Linux','x86_64','',('glibc','2.35'),'linux-x86_64'),('Linux','aarch64','',('glibc','2.28'),'linux-aarch64'),('Darwin','arm64','14.0',('',''),'macos-arm64'),('Darwin','x86_64','15.1',('',''),'macos-x86_64'),('Darwin','arm64','13.6',('',''),None),('Linux','x86_64','',('musl','1.2.4'),None),('Linux','aarch64','',('glibc','2.27'),None),('Windows','ARM64','',('',''),None)]
        for system,machine,mac,libc,expected in cases:
            with self.subTest(system=system,machine=machine,version=mac,libc=libc),patch.object(em.platform,'architecture',return_value=('64bit','')),patch.object(em.platform,'system',return_value=system),patch.object(em.platform,'machine',return_value=machine),patch.object(em.platform,'mac_ver',return_value=(mac,('','',''),'')),patch.object(em.platform,'libc_ver',return_value=libc):
                self.assertEqual(em.platform_key(),expected)
                if expected:
                    for profile in ('core','agent'):self.assertTrue(em.dependency_lock(profile).is_file())
                else:self.assertFalse(em.dependency_lock('core').is_file())
        with patch.object(em.platform,'architecture',return_value=('32bit','')):self.assertIsNone(em.platform_key())

    def test_extra_platform_bootstrap_hash_and_executable(self):
        from io import BytesIO
        import hashlib,zipfile
        for key in ('macos-arm64','macos-x86_64','linux-aarch64'):
            with self.subTest(platform=key),tempfile.TemporaryDirectory() as tmp,patch.object(em,'platform_key',return_value=key):
                content=BytesIO()
                with zipfile.ZipFile(content,'w') as archive:archive.writestr('uv.data/scripts/uv',b'fixture-tool')
                payload=content.getvalue()
                with patch.object(em,'bootstrap_artifact',return_value={'url':'https://example.com/uv.whl','sha256':hashlib.sha256(payload).hexdigest()}),patch('urllib.request.urlopen',return_value=BytesIO(payload)):
                    tool=self.manager(tmp).bootstrap()
                self.assertEqual(tool.name,'uv');self.assertEqual(tool.parent.name,key)
                self.assertEqual(tool.read_bytes(),b'fixture-tool')

    def test_platform_recipe_versions_do_not_modify_windows(self):
        for key,profile,release in [('macos-x86_64','core','2026.11.02.1'),('macos-arm64','agent','2026.11.03.1'),('windows-x86_64','core','2026.10.29.2'),('windows-x86_64','agent','2026.11.03.1')]:
            with patch.object(em,'platform_key',return_value=key):self.assertEqual(em.profile_spec(profile)['release'],release)

    def test_linux_bootstrap_extracts_hash_checked_uv(self):
        from io import BytesIO
        import hashlib,zipfile
        archive=BytesIO()
        with zipfile.ZipFile(archive,'w') as z:z.writestr('uv.data/scripts/uv',b'linux-tool-fixture')
        content=archive.getvalue()
        with tempfile.TemporaryDirectory() as tmp,patch.object(em,'platform_key',return_value='linux-x86_64'),patch.object(em,'LINUX_UV_HASH',hashlib.sha256(content).hexdigest()),patch('urllib.request.urlopen',return_value=BytesIO(content)):
            tool=self.manager(tmp).bootstrap()
            self.assertEqual(tool.name,'uv');self.assertEqual(tool.read_bytes(),b'linux-tool-fixture')

    def test_each_platform_install_uses_matching_recipe_and_interpreter(self):
        keys=('windows-x86_64','linux-x86_64','linux-aarch64','macos-arm64','macos-x86_64')
        for key in keys:
            for profile in ('core','agent'):
                with self.subTest(platform=key,profile=profile),tempfile.TemporaryDirectory() as tmp,patch.object(em,'platform_key',return_value=key):
                    manager=self.manager(tmp);commands=[];spec=em.profile_spec(profile)
                    def command(args):
                        commands.append([str(a) for a in args])
                        if 'venv' in args:
                            executable=Path(args[-1])/('Scripts/python.exe' if key=='windows-x86_64' else 'bin/python');executable.parent.mkdir(parents=True);executable.touch()
                        if str(args[-1]).endswith('.json'):em.atomic_json(Path(args[-1]),report(profile,python=spec['python'],checks=['fixture']))
                    with patch.object(manager,'bootstrap',return_value=Path(tmp)/'uv'),patch.object(manager,'command',side_effect=command):manager._run('install',profile,None)
                    self.assertEqual(manager.state['status'],'ready',manager.state)
                    row=manager.read_registry()['environments'][0]
                    ending='/Scripts/python.exe' if key=='windows-x86_64' else '/bin/python'
                    self.assertTrue(row['python'].replace(chr(92),'/').endswith(ending))
                    self.assertTrue(any(c[-1]==str(em.dependency_lock(profile)) and '--require-hashes' in c for c in commands))
                    self.assertEqual(row['platform_key'],key);self.assertEqual(row['release'],spec['release'])
                    self.assertFalse(manager.read_registry()['active'])

    def test_pose_can_be_validated_without_installation_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager=self.manager(tmp);python=Path(tmp)/'python.exe';python.touch()
            def checked(args):em.atomic_json(Path(args[-1]),report('pose',python='3.10.18',checks=['MMPose','EasyMocap']))
            with patch.object(manager,'command',side_effect=checked):manager._run('check','pose',python)
            self.assertEqual(manager.state['status'],'ready')
            record=manager.read_registry()['environments'][0]
            activated=manager.activate(record['id'])
            self.assertEqual(set(activated),{'pose'})
            self.assertEqual(Path(activated['pose']).resolve(),python.resolve())
            with self.assertRaises(ValueError):manager.start('install','pose')

    def test_failed_validation_does_not_register_or_replace_active(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager=self.manager(tmp);python=Path(tmp)/'python.exe';python.touch()
            before={'schema':1,'environments':[],'active':{'core':'previous-valid-id'}}
            em.atomic_json(manager.registry,before)
            with patch.object(manager,'command',side_effect=RuntimeError('incompatible dependency')):
                manager._run('check','core',python)
            self.assertEqual(manager.state['status'],'failed')
            self.assertEqual(manager.read_registry(),before)

    def test_validated_custom_environment_requires_explicit_activation(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager=self.manager(tmp);python=Path(tmp)/'python.exe';python.touch()
            def checked(args):
                target=Path(args[-1]);em.atomic_json(target,report('core',python='3.12.11',checks=['offline check']))
            with patch.object(manager,'command',side_effect=checked):manager._run('check','core',python)
            self.assertEqual(manager.state['status'],'ready')
            registry=manager.read_registry();self.assertFalse(registry['active'])
            row=registry['environments'][0]
            self.assertFalse(row['managed']);self.assertEqual(row['release'],'custom')
            result=manager.activate(row['id'])
            self.assertEqual(set(result),{'analysis','preprocess','projection'})
            self.assertEqual(manager.read_registry()['active']['core'],row['id'])
            self.assertTrue(python.exists())

    def test_unknown_profile_rejected_and_other_process_lock_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            one=self.manager(tmp);two=self.manager(tmp)
            with self.assertRaises(ValueError):one.start('install','not-a-profile')
            with one.disk_lock():
                with self.assertRaises(ValueError):
                    with two.disk_lock():pass

    def test_settings_failure_does_not_mark_environment_active(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager=self.manager(tmp);python=Path(tmp)/'python.exe';python.touch()
            registry={'schema':1,'environments':[{'id':'candidate','profile':'core','python':str(python),'validation':report('core')}],'active':{'core':'previous'}}
            em.atomic_json(manager.registry,registry)
            with self.assertRaises(OSError):manager.activate('candidate',persist=lambda _:(_ for _ in ()).throw(OSError('settings unavailable')))
            self.assertEqual(manager.read_registry()['active'],{'core':'previous'})

    def test_cancel_does_not_register_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager=self.manager(tmp);python=Path(tmp)/'python.exe';python.touch();manager.cancelled.set()
            manager._run('check','agent',python)
            self.assertEqual(manager.state['status'],'cancelled')
            self.assertEqual(manager.read_registry()['environments'],[])

    def test_bootstrap_checksum_failure_never_executes_tool(self):
        from io import BytesIO
        with tempfile.TemporaryDirectory() as tmp:
            manager=self.manager(tmp)
            with patch('urllib.request.urlopen',return_value=BytesIO(b'wrong archive')):
                with self.assertRaisesRegex(ValueError,'checksum'):manager.bootstrap()
            self.assertFalse((Path(tmp)/'tools'/f'uv-{em.UV_VERSION}'/'uv.exe').exists())


if __name__=='__main__':unittest.main()
