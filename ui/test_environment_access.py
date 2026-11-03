import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from environment_access import access,same_python
from environment_manager import PROFILES,EnvironmentManager,atomic_json
import server
from environment_validation import application_fingerprint

def report(profile, **values):
    return {"passed":True, "application_fingerprint":application_fingerprint(profile), **values}



class EnvironmentAccessTests(unittest.TestCase):
    def test_virtual_environment_identity_does_not_follow_python_symlinks(self):
        with patch.object(Path,'resolve',side_effect=AssertionError('Do not follow venv interpreter symlinks')):
            self.assertFalse(same_python('/env/core/bin/python','/env/agent/bin/python'))
            self.assertTrue(same_python('/env/core/bin/python','/env/core/bin/python'))

    def test_validation_activation_and_selected_path_are_all_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            python=Path(tmp)/'python.exe';python.touch()
            interpreters={'analysis':str(python),'preprocess':str(python),'projection':str(python),'pose':str(Path(tmp)/'pose.exe')}
            record={'id':'core-one','profile':'core','python':str(python),'validation':report('core',versions={'matplotlib':'3.10.1'})}
            registry={'environments':[record],'active':{}}
            self.assertFalse(access('describe',registry,interpreters,PROFILES)['ready'])
            registry['active']['core']='core-one'
            self.assertTrue(access('describe',registry,interpreters,PROFILES)['ready'])
            self.assertFalse(access('agent-launch',registry,interpreters,PROFILES)['ready'])
            interpreters['analysis']=str(Path(tmp)/'other.exe')
            self.assertFalse(access('describe',registry,interpreters,PROFILES)['ready'])
            interpreters['analysis']=str(python);python.unlink()
            self.assertFalse(access('describe',registry,interpreters,PROFILES)['ready'])

    def test_original_preprocessing_and_animation_routes_remain_usable(self):
        with tempfile.TemporaryDirectory() as tmp:
            core=Path(tmp)/'core.exe';core.touch();pose=Path(tmp)/'pose.exe';pose.touch()
            interpreters={'analysis':str(core),'preprocess':str(pose),'pose':str(pose)}
            registry={'active':{'core':'c','pose':'p'},'environments':[
                {'id':'c','profile':'core','python':str(core),'validation':report('core',versions={})},
                {'id':'p','profile':'pose','python':str(pose),'validation':report('pose',versions={'matplotlib':'3.10.1'})}]}
            self.assertEqual(access('preprocess',registry,interpreters,PROFILES)['profile'],'pose')
            self.assertEqual(access('filter',registry,interpreters,PROFILES)['role'],'pose')
            del registry['active']['pose']
            self.assertFalse(access('preprocess',registry,interpreters,PROFILES)['ready'])
            blocked=access('filter',registry,interpreters,PROFILES)
            self.assertFalse(blocked['ready'])
            self.assertEqual(blocked['reason'],'missing_animation_validation')
            self.assertIn('is active',blocked['message'])

    def test_rejected_jobs_do_not_enter_queue_or_start_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager=EnvironmentManager(Path(tmp)/'home')
            with patch.object(server,'ENVIRONMENTS',manager),patch.object(server,'RUNTIME',Path(tmp)/'tasks'),patch.object(server,'run_job') as worker:
                before=set(server.JOBS)
                for op in ('pose','triangulate','preprocess','describe','embed','analysis:major','chat'):
                    with self.assertRaisesRegex(ValueError,'No suitable environment'):server.start_job({'op':op})
                self.assertEqual(set(server.JOBS),before)
                self.assertFalse((Path(tmp)/'tasks').exists());worker.assert_not_called()

    def test_activation_unlocks_existing_configured_interpreter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);python=root/'python.exe';python.touch();manager=EnvironmentManager(root/'home')
            atomic_json(manager.registry,{'environments':[{'id':'c','profile':'core','python':str(python),'validation':report('core')}],'active':{}})
            cfg={'interpreters':{'analysis':str(python),'preprocess':str(python),'projection':str(python)}}
            with patch.object(server,'ENVIRONMENTS',manager),patch.object(server,'read_settings',return_value=cfg):
                self.assertFalse(server.environment_access('describe')['ready'])
                manager.activate('c')
                self.assertTrue(server.environment_access('describe')['ready'])


if __name__=='__main__':unittest.main()
