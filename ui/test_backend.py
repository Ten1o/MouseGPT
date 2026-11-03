"""Local integration checks. No live provider requests or source-data modifications."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import sys
from unittest.mock import patch

UI=Path(__file__).resolve().parent
sys.path.insert(0,str(UI))
import server


class BackendTests(unittest.TestCase):
    def test_filter_ignores_legacy_preview_settings(self):
        import worker
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'tri.npy';source.touch()
            def project_command(command,**kwargs):
                self.assertEqual(command[command.index('--start')+1],'0')
                self.assertEqual(command[command.index('--frames')+1],'0')
                self.assertNotIn('--fps',command)
                Path(command[command.index('--result')+1]).write_text(json.dumps({'files':[],'preview_frames':24}))
            with patch.object(worker.subprocess,'run',side_effect=project_command):
                result=worker.run('filter',{'rawTri':str(source),'units':'m','filterStart':2,'filterFrames':300,'filterFps':60},{})
            self.assertEqual(result['preview_frames'],24)

    def test_unactivated_interpreter_is_not_used(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(server.ENVIRONMENTS,'read_registry',return_value={'environments':[],'active':{}}),patch.object(server.subprocess,'run') as run:
            with self.assertRaisesRegex(ValueError,'No suitable environment'):
                server.worker_command('filter',{}, {},Path(tmp))
            run.assert_not_called()

    def test_ui_port_cannot_be_bound_twice(self):
        with server.LocalUIServer(('127.0.0.1',0),server.Handler) as first:
            with self.assertRaises(OSError):
                with server.LocalUIServer(first.server_address,server.Handler):pass

    def test_api_config_exports_omit_keys_by_default_and_round_trip(self):
        env={f'MOUSEGPT_{role}_{key}':value for role in server.ROLES for key,value in {
            'PROVIDER':'openai','ENDPOINT':'https://api.openai.com/v1','MODEL':server.DEFAULT_MODELS[role],'API_KEY':'fixture-private-key'}.items()}
        with tempfile.TemporaryDirectory() as tmp,patch.object(server,'SETTINGS',Path(tmp)/'settings.json'),patch.object(server,'environment',return_value=env):
            safe=server.export_api_settings()
            self.assertFalse(safe['contains_api_keys']);self.assertNotIn('fixture-private-key',json.dumps(safe))
            full=server.export_api_settings(True)
            self.assertTrue(full['contains_api_keys'])
            self.assertEqual(full['roles']['VISION']['api_key'],'fixture-private-key')
            server.save_settings({'roles':full['roles']})
            self.assertEqual(server.read_settings()['roles']['VISION']['api_key'],'fixture-private-key')

    def test_shared_api_inheritance_and_override_reset(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(server,'SETTINGS',Path(tmp)/'settings.json'):
            shared={'provider':'openai','endpoint':'https://api.openai.com/v1','model':'gpt-4o','api_key':'shared-fixture-secret'}
            server.save_settings({'shared':shared,'roles':{'ANALYSIS':{'model':'custom-analysis'}}})
            env=server.environment()
            self.assertEqual(server.load_config('VISION',env=env).api_key,'shared-fixture-secret')
            self.assertEqual(server.load_config('ANALYSIS',env=env).model,'custom-analysis')
            self.assertEqual(server.load_config('EMBEDDING',env=env).model,'text-embedding-3-large')
            self.assertNotIn('shared-fixture-secret',json.dumps(server.public_settings()))
            safe=server.export_api_settings()
            self.assertEqual(safe['schemaVersion'],2)
            self.assertNotIn('shared-fixture-secret',json.dumps(safe))
            self.assertEqual(server.export_api_settings(True)['shared']['api_key'],'shared-fixture-secret')
            server.save_settings({'inherit':['ANALYSIS']})
            self.assertFalse(server.public_settings()['overrides']['ANALYSIS'])
            self.assertEqual(server.load_config('ANALYSIS',env=server.environment()).model,'gpt-4o')

    def test_shared_api_saves_without_requiring_unused_phenotype_support(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(server,'SETTINGS',Path(tmp)/'settings.json'):
            result=server.save_settings({'shared':{'provider':'openai_compatible','endpoint':'https://example.com/v1','model':'chat-model','api_key':'fixture-shared-secret'}})
            self.assertEqual(result['roles']['VISION']['provider'],'openai_compatible')
            self.assertEqual(result['roles']['PHENOTYPE']['provider'],'openai_compatible')
            self.assertIn('Responses API',result['roles']['PHENOTYPE']['error'])
            self.assertNotIn('fixture-shared-secret',json.dumps(result))
            env=server.environment()
            self.assertEqual(server.load_config('VISION',env=env).model,'chat-model')
            with self.assertRaisesRegex(ValueError,'Responses API'):server.load_config('PHENOTYPE',env=env)
            self.assertEqual(server.export_api_settings()['schemaVersion'],2)

    def test_preview_frame_list_uses_existing_sparse_jpgs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ('00000816.jpg','00000000.jpg','00000012.jpg','notes.jpg','00000001.png','12.jpg'):(root/name).touch()
            (root/'00000100.jpg').mkdir()
            self.assertEqual(server.preview_frames({'previews':tmp})['frame_ids'],[0,12,816])
            with self.assertRaisesRegex(ValueError,'not found'):server.preview_frames({'previews':str(root/'missing')})

    def test_backend_identity_is_stable_and_machine_scoped(self):
        from local_paths import backend_id
        with tempfile.TemporaryDirectory() as tmp:
            first=backend_id(Path(tmp)/'one',machine='machine-a')
            self.assertEqual(first,backend_id(Path(tmp)/'one',machine='machine-a'))
            self.assertNotEqual(first,backend_id(Path(tmp)/'one',machine='machine-b'))
            self.assertNotEqual(first,backend_id(Path(tmp)/'two',machine='machine-a'))
            self.assertEqual(len(first),64)

    def test_reprojection_validates_paths_units_and_default_output(self):
        from workflow_inputs import reprojection_inputs
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'tri_filter.npy';source.touch();videos=root/'videos';videos.mkdir();calibration=root/'calibration';calibration.mkdir()
            for i in range(1,9):(videos/f'data{i}.mp4').touch();(calibration/f'Camera{i}_params.mat').touch()
            fields={'tri':str(source),'videos':str(videos),'calibration':str(calibration),'calibrationUnit':'mm'}
            args=reprojection_inputs(fields)
            self.assertEqual(args[3],root/'reprojection');self.assertEqual(args[-2:],('m','mm'))
            self.assertFalse(args[3].exists())
            with self.assertRaisesRegex(ValueError,'unit'):reprojection_inputs({**fields,'calibrationUnit':''})
            (root/'reprojection').mkdir();(root/'reprojection/data1.mp4').touch()
            with self.assertRaisesRegex(ValueError,'already exist'):reprojection_inputs(fields)

    def test_mac_app_storage_and_legacy_preservation(self):
        import local_paths
        import os
        with tempfile.TemporaryDirectory() as tmp,patch.object(local_paths.sys,'platform','darwin'),patch.object(local_paths.Path,'home',return_value=Path(tmp)),patch.dict(os.environ,{},clear=True):
            self.assertEqual(local_paths.app_home(),Path(tmp)/'Library/Application Support/MouseGPT')
            legacy=Path(tmp)/'.local/share/MouseGPT';legacy.mkdir(parents=True)
            self.assertEqual(local_paths.app_home(),legacy)

    def test_external_session_paths(self):
        import importlib.util
        import os
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);h5=folder/'external-session.h5';h5.touch()
            images=folder/'arbitrary-image-folder';images.mkdir();(images/'00000012.jpg').touch()
            env={'MOUSEGPT_NLUI_H5':str(h5),'MOUSEGPT_NLUI_IMAGES':str(images),'MOUSEGPT_NLUI_WORKSPACE':str(folder/'results')}
            with patch.dict(os.environ,env):
                spec=importlib.util.spec_from_file_location('ui_external_runtime',UI.parent/'4_NLUI/runtime_config.py')
                runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
                self.assertEqual(runtime.available_sessions(),['external-session'])
                cfg=runtime.session_config('external-session')
                self.assertEqual(cfg['img_folder'],str(images.resolve()))
                self.assertEqual(cfg['file_path'],str(h5.resolve()))

    def test_settings_masks_existing_keys(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(server,'SETTINGS',Path(tmp)/'settings.json'):
            server.save_settings({'roles':{'VISION':{'provider':'openai_compatible','endpoint':'http://localhost:9999/v1','model':'test','api_key':'secret-test-key'}}})
            result=server.public_settings()
            self.assertNotIn('secret-test-key',json.dumps(result))
            self.assertTrue(result['roles']['VISION']['api_key_configured'])
            server.save_settings({'roles':{'VISION':{'api_key':''}}})
            self.assertEqual(server.read_settings()['roles']['VISION']['api_key'],'secret-test-key')



if __name__=='__main__':unittest.main()
