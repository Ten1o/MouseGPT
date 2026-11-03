import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from workflow_inputs import triangulation_inputs


class TriangulationInputs(unittest.TestCase):
    def fixture(self,root):
        source=root/'predictions';source.mkdir();calibration=root/'calibration';calibration.mkdir()
        for i in range(1,9):
            (source/f'data{i}.json').write_text('[]')
            (calibration/f'camera{i}.mat').touch()
        return {'keypoints':str(source),'calibration':str(calibration)},source

    def test_default_uses_json_directory_ignoring_other_stage_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            f,source=self.fixture(Path(tmp));f['out1']=str(Path(tmp)/'unrelated-stage-output')
            self.assertEqual(triangulation_inputs(f)[2],source.resolve())

    def test_custom_output_does_not_require_out1_or_create_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            f,source=self.fixture(Path(tmp));dest=Path(tmp)/'new-result'
            f.update(triOutputMode='custom',triOutput=str(dest))
            self.assertEqual(triangulation_inputs(f)[2],dest.resolve());self.assertFalse(dest.exists())

    def test_missing_inputs_and_existing_result_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'2D JSON'):triangulation_inputs({})
        with tempfile.TemporaryDirectory() as tmp:
            f,source=self.fixture(Path(tmp));(source/'tri_new.npy').touch()
            with self.assertRaisesRegex(ValueError,'already exists'):triangulation_inputs(f)
            (source/'tri_new.npy').unlink();(source/'data8.json').unlink()
            with self.assertRaisesRegex(ValueError,'data8.json'):triangulation_inputs(f)

    def test_blank_custom_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            f,_=self.fixture(Path(tmp));f['triOutputMode']='custom'
            with self.assertRaisesRegex(ValueError,'custom reconstruction'):triangulation_inputs(f)

    def test_worker_passes_separate_input_and_output_without_copying_json(self):
        import worker
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);f,source=self.fixture(root);dest=root/'results'
            f.update(triOutputMode='custom',triOutput=str(dest))
            def reconstruct(command,**kwargs):
                self.assertEqual(command[command.index('--demo_folder')+1],str(source.resolve()))
                target=Path(command[command.index('--output_path')+1])
                self.assertEqual(target,dest.resolve()/'tri_new.npy');target.write_bytes(b'test-output')
            with patch.object(worker.subprocess,'run',side_effect=reconstruct):result=worker.run('triangulate',f,{})
            self.assertEqual(result['files'],[str(dest.resolve()/'tri_new.npy')])
            self.assertFalse(list(dest.glob('data*.json')))
            self.assertEqual(len(list(source.glob('*.json'))),8)

    def test_invalid_request_does_not_queue_a_task(self):
        import server
        count=len(server.JOBS)
        with patch.object(server,'require_environment'),self.assertRaisesRegex(ValueError,'2D JSON'):
            server.start_job({'op':'triangulate','fields':{}})
        self.assertEqual(len(server.JOBS),count)


if __name__=='__main__':unittest.main()
