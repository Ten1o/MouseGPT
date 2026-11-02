"""Project-level checks for NPY filtering, animation and reusable pose videos."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from scipy.signal import savgol_filter

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'demo'))
import batch_vis
from video_utils import pose_videos


class BatchVisTests(unittest.TestCase):
    def test_filter_fills_and_converts_using_original_batch_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'tri_new.npy'
            a=np.random.default_rng(42).normal(size=(24,24,3));a[6,2]=np.nan
            np.save(source,a)
            filled=a.copy();filled[6,2]=(a[5,2]+a[7,2])/2
            result,count=batch_vis.prepare_coordinates(source,'mm')
            np.testing.assert_allclose(result,savgol_filter(filled,15,5,axis=0)*.001)
            self.assertEqual(count,1)
            np.testing.assert_equal(np.load(source),a)
            a[0,0]=0;np.save(source,a)
            with self.assertRaisesRegex(ValueError,'missing endpoint'):batch_vis.prepare_coordinates(source,'m')

    def test_animation_and_filter_outputs_are_paired(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'tri_new.npy'
            a=np.random.default_rng(7).normal(size=(20,24,3));np.save(source,a)
            result=batch_vis.process_file(source,root/'out','cm',start=3,frames=2,fps=10)
            self.assertEqual({p.name for p in (root/'out').iterdir()},{'tri_filter.npy','tri_filter.mp4'})
            self.assertEqual(result['viewer'],'3d-video');self.assertEqual(result['videos'],[{'index':1}])
            self.assertEqual(result['frames'],20);self.assertEqual(result['preview_start'],3)
            np.testing.assert_allclose(np.load(result['files'][0]),savgol_filter(a,15,5,axis=0)*.01)
            cap=cv2.VideoCapture(result['files'][1])
            try:
                self.assertTrue(cap.isOpened());self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),2)
                ok,image=cap.read();self.assertTrue(ok);self.assertGreater(float(image.std()),1)
            finally:cap.release()
            with self.assertRaisesRegex(ValueError,'already exists'):batch_vis.process_file(source,root/'out')

    def test_failed_render_does_not_commit_filter_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'tri_new.npy';np.save(source,np.ones((16,24,3)))
            with patch.object(batch_vis,'render_animation',side_effect=RuntimeError('encoder failed')):
                with self.assertRaisesRegex(RuntimeError,'encoder failed'):batch_vis.process_file(source)
            self.assertFalse((Path(tmp)/'tri_filter.npy').exists())
            self.assertFalse((Path(tmp)/'tri_filter.mp4').exists())

    def test_video_reservation_never_removes_an_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'tri_new.npy';np.save(source,np.ones((16,24,3)))
            video=Path(tmp)/'tri_filter.mp4';video.write_bytes(b'existing video')
            with self.assertRaisesRegex(ValueError,'already exists'):batch_vis.process_file(source)
            self.assertEqual(video.read_bytes(),b'existing video')
            self.assertFalse((Path(tmp)/'tri_filter.npy').exists())

    def test_default_animation_includes_every_frame_and_uses_video_fps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'session'/'keypoints'/'tri_new.npy';source.parent.mkdir(parents=True)
            videos=source.parent.parent/'videos';videos.mkdir()
            writer=cv2.VideoWriter(str(videos/'data1.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),19.98,(64,64))
            for _ in range(16):writer.write(np.zeros((64,64,3),np.uint8))
            writer.release()
            np.save(source,np.random.default_rng(9).normal(size=(16,24,3)))
            result=batch_vis.process_file(source)
            self.assertEqual(result['preview_start'],0);self.assertEqual(result['preview_frames'],16)
            self.assertAlmostEqual(result['fps'],19.98,places=2)
            cap=cv2.VideoCapture(result['files'][1])
            try:
                self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),16)
                self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS),19.98,places=2)
            finally:cap.release()

    def test_existing_pose_videos_are_not_reinferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for view in range(1,9):
                writer=cv2.VideoWriter(str(root/f'data{view}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(64,64))
                for _ in range(2):writer.write(np.zeros((64,64,3),np.uint8))
                writer.release()
            result=pose_videos(root,root/'browser')
            self.assertEqual(len(result['files']),8)
            self.assertEqual([v['camera'] for v in result['videos']],list(range(1,9)))


if __name__=='__main__':unittest.main()
