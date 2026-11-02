"""Small real-video and geometric checks, without models or user data."""
from pathlib import Path
import json
import sys
import subprocess
import os
import tempfile
import unittest

import cv2
import numpy as np
from scipy.io import savemat

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'demo'))
from project_3d_keypoints import KeypointProjector,render_reprojection


class ReprojectionTests(unittest.TestCase):
    def setup_recording(self,root):
        calibration=root/'calibration';calibration.mkdir()
        videos=root/'videos';videos.mkdir()
        intrinsic=np.array([[30.,0,32],[0,30,24],[0,0,1]])
        radial=np.array([.1,-.02]);tangent=np.array([.001,-.002])
        for camera in range(1,9):
            savemat(calibration/f'Camera{camera}_params.mat',{'K':intrinsic.T,'r':np.eye(3),'t':np.array([[camera*2.,0,0]]),'RDistort':radial,'TDistort':tangent})
            writer=cv2.VideoWriter(str(videos/f'data{camera}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(64,48))
            self.assertTrue(writer.isOpened())
            for i in range(3):writer.write(np.full((48,64,3),50+i*10,dtype=np.uint8))
            writer.release()
        points=np.zeros((3,24,3));points[:,:,0]=np.linspace(-.1,.1,24);points[:,:,1]=.05;points[:,:,2]=1.
        np.save(root/'tri_filter.npy',points)
        return calibration,videos,points,intrinsic,radial,tangent

    def test_forward_distortion_matches_opencv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);calibration,_,points,k,radial,tangent=self.setup_recording(root)
            projected=KeypointProjector(calibration).project_3d_to_2d(points[0]*1000,1)
            expected,_=cv2.projectPoints(points[0]*1000,np.zeros(3),np.array([2.,0,0]),k,np.r_[radial,tangent,0.])
            np.testing.assert_allclose(projected,expected.reshape(-1,2),rtol=1e-5,atol=1e-5)

    def test_eight_full_length_videos_json_units_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);calibration,videos,points,*_=self.setup_recording(root)
            manifest=root/'result.json'
            subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'demo/project_3d_keypoints.py'),'--keypoints_3d',str(root/'tri_filter.npy'),'--calib_folder',str(calibration),'--videos',str(videos),'--output_folder',str(root/'output'),'--input_unit','m','--calibration_unit','mm','--result',str(manifest)],check=True,capture_output=True,text=True,encoding='utf-8',env={**os.environ,'PYTHONIOENCODING':'utf-8'})
            result=json.loads(manifest.read_text())
            self.assertEqual(result['frames'],3);self.assertEqual(len(result['videos']),8)
            for camera in range(1,9):
                records=json.loads((root/f'output/data{camera}.json').read_text())
                self.assertEqual([r['frame_id'] for r in records],[0,1,2])
                projected=KeypointProjector(calibration).project_3d_to_2d(points[0]*1000,camera)
                np.testing.assert_allclose(records[0]['instances'][0]['keypoints'],projected)
                cap=cv2.VideoCapture(str(root/f'output/data{camera}.mp4'))
                try:
                    self.assertTrue(cap.isOpened());self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),3)
                    self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS),20,places=2);self.assertTrue(cap.read()[0])
                finally:cap.release()
            before=(root/'output/data1.json').read_bytes()
            with self.assertRaises(FileExistsError):render_reprojection(root/'tri_filter.npy',calibration,videos,root/'output','m','mm')
            self.assertEqual((root/'output/data1.json').read_bytes(),before)
            np.save(root/'short.npy',points[:2])
            with self.assertRaisesRegex(ValueError,'video frames'):render_reprojection(root/'short.npy',calibration,videos,root/'bad-output','m','mm')
            self.assertFalse((root/'bad-output').exists())


if __name__=='__main__':unittest.main()
