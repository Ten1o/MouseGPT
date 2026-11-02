"""Offline interface compatibility checks without importing GPU libraries."""
import ast
import inspect
from pathlib import Path
import unittest


SOURCE=Path(__file__).resolve().parents[1]/'1_keypoints/mmpose/apis/inferencers/pose2d_inferencer.py'
tree=ast.parse(SOURCE.read_text(encoding='utf-8'))
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_detector_datasample_keyword')
namespace={'inspect':inspect}
exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),namespace)
keyword=namespace['_detector_datasample_keyword']


class DetectorInterface(unittest.TestCase):
    def test_current_and_legacy_interfaces(self):
        class Current:
            def __call__(self,inputs,return_datasamples=False):return return_datasamples
        class Legacy:
            def __call__(self,inputs,return_datasample=False):return return_datasample
        for detector in (Current(),Legacy()):self.assertTrue(detector('frame',**{keyword(detector):True}))

    def test_generic_callable_uses_declared_postprocess_options(self):
        class Detector:
            postprocess_kwargs={'return_datasamples'}
            def __call__(self,inputs,**kwargs):return kwargs
        self.assertEqual(keyword(Detector()),'return_datasamples')

    def test_unknown_interface_is_not_silently_downgraded(self):
        class Detector:
            def __call__(self,inputs):pass
        with self.assertRaisesRegex(RuntimeError,'supported'):keyword(Detector())


if __name__=='__main__':unittest.main()
