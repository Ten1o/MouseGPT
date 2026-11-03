"""Offline smoke checks executed inside a candidate environment before activation."""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parent))
from environment_validation import application_fingerprint


def check_repository_agent(root):
    """Build the actual application against a temporary session, without networking."""
    import os
    import subprocess
    from PIL import Image

    project = Path(__file__).resolve().parents[1]
    previews = root / "sample_imgs"
    previews.mkdir()
    for frame in (0, 12, 24, 36):
        Image.new("RGB", (64, 32), "gray").save(previews / f"{frame:08d}.jpg")
    env = {k:v for k,v in os.environ.items()
           if not k.startswith(("MOUSEGPT_", "OPENAI_", "AZURE_OPENAI_", "LANGCHAIN_", "LANGSMITH_"))}
    env.update(MOUSEGPT_API_KEY="offline-check",
               MOUSEGPT_NLUI_H5=str(root / "sample.h5"),
               MOUSEGPT_NLUI_IMAGES=str(previews),
               MOUSEGPT_NLUI_WORKSPACE=str(root / "workspace"),
               GRADIO_ANALYTICS_ENABLED="False", LANGCHAIN_TRACING_V2="false",
               LANGSMITH_TRACING="false", PYTHONNOUSERSITE="1")
    script = "\n".join([
        "import runpy, socket, sys",
        "def deny_network(*args, **kwargs): raise RuntimeError('Offline environment validation must not access the network.')",
        "socket.socket.connect = deny_network",
        "socket.create_connection = deny_network",
        "sys.path[:0] = " + repr([str(project / "4_NLUI"), str(project)]),
        "import mousegpt_config",
        "mousegpt_config.read_env_file = lambda *args, **kwargs: {}",
        "sys.argv = " + repr([str(project / "4_NLUI/agent.py"), "--check"]),
        "runpy.run_path(sys.argv[0], run_name='__main__')",
    ])
    result = subprocess.run([sys.executable, "-I", "-c", script], env=env,
                            cwd=root, capture_output=True, text=True, timeout=90)
    if result.returncode or "Interface ready:" not in result.stdout:
        raise RuntimeError("Repository Agent startup failed: " + (result.stderr or result.stdout)[-3000:])



def probe(profile):
    fingerprint=application_fingerprint(profile)
    if profile=='pose':
        import os
        os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE','1')
        import numpy as np
        import cv2
        import torch
        from scipy.signal import savgol_filter
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'1_keypoints'))
        from mmpose.apis.inferencers import MMPoseInferencer
        import mmdet
        import pandas
        from PIL import Image
        import matplotlib
        from easymocap.mytools.triangulator import iterative_triangulate
        assert savgol_filter(np.arange(16.),15,3).shape==(16,)
        return {'passed':True,'application_fingerprint':fingerprint,'profile':'pose','python':platform.python_version(),'executable':sys.executable,'platform':platform.platform(),
                'checks':['OpenCV','MMPose inferencer import','MMDetection import','EasyMocap triangulation import','SciPy filtering'],
                'versions':{'numpy':np.__version__,'torch':torch.__version__,'opencv':cv2.__version__,'mmdet':mmdet.__version__,'matplotlib':matplotlib.__version__}}
    import numpy as np
    import pandas as pd
    from scipy.signal import savgol_filter
    from sklearn.ensemble import IsolationForest
    from PIL import Image
    from openai import OpenAI
    checks=[]
    with tempfile.TemporaryDirectory(prefix='mousegpt-env-check-') as tmp:
        root=Path(tmp);ids=['00000000','00000012','00000024','00000036']
        rng=np.random.default_rng(7);vectors=rng.normal(size=(4,3072))
        with pd.HDFStore(root/'sample.h5','w') as store:
            store.put('descriptions',pd.DataFrame({'id':ids,'description':['moving']*4}))
            store.put('features_des',pd.DataFrame(vectors,index=ids))
        with pd.HDFStore(root/'sample.h5','r') as store:
            assert store['features_des'].index.tolist()==ids
            assert np.allclose(store['features_des'].to_numpy(),vectors)
        checks.append('HDF5 frame/vector round-trip')
        assert np.isfinite(IsolationForest(random_state=42,n_estimators=10).fit(vectors).decision_function(vectors)).all()
        checks.append('Anomaly ranking')
        assert savgol_filter(np.arange(24.),15,3).shape==(24,)
        checks.append('Trajectory filtering')
        canvas=Image.new('RGB',(64,32));canvas.save(root/'paired.jpg')
        with Image.open(root/'paired.jpg') as im:assert im.size==(64,32)
        checks.append('Paired image encoding')
        if profile=='core':
            import cv2
            import umap
            import imageio_ffmpeg
            import subprocess
            assert cv2.imread(str(root/'paired.jpg')).shape==(32,64,3)
            projection=umap.UMAP(n_neighbors=2,random_state=42,n_epochs=10,init='random').fit_transform(rng.normal(size=(12,8)))
            assert projection.shape==(12,2) and np.isfinite(projection).all()
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-version'],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,check=True,timeout=20)
            import importlib.util
            demo=Path(__file__).resolve().parents[1]/'1_keypoints/demo'
            sys.path.insert(0,str(demo))
            spec=importlib.util.spec_from_file_location('keypoints_batch_vis',demo/'batch_vis.py')
            batch=importlib.util.module_from_spec(spec);spec.loader.exec_module(batch)
            coordinates=rng.normal(size=(16,24,3));coordinates[5,0]=np.nan
            np.save(root/'tri_new.npy',coordinates)
            preview=batch.process_file(root/'tri_new.npy',root/'filtered','mm',frames=2,fps=30)
            video=cv2.VideoCapture(preview['files'][1])
            try:assert video.isOpened() and int(video.get(cv2.CAP_PROP_FRAME_COUNT))==2
            finally:video.release()
            checks.append('Project batch_vis filtering and 3D MP4')
            assert hasattr(OpenAI(api_key='offline-check'),'responses')
            pd.DataFrame({'count':[1]}).to_excel(root/'counts.xlsx',index=False)
            checks.extend(['OpenCV image reading','UMAP projection','FFmpeg executable','Responses SDK interface','Excel export'])
        elif profile=='agent':
            check_repository_agent(root)
            checks.append('Repository Agent --check with external HDF5/images and networking disabled')
        else:raise ValueError('Unsupported profile.')
    versions={name:metadata.version(name) for name in ['numpy','pandas','tables','scipy','scikit-learn','Pillow','openai']}
    if profile=='core':versions['matplotlib']=metadata.version('matplotlib')
    if profile=='agent':
        for name in ('gradio','langchain-classic','langchain-openai','langchain-experimental'):
            versions[name]=metadata.version(name)
    return {'passed':True,'application_fingerprint':fingerprint,'profile':profile,'python':platform.python_version(),'executable':sys.executable,
            'platform':platform.platform(),'checks':checks,'versions':versions}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('profile',choices=['core','agent','pose']);p.add_argument('--output',required=True);args=p.parse_args()
    result=probe(args.profile)
    Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Validation passed: '+', '.join(result['checks']),flush=True)
