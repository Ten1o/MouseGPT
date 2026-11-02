"""Offline smoke checks executed inside a candidate environment before activation."""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import sys
import tempfile


def probe(profile):
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
        return {'passed':True,'profile':'pose','python':platform.python_version(),'executable':sys.executable,'platform':platform.platform(),
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
            import gradio as gr
            import langchain_openai
            from langchain.agents import create_tool_calling_agent,AgentExecutor
            from langchain_experimental.tools import PythonREPLTool
            from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
            model=langchain_openai.ChatOpenAI(model='offline-model',api_key='offline-check')
            prompt=ChatPromptTemplate.from_messages([('system','Offline validation.'),('human','{input}'),MessagesPlaceholder('agent_scratchpad')])
            agent=create_tool_calling_agent(model,[PythonREPLTool()],prompt)
            AgentExecutor(agent=agent,tools=[PythonREPLTool()])
            with gr.Blocks() as demo:
                gr.Chatbot(type='messages');gr.Textbox()
            checks.extend(['Gradio message UI construction','Tool-calling Agent construction'])
        else:raise ValueError('Unsupported profile.')
    versions={name:metadata.version(name) for name in ['numpy','pandas','tables','scipy','scikit-learn','Pillow','openai']}
    if profile=='core':versions['matplotlib']=metadata.version('matplotlib')
    return {'passed':True,'profile':profile,'python':platform.python_version(),'executable':sys.executable,
            'platform':platform.platform(),'checks':checks,'versions':versions}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('profile',choices=['core','agent','pose']);p.add_argument('--output',required=True);args=p.parse_args()
    result=probe(args.profile)
    Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Validation passed: '+', '.join(result['checks']),flush=True)
