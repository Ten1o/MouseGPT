"""Allowlisted local workflow operations; invoked by server.py in a stage environment."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'3_applications'))


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def path(fields, name, exists=False):
    value = str(fields.get(name, '')).strip()
    if not value:
        raise ValueError(f'Missing path: {name}')
    result = Path(value).expanduser().resolve()
    if exists and not result.exists():
        raise ValueError(f'Path not found: {result}')
    return result


def output(folder):
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def write_json(target, value):
    output(target.parent)
    temporary = target.with_suffix(target.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(target)


def no_overwrite(target):
    if target.exists():
        raise ValueError(f'Output already exists. Choose a new output path: {target}')


def aligned_session(source):
    import numpy as np
    import pandas as pd
    with pd.HDFStore(source, mode='r') as store:
        descriptions = store['descriptions']
        features = store['features_des']
    ids = descriptions['id'].map(str).tolist()
    if len(ids) != len(set(ids)) or ids != features.index.map(str).tolist():
        raise ValueError('Description and embedding frame IDs are not aligned.')
    if not np.isfinite(features.to_numpy()).all() or (np.linalg.norm(features.to_numpy(), axis=1) == 0).any():
        raise ValueError('Embeddings must be finite and nonzero.')
    return descriptions, features


def run(op, f, tabs):
    import numpy as np
    if op=='pose-previews':
        pose_videos=module('keypoint_video_utils','1_keypoints/demo/video_utils.py').pose_videos
        if not f.get('posePreviewDir') and f.get('out1'):f={**f,'posePreviewDir':str(path(f,'out1')/'preview')}
        source=path(f,'posePreviewDir',True)
        return pose_videos(source,source/('browser-'+str(f.get('_jobId','preview'))))
    if op=='preview-image':
        from io import BytesIO
        import base64
        from PIL import Image
        frame=str(int(f.get('frame',0))).zfill(8)
        sources=[path(f,k,True)/f'{frame}.jpg' for k in ('crop3','crop5') if f.get(k)]
        if not sources and f.get('previews'):sources=[path(f,'previews',True)/f'{frame}.jpg']
        if not sources or not all(p.is_file() for p in sources):raise ValueError(f'No images for frame {frame}.')
        images=[]
        for source in sources:
            with Image.open(source) as im:images.append(im.convert('RGB'))
        canvas=Image.new('RGB',(sum(im.width for im in images),max(im.height for im in images)))
        x=0
        for im in images:canvas.paste(im,(x,0));x+=im.width
        canvas.thumbnail((1600,1000));stream=BytesIO();canvas.save(stream,format='JPEG',quality=90)
        return {'image':'data:image/jpeg;base64,'+base64.b64encode(stream.getvalue()).decode()}
    if op == 'api-test':
        from mousegpt_config import load_config,sdk_client
        role=f.get('role','VISION');cfg=load_config(role);client=sdk_client(cfg,timeout=30)
        if role=='EMBEDDING':
            response=client.embeddings.create(model=cfg.model,input='connection test')
            return {'message':f'Embedding API connected ({len(response.data[0].embedding)} dimensions).'}
        if role=='PHENOTYPE':return {'message':client.responses.create(model=cfg.model,input='Reply OK.').output_text}
        return {'message':client.chat.completions.create(model=cfg.model,messages=[{'role':'user','content':'Reply OK.'}]).choices[0].message.content}
    if op == 'chat':
        from mousegpt_config import load_config,sdk_client
        desc,features=aligned_session(path(f,'h5',True));frame=str(f.get('frame',0)).zfill(8)
        lookup={str(i).zfill(8):d for i,d in zip(desc.id,desc.description)}
        if frame not in lookup:raise ValueError('Frame ID is not in the selected session.')
        question=str(f.get('question','')).strip()
        if not question:raise ValueError('Enter a question.')
        sys.path.insert(0,str(ROOT/'4_NLUI'))
        from tools_function.search import search
        found,scores=search(str(path(f,'h5',True)),question,load_config('EMBEDDING').api_key)
        relevant=[{'frame_id':str(i).zfill(8),'description':lookup[str(i).zfill(8)],'similarity':float(s)} for i,s in zip(found[:5],scores[:5])]
        cfg=load_config('AGENT');client=sdk_client(cfg,timeout=90)
        response=client.chat.completions.create(model=cfg.model,messages=[{'role':'system','content':'Answer using only the supplied mouse descriptions. Cite frame IDs. Do not invent counts, durations, or tool results.'},{'role':'user','content':json.dumps({'question':question,'frame':frame,'description':lookup[frame],'semantic_matches':relevant},ensure_ascii=False)}])
        result={'answer':response.choices[0].message.content,'retrieved_frames':[r['frame_id'] for r in relevant]}
        if f.get('out4'):
            target=output(path(f,'out4'))/('conversation-'+str(f['_jobId'])+'.json');no_overwrite(target)
            write_json(target,{'question':question,**result});result['files']=[str(target)]
        return result
    if op == 'filter':
        source=path(f,'rawTri',True)
        dest=path(f,'filterOutput') if str(f.get('filterOutput','')).strip() else source.parent
        if f.get('units') not in {'m','cm','mm'}:raise ValueError('Select the source calibration unit.')
        with tempfile.TemporaryDirectory(prefix='mousegpt-filter-') as tmp:
            result=Path(tmp)/'result.json'
            command=[sys.executable,str(ROOT/'1_keypoints/demo/batch_vis.py'),'--input',str(source),
                     '--output-dir',str(dest),'--unit',f['units'],'--start','0','--frames','0','--result',str(result)]
            if f.get('animationFps'):command+=['--fps',str(f['animationFps'])]
            if f.get('animationVideo'):command+=['--reference-video',str(path(f,'animationVideo',True))]
            elif f.get('videos') and (path(f,'videos')/'data1.mp4').is_file():
                command+=['--reference-video',str(path(f,'videos')/'data1.mp4')]
            print('Running 1_keypoints/demo/batch_vis.py: filter and 3D animation',flush=True)
            print(f'Python environment: {sys.executable}',flush=True)
            subprocess.run(command,check=True)
            metadata=json.loads(result.read_text(encoding='utf-8'));metadata['interpreter']=sys.executable
            return metadata

    if op == 'convert':
        import pandas as pd
        source = path(f, 'external2d', True)
        mapping = json.loads(path(f, 'mapping', True).read_text(encoding='utf-8'))
        names = mapping.get('keypoints', [])
        cameras = mapping.get('cameras', {})
        if len(names) != 24 or len(set(names)) != 24 or set(cameras) != {f'data{i}' for i in range(1, 9)}:
            raise ValueError('Mapping JSON requires 24 unique keypoints and cameras data1…data8 mapped to filenames.')
        dest = output(path(f, 'out1'))
        pending = {}
        for camera, filename in cameras.items():
            file = source / filename if source.is_dir() else source
            if 'DLC' in f.get('format', ''):
                data = pd.read_csv(file, header=[0, 1, 2], index_col=0) if file.suffix.lower() == '.csv' else pd.read_hdf(file)
                if data.columns.nlevels != 3:
                    raise ValueError('This adapter supports single-animal DLC with scorer/bodyparts/coords columns.')
                data.columns = data.columns.droplevel(0)
                if list(data.index) != list(range(len(data))):
                    raise ValueError('DLC frame index must start at 0 with no missing rows.')
                records = []
                for _, row in data.iterrows():
                    coords = [[float(row[(name, 'x')]), float(row[(name, 'y')])] for name in names]
                    scores = [float(row[(name, 'likelihood')]) for name in names]
                    if not np.isfinite(coords).all() or not np.isfinite(scores).all():
                        raise ValueError('Nonfinite DLC points: resolve missing points before conversion.')
                    xy = np.array(coords)
                    records.append({'instances': [{'keypoints': coords, 'keypoint_scores': scores,
                        'bbox': [[*xy.min(axis=0).tolist(), *xy.max(axis=0).tolist()]]}]})
            elif 'SLEAP' in f.get('format', ''):
                adapter = module('sleap_conversion', '1_keypoints/demo/convert_sleap.py')
                records = adapter.read_analysis(file, names, mapping.get('tracks', {}).get(camera))
            elif 'MMPose' in f.get('format', ''):
                records = json.loads(file.read_text(encoding='utf-8'))
                for frame in records:
                    inst = frame['instances'][0]
                    if np.array(inst['keypoints']).shape != (24, 2) or len(inst['keypoint_scores']) != 24:
                        raise ValueError('Expected 24 keypoints per MMPose frame.')
            else:
                raise ValueError('Select DLC, MMPose JSON or SLEAP Analysis HDF5.')
            no_overwrite(dest / f'{camera}.json')
            pending[camera] = records
        if len({len(v) for v in pending.values()}) != 1:
            raise ValueError('Camera frame counts differ.')
        for camera, data in pending.items():
            write_json(dest / f'{camera}.json', data)
        write_json(dest / 'mapping.json', mapping)
        return {'files': [str(dest / f'{camera}.json') for camera in pending], 'frames': len(next(iter(pending.values())))}

    if op == 'pose':
        videos = path(f, 'videos', True)
        dest = output(path(f, 'out1'))
        for i in range(1, 9):
            if not (videos / f'data{i}.mp4').is_file():
                raise ValueError(f'Missing data{i}.mp4')
        model_args = ['--pose2d', str(path(f, 'poseConfig', True)), '--pose2d-weights', str(path(f, 'poseWeights', True)),
            '--det-model', str(path(f, 'detConfig', True)), '--det-weights', str(path(f, 'detWeights', True))]
        for i in range(1, 9):
            print(f'Camera {i}/8', flush=True)
            subprocess.run([sys.executable, str(ROOT / '1_keypoints/demo/inferencer_demo.py'), str(videos / f'data{i}.mp4'),
                *model_args, '--pred-out-dir', str(dest), '--vis-out-dir', str(dest / 'preview'), '--draw-bbox'], cwd=ROOT/'1_keypoints', check=True)
        pose_videos=module('keypoint_video_utils','1_keypoints/demo/video_utils.py').pose_videos
        preview=pose_videos(dest/'preview',dest/'preview'/('browser-'+str(f.get('_jobId','preview'))))
        preview['files'].insert(0,str(dest))
        for view in preview['videos']:view['index']+=1
        return preview

    if op == 'triangulate':
        from workflow_inputs import triangulation_inputs
        keypoints,calibration,dest=triangulation_inputs(f)
        output(dest)
        no_overwrite(dest / 'tri_new.npy')
        subprocess.run([sys.executable, str(ROOT/'1_keypoints/demo/triangulation.py'), '--cali_folder', str(calibration),
                        '--demo_folder', str(keypoints), '--output_path', str(dest/'tri_new.npy')], check=True)
        return {'files': [str(dest/'tri_new.npy')]}

    if op == 'reprojection':
        from workflow_inputs import reprojection_inputs
        source,videos,calibration,dest,unit,calibration_unit=reprojection_inputs(f)
        with tempfile.TemporaryDirectory(prefix='mousegpt-reprojection-') as tmp:
            result=Path(tmp)/'result.json'
            subprocess.run([sys.executable,str(ROOT/'1_keypoints/demo/project_3d_keypoints.py'),'--keypoints_3d',str(source),'--calib_folder',str(calibration),'--videos',str(videos),'--output_folder',str(dest),'--input_unit',unit,'--calibration_unit',calibration_unit,'--result',str(result)],check=True)
            return json.loads(result.read_text(encoding='utf-8'))

    if op == 'preprocess':
        dest = output(path(f, 'out2'))
        fps = float(f.get('fps', 0))
        if fps <= 0 or not np.isfinite(fps):
            raise ValueError('FPS must be positive.')
        extractor = module('ui_extract', '2_VLM/data_prepareation/vids_to_imgs.py')
        pre = module('ui_preprocess', '2_VLM/data_prepareation/preprocess_imgs.py')
        ffmpeg=shutil.which('ffmpeg')
        if not ffmpeg:
            candidate=Path(sys.prefix)/'Library/bin/ffmpeg.exe'
            if candidate.is_file():ffmpeg=str(candidate)
        if not ffmpeg:
            try:
                import imageio_ffmpeg
                ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:pass
        if not ffmpeg:raise ValueError('FFmpeg not found in PATH or the selected pose environment.')
        # Copy only small keypoint inputs in a temporary directory; videos stay in place.
        with tempfile.TemporaryDirectory(prefix='mousegpt-pre-') as tmp:
            tmp = Path(tmp)
            video_dir = output(tmp / 'videos')
            for camera in (3, 5):
                video = path(f, f'video{camera}', True)
                try:
                    os.link(video, video_dir/f'data{camera}.mp4')
                except OSError:
                    # Direct ffmpeg calls avoid copying recordings across drives.
                    pass
            if all((video_dir/f'data{i}.mp4').exists() for i in (3, 5)):
                extractor.extract_frames(video_dir, dest, 0, 0, fps, ffmpeg=ffmpeg)
            else:
                for camera in (3, 5):
                    image_dir = dest/'images'/f'data{camera}'
                    no_overwrite(image_dir)
                    output(image_dir)
                    filt = r'select=not(mod(n\,12))'
                    raw = output(tmp/f'raw{camera}')
                    subprocess.run([ffmpeg, '-nostdin', '-n', '-i', str(path(f,f'video{camera}',True)), '-filter:v', filt, '-vsync','0',str(raw/'%08d.jpg')],check=True,stdout=subprocess.DEVNULL)
                    extractor.rename(fps, raw, image_dir, 0)
            key_dir = output(tmp/'keypoints')
            for camera in (3,5):
                shutil.copy2(path(f,'keypoints',True)/f'data{camera}.json', key_dir/f'data{camera}.json')
            shutil.copy2(path(f,'tri',True),key_dir/'tri_filter.npy')
            pre.process_data(str(dest),str(key_dir),fps)
        return {'files':[str(dest/'pre_images'),str(dest/'velocity.json')]}

    if op in {'describe', 'previews'}:
        from PIL import Image
        crops = [path(f, f'crop{i}', True) for i in (3, 5)]
        ids = [{p.stem:p for p in folder.glob('*.jpg')} for folder in crops]
        if not ids[0] or ids[0].keys() != ids[1].keys():
            raise ValueError('Both crop directories must contain the same nonempty frame IDs.')
        dest = output(path(f, 'out2'))
        limit = int(f.get('limit', 10))
        if limit < 0:
            raise ValueError('Frame limit must be >= 0 (0 = all).')
        chosen = sorted(ids[0])[:limit or None]
        if op == 'previews':
            target = output(dest/'paired_previews')
            for frame in chosen:
                out = target/f'{frame}.jpg'
                no_overwrite(out)
                with Image.open(ids[0][frame]) as a, Image.open(ids[1][frame]) as b:
                    if a.height != b.height:
                        raise ValueError('Paired crops must have equal heights.')
                    canvas = Image.new('RGB',(a.width+b.width,a.height))
                    canvas.paste(a,(0,0));canvas.paste(b,(a.width,0));canvas.save(out,quality=95)
            return {'files':[str(target)],'frames':len(chosen)}
        from mousegpt_config import load_config
        script = module('ui_prompt','2_VLM/data_prepareation/prompt_request.py')
        prompt = str(f.get('prompt','')).strip() or script.build_system_prompt()
        extra = str(f.get('customPrompt','')).strip()
        script.build_system_prompt = lambda: prompt + ('\n'+extra if extra else '')
        velocities = json.loads(path(f,'velocity',True).read_text(encoding='utf-8'))
        load_config('VISION')
        for i,frame in enumerate(chosen):
            no_overwrite(dest/'descriptions'/f'{frame}.json')
            print(f'Describing {i+1}/{len(chosen)}: {frame}',flush=True)
            script.process_images(ids[0][frame],ids[1][frame],dest,velocities,max_attempts=3)
        return {'files':[str(dest/'descriptions')],'frames':len(chosen)}

    if op == 'embed':
        script = module('ui_embed','3_applications/embedding_request.py')
        source = path(f,'descriptions',True)
        dest = output(path(f,'features'))
        from mousegpt_config import load_config
        key = load_config('EMBEDDING').api_key
        files = script.description_paths_by_id(source)
        limit = int(f.get('limit',10))
        if limit<0:raise ValueError('Frame limit must be >= 0.')
        chosen = list(files.values())[:limit or None]
        for i,file in enumerate(chosen):
            print(f'Embedding {i+1}/{len(chosen)}: {file.stem}',flush=True)
            script.process_file(file.name,str(source),str(dest),key,max_attempts=3,timeout=60)
        return {'files':[str(dest)],'frames':len(chosen)}

    if op == 'package':
        import pandas as pd
        exporter = module('ui_export','3_applications/create_h5.py')
        source = path(f,'descriptions',True)
        features = path(f,'features',True)
        target = path(f,'outH5')
        no_overwrite(target)
        by_id = exporter.description_paths_by_id(source)
        ids = sorted(by_id)
        actual = {p.name[:-8] for p in features.glob('*_des.csv')}
        if actual != set(ids):
            raise ValueError('Description and vector IDs differ. Embed every description before packaging, or select a matching subset directory.')
        vectors = exporter.load_data_matrix([features/f'{i}_des.csv' for i in ids])
        descriptions = [exporter.read_description(by_id[i]) for i in ids]
        output(target.parent)
        temp = target.with_suffix('.building.h5')
        try:
            with pd.HDFStore(temp,mode='w') as store:
                store.put('descriptions',pd.DataFrame({'id':ids,'description':descriptions}))
                store.put('features_des',pd.DataFrame(vectors,index=ids))
            temp.replace(target)
        finally:
            temp.unlink(missing_ok=True)
        return {'files':[str(target)],'frames':len(ids)}

    if op == 'session':
        h5 = path(f,'h5',True)
        desc,features = aligned_session(h5)
        imgs = path(f,'previews',True)
        ids = [str(i).zfill(8) for i in desc.id]
        present = {p.stem for p in imgs.glob('*.jpg')}
        missing = sorted(set(ids)-present)
        return {'frames':len(ids),'dimensions':features.shape[1], 'matched_images':len(set(ids)&present),
            'missing_images':len(missing),'missing_examples':missing[:10], 'frame_ids':ids,
            'files':[str(h5),str(imgs)]}

    if op == 'review':
        exporter = module('ui_review','3_applications/create_h5.py')
        source = path(f,'descriptions',True)
        records = exporter.description_paths_by_id(source)
        frame = str(f.get('frame','0')).zfill(8)
        if frame not in records:
            raise ValueError(f'Description not found for frame {frame}')
        desc = exporter.read_description(records[frame])
        velocity = {}
        if f.get('velocity'):
            velocity = json.loads(path(f,'velocity',True).read_text(encoding='utf-8')).get(str(int(frame)),{})
        return {'description':desc,'velocity':velocity,'frame':frame,'frame_ids':list(records)}

    if op == 'project':
        import umap
        x = np.load(path(f,'vectors',True))
        n = min(int(f.get('neighbors',15)),len(x)-1)
        if n < 2:
            raise ValueError('UMAP needs at least 3 rows.')
        result = umap.UMAP(n_neighbors=n,n_components=2,random_state=42,init='random').fit_transform(x)
        target=path(f,'projection');np.save(target,result)
        return {'files':[str(target)]}

    if op.startswith('analysis:'):
        import pandas as pd
        from collections import Counter
        from scipy.cluster.hierarchy import linkage
        if op=='analysis:phenotype':
            dest=output(path(f,'analysisOutput'))
            subprocess.run([sys.executable,str(ROOT/'3_applications/behavior_phenotype_predict.py'),'--data_csv',str(path(f,'cohortCsv',True)),
                '--groups_json',str(path(f,'cohorts',True)),'--output',str(dest/'phenotype')],check=True)
            return {'files':[str(p) for p in (dest/'phenotype').glob('*')]}
        source = path(f,'h5',True)
        desc,features=aligned_session(source)
        dest=output(path(f,'analysisOutput'))
        kind=op.split(':')[1]
        values=desc.description.tolist()
        ids=desc.id.map(str).tolist()
        result_files=[]
        if kind=='keywords':
            rows=[]
            for frame,d in zip(ids,values):
                words=d.get('Keywords','') if isinstance(d,dict) else ''
                for w in words.split(','):
                    if w.strip():rows.append({'frame_id':frame,'keyword':w.strip()})
            if not rows:raise ValueError('No Keywords fields found.')
            pd.DataFrame(rows).to_csv(dest/'frame_keywords.csv',index=False)
            pd.DataFrame(Counter(r['keyword'] for r in rows).most_common(),columns=['keyword','frames']).to_csv(dest/'keyword_counts.csv',index=False)
            result_files=[dest/'frame_keywords.csv',dest/'keyword_counts.csv']
        elif kind=='novel':
            script=module('ui_novel','3_applications/novel_behavior.py')
            indices,scores=script.detect_outliers(features.to_numpy())
            pd.DataFrame({'frame_id':[ids[i] for i in indices],'anomaly_score':scores}).to_csv(dest/'novel_candidates.csv',index=False)
            result_files=[dest/'novel_candidates.csv']
        elif kind=='major':
            np.save(dest/'vectors.npy',features.to_numpy())
            projector=Path(f['projectionPython'])
            spec=dest/'projection_input.json'
            write_json(spec,{'op':'project','fields':{'vectors':str(dest/'vectors.npy'),'projection':str(dest/'umap.npy'),'neighbors':f.get('neighbors',15)}})
            subprocess.run([str(projector),str(Path(__file__)),str(spec)],check=True)
            projection=np.load(dest/'umap.npy')
            Z=linkage(projection,method='single')
            sys.path.insert(0,str(ROOT/'3_applications'))
            import llm_cluster as coarse
            from mousegpt_config import load_config,sdk_client
            coarse.client=sdk_client(load_config('ANALYSIS'),max_retries=0,timeout=60)
            clusters=int(f.get('clusters',8));minimum=int(f.get('minCluster',3))
            labels=coarse.filter_minor_cluster(Z,clusters,minimum)
            words=coarse.get_most_common_words(labels,pd.DataFrame(values))
            merged={'New Cluster Names':{},'Mapping':{}}
            items=list(words.items())
            for start in range(0,len(items),60):
                chunk=coarse.request_cluster_merges(dict(items[start:start+60]),merged['New Cluster Names'])
                merged['New Cluster Names'].update(chunk['New Cluster Names']);merged['Mapping'].update(chunk['Mapping'])
            labels,names=coarse.update_cluster_labels(labels,merged)
            target=dest/source.name;no_overwrite(target);shutil.copy2(source,target)
            categories,assigned=coarse.category_tables(labels,names,ids)
            with pd.HDFStore(target) as store:
                store.put('linkage_matrix',pd.DataFrame(Z));store.put('category_All_cluster_ui',categories);store.put('label_All_cluster_ui',assigned)
            pd.DataFrame({'frame_id':ids,'x':projection[:,0],'y':projection[:,1],'label':labels,'behavior':[names[int(i)] for i in labels]}).to_csv(dest/'major_behaviors.csv',index=False)
            result_files=[target,dest/'major_behaviors.csv']
        elif kind=='fine':
            sys.path.insert(0,str(ROOT/'3_applications'))
            import fine_behaviors as fine
            from mousegpt_config import load_config,sdk_client
            fine.client=sdk_client(load_config('ANALYSIS'),max_retries=0,timeout=60)
            categories=fine.load_categories(str(path(f,'categories',True)))
            prompt=path(f,'finePrompt',True).read_text(encoding='utf-8')
            part=f.get('part','Head')
            labels,counts,categories=fine.get_labels_category(prompt,categories,values,part,max_category_updates=int(f.get('maxUpdates',20)),output_path=str(dest/'categories.json'))
            target=dest/source.name;no_overwrite(target);shutil.copy2(source,target)
            with pd.HDFStore(target) as store:
                store.put(f'category_{part}_keyword_ui',pd.DataFrame(list(categories)))
                store.put(f'label_{part}_keyword_ui',pd.DataFrame(labels,index=ids))
            pd.DataFrame({'frame_id':ids,'label':labels}).to_csv(dest/'fine_labels.csv',index=False)
            write_json(dest/'counts.json',counts);result_files=[target,dest/'fine_labels.csv',dest/'categories.json',dest/'counts.json']
        elif kind=='temporal':
            fps=float(f.get('fps',0));step=int(f.get('sampleStep',12))
            if fps<=0 or step<1:raise ValueError('FPS and sample step must be positive.')
            with pd.HDFStore(source,mode='r') as store:
                label_key=str(f.get('labelKey',''))
                if not label_key.startswith('label_'):raise ValueError('Select an existing label_* HDF5 table.')
                table=store[label_key]
            if table.index.map(str).tolist()!=ids:raise ValueError('Label IDs are not aligned.')
            labels=table.iloc[:,0].tolist();frames=[int(i) for i in ids]
            bouts=[];transitions=[];start=0
            for i in range(1,len(ids)+1):
                contiguous=i<len(ids) and frames[i]-frames[i-1]==step
                same=contiguous and labels[i]==labels[start]
                if same:continue
                bouts.append({'label':labels[start],'start_frame':frames[start],'end_frame':frames[i-1], 'samples':i-start,
                    'observed_duration_s':(frames[i-1]-frames[start]+step)/fps})
                if contiguous:transitions.append({'from':labels[i-1],'to':labels[i]})
                start=i
            pd.DataFrame(bouts).to_csv(dest/'bouts.csv',index=False)
            counts=Counter((r['from'],r['to']) for r in transitions)
            pd.DataFrame([{'from':a,'to':b,'count':n} for (a,b),n in counts.items()],columns=['from','to','count']).to_csv(dest/'transitions.csv',index=False)
            result_files=[dest/'bouts.csv',dest/'transitions.csv']
        elif kind=='phenotype':
            subprocess.run([sys.executable,str(ROOT/'3_applications/behavior_phenotype_predict.py'),'--data_csv',str(path(f,'cohortCsv',True)),
                '--groups_json',str(path(f,'cohorts',True)),'--output',str(dest/'phenotype')],check=True)
            result_files=list((dest/'phenotype').glob('*'))
        else:raise ValueError('Unsupported analysis.')
        return {'files':[str(p) for p in result_files],'frames':len(ids)}
    raise ValueError(f'Unsupported operation: {op}')


def main():
    p=argparse.ArgumentParser();p.add_argument('job');args=p.parse_args()
    spec=json.loads(Path(args.job).read_text(encoding='utf-8'))
    sys.path.insert(0,str(ROOT/'3_applications'))
    try:
        result=run(spec['op'],spec.get('fields',{}),spec.get('tabs',{}))
    except subprocess.CalledProcessError as error:
        print(f'Processing command failed (exit {error.returncode}). See the preceding log for the original error.',file=sys.stderr,flush=True)
        raise SystemExit(error.returncode or 1)
    if spec.get('result'):
        write_json(Path(spec['result']),result)
    print('Completed.',flush=True)


if __name__=='__main__':main()
