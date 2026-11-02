function filterVisualizationPanel(){return panel('Filter & 3D visualization',`
 <div class="row">${path('Reconstructed coordinates .npy','rawTri','','Use tri_new.npy from reconstruction above.')}${select('Source spatial unit','units',[['','Select unit'],['m','Metres'],['mm','Millimetres'],['cm','Centimetres']])}</div>
 ${path('Filtered output directory (optional)','filterOutput','','Default: beside the input NPY. Outputs: tri_filter.npy and tri_filter.mp4.')}
 <p class="hint">The 3D animation includes every frame, starting at frame 0.</p>
 <details><summary>Playback settings</summary>${path('Original video for FPS (optional)','animationVideo','','Uses data1.mp4 from the Video directory above, or detects videos/data1.mp4 beside the session.')}${field('Override playback FPS (optional)','animationFps','','number','Default: original video FPS. If no video is available, 60 FPS.')}</details>
 <p class="hint">Fills bounded missing values, smooths trajectories and converts to metres. The 3D animation appears when processing completes.</p>
 <details><summary>Script</summary><p>1_keypoints/demo/batch_vis.py: filter, convert units and render the 3D animation.</p></details>
 <div class="button-group">${runButton('filter','Filter & visualize')}</div>`,'3')}
function stage1ReviewPanel(){return panel('Reprojection review',`
 <div class="row">${path('3D keypoints .npy (optional)','reprojectionTri','','Leave blank to use tri_filter.npy from the filtering step above.')}${select('NPY spatial unit','reprojectionUnit',[['m','Metres (tri_filter.npy)'],['mm','Millimetres'],['cm','Centimetres']])}</div>
 <div class="row">${path('Original video directory (optional)','reprojectionVideos','','Leave blank to use the video directory above. Requires data1.mp4 ... data8.mp4.')}${path('Calibration directory (optional)','reprojectionCalibration','','Leave blank to use the calibration directory above. Camera1...Camera8 filenames are mapped by ID; other MAT filenames use sorted order.')}</div>
 ${select('Calibration translation unit','calibrationUnit',[['','Select unit'],['mm','Millimetres'],['cm','Centimetres'],['m','Metres']])}
 ${path('Reprojection output directory (optional)','reprojectionOutput','','Default: reprojection/ beside the NPY. Existing results are not overwritten.')}
 <p class="hint">Outputs data1.json ... data8.json and eight overlay MP4 videos. Every NPY frame must correspond to the same original video frame, starting at 0. Select a camera below after completion.</p>
 <details><summary>Script</summary><p>1_keypoints/demo/project_3d_keypoints.py, adapted from MouseIMU/project_3d_keypoints.py.</p></details>
 <div class="button-group">${runButton('reprojection','Reproject & visualize')}</div>`,'4')}


function posePreviewPanel(){return `<details class="pose-existing"><summary>Display existing 2D overlay videos</summary>${path('2D overlay video directory','posePreviewDir','','data1.mp4 ... data8.mp4. Default: preview/ in the keypoint output directory.')}<p class="hint">Uses videos saved by 1_keypoints/demo/inferencer_demo.py; no model inference is repeated.</p><div class="button-group">${runButton('pose-previews','Load 2D videos')}</div></details>`}

function attachCameraVideos(result,job,title){
 const holder=document.createElement('section');holder.className='camera-video-viewer';
 const label=document.createElement('label');label.className='field';const text=document.createElement('span');text.textContent=title+' · Camera view';
 const select=document.createElement('select');select.setAttribute('aria-label',title+' camera');
 for(const view of job.result.videos){const option=document.createElement('option');option.value=view.index;option.textContent='data'+view.camera;select.append(option)}
 const video=document.createElement('video');video.controls=true;video.preload='metadata';video.className='pose-review-video';video.setAttribute('aria-label',title);
 label.append(text,select);holder.append(label,video);result.prepend(holder);
 function switchView(){const time=video.currentTime||0,playing=!video.paused;video.onloadedmetadata=()=>{video.currentTime=Math.min(time,Math.max(0,video.duration-.01));if(playing)video.play().catch(()=>{})};video.src='/api/media?job='+encodeURIComponent(job.id)+'&index='+select.value;video.load()}
 select.onchange=switchView;switchView();
}

function attachPoseReview(result,job){
 if(job.result?.viewer==='reprojection-videos'){attachCameraVideos(result,job,'3D reprojection');return}
 if(job.result?.viewer==='pose-videos'){attachCameraVideos(result,job,'2D keypoint overlay');return}
 if(job.result?.viewer!=='3d-video')return;
 const title=document.createElement('h3');title.textContent='Filtered 3D skeleton';
 const video=document.createElement('video');video.controls=true;video.preload='metadata';video.className='pose-review-video';video.setAttribute('aria-label','Filtered 3D skeleton');
 video.src='/api/media?job='+encodeURIComponent(job.id)+'&index='+job.result.videos[0].index+'&v='+Date.now();
 result.prepend(title,video);
}
