/* Browse actual JPG frame IDs, including sparse recordings. No model calls. */
let stage4Frames=[],stage4Directory='',stage4Generation=0,stage4Timer,stage4PathTimer;
let stage4ImageBusy=false,stage4Wanted=null;
function stage4FrameControls(){return `<div class="frame-browser"><div class="review-toolbar"><label>Frame <input id="frame" type="number" min="0" value="${Number(state.frame)||0}" disabled></label><span id="frame-position" class="muted"></span></div><input id="frame-slider" type="range" min="0" max="0" step="1" value="0" aria-label="Available frame images" disabled><p id="frame-browser-status" class="muted">Select a frame image directory.</p></div>`}
function stage4Message(message){const e=$('#frame-browser-status');if(e)e.textContent=message;}
function stage4Current(generation,directory){return location.hash==='#stage4'&&generation===stage4Generation&&directory===val('previews').trim()&&!!$('#frame-slider');}
async function refreshStage4Frames(){
 const generation=++stage4Generation,directory=val('previews').trim();
 clearTimeout(stage4Timer);stage4Wanted=null;stage4Frames=[];stage4Directory=directory;
 if(!$('#frame-slider'))return;
 $('#frame-slider').disabled=true;$('#frame').disabled=true;$('#frame-position').textContent='';
 $('#preview-image').hidden=true;$('#preview-empty').hidden=false;
 if(!directory){stage4Message('Select a frame image directory.');return;}
 stage4Message('Reading available images...');
 try{
  const data=await api('preview-frames',{previews:directory});
  if(!stage4Current(generation,directory))return;
  stage4Frames=data.frame_ids;
  if(!stage4Frames.length){stage4Message('No frame JPGs found. Expected filenames such as 00000012.jpg.');return;}
  $('#frame-slider').max=stage4Frames.length-1;$('#frame-slider').disabled=false;
  $('#frame').disabled=false;$('#frame').min=stage4Frames[0];$('#frame').max=stage4Frames.at(-1);
  stage4Select(nearestStage4Frame(Number(state.frame)||0));
 }catch(e){if(stage4Current(generation,directory))stage4Message(e.message);}
}
function nearestStage4Frame(frame){let best=0;for(let i=1;i<stage4Frames.length;i++)if(Math.abs(stage4Frames[i]-frame)<Math.abs(stage4Frames[best]-frame))best=i;return best;}
function stage4Select(index){
 if(!stage4Frames.length)return;
 index=Math.max(0,Math.min(stage4Frames.length-1,index));
 const frame=stage4Frames[index];state.frame=frame;save();
 $('#frame').value=frame;$('#frame-slider').value=index;
 $('#frame-position').textContent=`${index+1} / ${stage4Frames.length} images`;
 $('.preview-label').textContent='Frame '+String(frame).padStart(8,'0');
 $('#preview-image').hidden=true;$('#preview-empty').hidden=false;$('#preview-empty').textContent='Loading image...';
 stage4Message('');clearTimeout(stage4Timer);
 stage4Timer=setTimeout(()=>{stage4Wanted={frame,directory:stage4Directory,generation:stage4Generation};drainStage4Images();},200);
}
async function drainStage4Images(){
 if(stage4ImageBusy)return;
 stage4ImageBusy=true;
 try{while(stage4Wanted){
  const request=stage4Wanted;stage4Wanted=null;
  if(!stage4Current(request.generation,request.directory))continue;
  try{
   const result=await api('preview',{frame:request.frame,previews:request.directory});
   if(!stage4Current(request.generation,request.directory)||Number(state.frame)!==request.frame)continue;
   const img=$('#preview-image');
   img.onload=()=>{if(stage4Current(request.generation,request.directory)&&Number(state.frame)===request.frame){img.hidden=false;$('#preview-empty').hidden=true;}};
   img.onerror=()=>{if(stage4Current(request.generation,request.directory))stage4Message('Could not display this image.');};
   img.src=result.image;
  }catch(e){if(stage4Current(request.generation,request.directory)&&Number(state.frame)===request.frame){stage4Message(e.message);$('#preview-empty').textContent='Image unavailable.';}}
 }}finally{stage4ImageBusy=false;}
}
document.addEventListener('input',e=>{
 if(location.hash!=='#stage4')return;
 if(e.target.id==='frame-slider')stage4Select(Number(e.target.value));
 if(e.target.dataset.field==='previews'){
  ++stage4Generation;stage4Wanted=null;clearTimeout(stage4Timer);clearTimeout(stage4PathTimer);
  if($('#frame-slider')){$('#frame-slider').disabled=true;$('#frame').disabled=true;$('#preview-image').hidden=true;$('#preview-empty').hidden=false;}
  stage4PathTimer=setTimeout(refreshStage4Frames,600);
 }
});
document.addEventListener('change',e=>{
 if(location.hash==='#stage4'&&e.target.id==='frame'&&stage4Frames.length){
  const frame=Number(e.target.value);if(Number.isFinite(frame))stage4Select(nearestStage4Frame(frame));
 }
});
