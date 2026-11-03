/* Stable task DOM: polling changes text, never replaces open log elements. */
const TASK_NAMES={reprojection:'3D reprojection','pose-previews':'2D overlay videos',pose:'2D pose estimation',convert:'Keypoint conversion',triangulate:'3D reconstruction',filter:'Filter & 3D visualization',preprocess:'Image & velocity preprocessing',describe:'Behavior descriptions',previews:'Paired preview images',embed:'Description embeddings',package:'Session packaging',session:'Session validation',review:'Frame loading',chat:'Session Q&A'};
function taskName(job){if(job.op==='api-test')return 'API test'+(job.monitor?.startsWith('api-test:')?' · '+job.monitor.split(':')[1]:'');return TASK_NAMES[job.op]||analysis?.find(a=>'analysis:'+a[0]===job.op)?.[1]||job.op}
function captureView(root){
 const details=[...root.querySelectorAll('details')].map((e,i)=>[e.dataset.viewKey||i+':'+e.querySelector('summary')?.textContent,e.open]);
 const scroll=[...root.querySelectorAll('pre,textarea')].map((e,i)=>[e.dataset.viewKey||i,e.scrollTop,e.scrollLeft]);
 const inputs=[...root.querySelectorAll('input,textarea,select')].map((e,i)=>({key:e.dataset.existingPython||e.id||i,value:e.value,checked:e.checked,focused:e===document.activeElement,start:e.selectionStart,end:e.selectionEnd}));
 const pageY=window.scrollY;
 return()=>{
  [...root.querySelectorAll('details')].forEach((e,i)=>{const saved=details.find(d=>d[0]===(e.dataset.viewKey||i+':'+e.querySelector('summary')?.textContent));if(saved)e.open=saved[1]});
  [...root.querySelectorAll('input,textarea,select')].forEach((e,i)=>{const saved=inputs.find(d=>d.key===(e.dataset.existingPython||e.id||i));if(!saved)return;e.value=saved.value;if(e.type==='checkbox')e.checked=saved.checked;if(saved.focused){e.focus({preventScroll:true});if(saved.start!=null&&typeof e.setSelectionRange==='function')try{e.setSelectionRange(saved.start,saved.end)}catch{}}});
  [...root.querySelectorAll('pre,textarea')].forEach((e,i)=>{const saved=scroll.find(d=>d[0]===(e.dataset.viewKey||i));if(saved){e.scrollTop=saved[1];e.scrollLeft=saved[2]}});
  if(window.scrollY!==pageY)window.scrollTo(window.scrollX,pageY);
 }
}
function mountTaskMonitors(root=document){
 const buttons=[...root.querySelectorAll('[data-run],[data-api-test],[data-analysis]')];
 if(root.querySelector('#chat-form'))buttons.push(root.querySelector('#chat-form'));
 if(root.querySelector('#review-facts')){const b=root.querySelector('[data-action=load-frame]');if(b)buttons.push(b)}
 for(const b of buttons){
  const scope=b.dataset.run||(b.dataset.apiTest?'api-test:'+b.dataset.apiTest:b.dataset.analysis?'analysis:'+b.dataset.analysis:b.id==='chat-form'?'chat':'review');
  if(b._taskMonitor?.isConnected)continue;
  const anchor=b.closest('.button-group,.review-toolbar')||b;
  let container=anchor._taskMonitorContainer;
  if(!container?.isConnected){container=document.createElement('div');container.className='local-monitors';anchor.insertAdjacentElement('afterend',container);anchor._taskMonitorContainer=container}
  const monitor=document.createElement('section');monitor.className='local-monitor';monitor.dataset.monitorScope=scope;monitor.setAttribute('aria-label',(TASK_NAMES[scope]||scope)+' task status');
  const title=document.createElement('div');title.className='local-monitor-title';title.textContent='Task status · '+(TASK_NAMES[scope]||scope.replace(/^analysis:/,''));
  const content=document.createElement('div');content.className='local-monitor-content';content.innerHTML='<span class="status-line">No task submitted.</span>';
  monitor.append(title,content);container.append(monitor);b._taskMonitor=monitor;
 }
}
function createTaskView(job){
 const node=document.createElement('article');node.className='task';node.dataset.jobId=job.id;
 node.innerHTML='<div class="button-group"><strong data-task-name></strong><span class="badge" data-task-status></span><button class="small-button" data-cancel="'+esc(job.id)+'">Cancel</button></div><p class="task-latest status-line" data-task-latest></p><pre class="task-log error" data-task-error hidden></pre><div data-task-result></div><details data-task-log><summary>Log</summary><label class="follow-log"><input type="checkbox" data-log-follow>Follow latest log</label><pre class="task-log" data-task-log-text tabindex="0" aria-label="Task log"></pre></details>';
 return node;
}
function updatePre(element,value,follow=false){
 if(element.textContent===value)return;
 const top=element.scrollTop,left=element.scrollLeft;
 element.textContent=value;element.scrollTop=follow?element.scrollHeight:top;element.scrollLeft=left;
}
function updateTaskView(node,job){
 node.querySelector('[data-task-name]').textContent=taskName(job);
 const status=node.querySelector('[data-task-status]');if(status.textContent!==job.status){status.textContent=job.status;status.className='badge status-'+job.status}
 node.querySelector('[data-cancel]').hidden=!['queued','running'].includes(job.status);
 const log=(job.log||[]).join('\n');const latest=[...(job.log||[])].reverse().find(x=>x.trim())||'';
 const line=node.querySelector('[data-task-latest]');if(line.textContent!==latest)line.textContent=latest;line.hidden=!latest||job.status==='failed';
 const error=node.querySelector('[data-task-error]');error.hidden=!job.error;updatePre(error,job.error||'');
 const details=node.querySelector('[data-task-log]');details.hidden=!log;
 updatePre(node.querySelector('[data-task-log-text]'),log,node.querySelector('[data-log-follow]').checked);
 const result=node.querySelector('[data-task-result]'),fingerprint=JSON.stringify(job.result||null);
 if(result._fingerprint!==fingerprint){
  result._fingerprint=fingerprint;
  result.innerHTML=job.result?resultSummary(job.result)+(job.result.files||[]).map((p,i)=>/\.[a-z0-9]+$/i.test(p)?'<a class="result-link" href="/api/download?job='+encodeURIComponent(job.id)+'&index='+i+'">Download '+esc(p.split(/[\\/]/).pop())+'</a>':'').join(''):'';
  if(job.result)attachPoseReview(result,job);
 }
}
function drawTaskViews(){
 const ordered=[...jobs].sort((a,b)=>(b.created_at||'').localeCompare(a.created_at||'')||b.id.localeCompare(a.id));
 const list=$('#task-list');
 if(list){
  if(!ordered.length){if(!list.querySelector('.status-line'))list.innerHTML='<div class="status-line">No tasks submitted.</div>'}
  else{
   list.querySelector(':scope > .status-line')?.remove();
   for(let i=0;i<ordered.length;i++){
    const job=ordered[i];let node=[...list.children].find(n=>n.dataset.jobId===job.id);
    if(!node){node=createTaskView(job);list.insertBefore(node,list.children[i]||null)}
    updateTaskView(node,job);
   }
  }
 }
 document.querySelectorAll('[data-monitor-scope]').forEach(monitor=>{
  const scope=monitor.dataset.monitorScope,job=ordered.find(j=>(j.monitor||j.op)===scope);
  const content=monitor.querySelector('.local-monitor-content');if(!job)return;
  let node=content.querySelector('.task');
  if(!node||node.dataset.jobId!==job.id){node=createTaskView(job);content.replaceChildren(node)}
  updateTaskView(node,job);
 });
}
