/* Environment installation UI; system operations are owned by the backend. */
let envData,envFingerprint;
function environments(){return `<div class="page-title"><div><h1>Environments</h1><p class="subtitle">Install a fixed release, validate it, then select Use. Existing environments are preserved.</p></div><button class="secondary" data-env-refresh>Check system</button></div><div id="environment-content"><div class="empty">Checking this computer…</div></div>`}
function drawEnvironments(){
 const root=$('#environment-content');if(!root||!envData)return;
 const restore=captureView(root);
 const {system,profiles,environments:records,active,operation}=envData;
 const busy=['queued','creating','installing','validating'].includes(operation.status);
 const validated=panel('Validated environments',records.length?`<div class="environment-table">${records.map(r=>{
  const enabled=active[r.profile]===r.id&&r.available;
  const status=enabled?'Active':r.available?'Validated':'Unavailable';
  return `<article class="environment-row" data-env-card="${esc(r.id)}"><div class="environment-card-main"><div class="environment-card-heading"><strong>${esc(profiles.find(p=>p.id===r.profile)?.name||r.profile)}</strong><span class="badge env-status-${status.toLowerCase()}">${status}</span></div><p class="muted">${r.managed?'Release '+esc(r.release):'Custom environment'} / Python ${esc(r.validation.python)}</p><code>${esc(r.python)}</code>${(r.limitations||[]).map(text=>`<p class="environment-limitation">${esc(text)}</p>`).join('')}<details><summary>Validation details</summary><p class="muted">${r.validation.checks.map(esc).join(' / ')}</p></details></div><div class="environment-card-actions"><button class="secondary" data-env-use="${r.id}" ${busy||!r.available||enabled?'disabled':''}>Use</button></div></article>`
 }).join('')}</div>`:'<div class="empty">No validated environments. Install a profile below or validate an existing environment.</div>');
 const installs=`<div class="grid">${profiles.map(p=>{
  const enabled=records.find(r=>r.id===active[p.id]&&r.available);
  return panel(p.name,`<p class="muted">${esc(p.description)}</p>${p.install_supported===false?'':`<div class="io-line"><b>Release</b> ${p.release} / Python ${p.python} / ${esc(system.os)} ${esc(system.architecture)}</div>`}<div class="io-line"><b>Active</b> ${enabled?esc(enabled.managed?'Managed release '+enabled.release:'Validated custom environment'):'No environment selected'}</div>${p.install_supported===false?'':`<div class="button-group" style="margin-top:16px"><button class="primary" data-env-install="${p.id}" ${busy||!system.install_supported||!p.lock_available?'disabled':''}>Install & validate</button></div>`}<details><summary>Use an existing environment</summary><label class="field"><span>Python executable</span><input data-existing-python="${p.id}" value="${esc(p.suggested_python||'')}" placeholder="${system.os==='Windows'?'C:/path/to/environment/python.exe':'/path/to/environment/bin/python'}"></label><button class="secondary" data-env-check="${p.id}" ${busy?'disabled':''}>Validate existing environment</button></details>`)
 }).join('')}</div>`;
 root.innerHTML=validated+installs+
 panel('Installation status',`<div class="button-group"><span class="badge status-${operation.status}">${esc(operation.status)}</span>${operation.profile?`<span>${esc(operation.profile)}</span>`:''}${busy?'<button class="secondary" data-env-cancel>Cancel</button>':''}</div>${operation.error?`<p class="error">${esc(operation.error)}</p>`:''}${operation.directory?`<p class="muted">${esc(operation.directory)}</p>`:''}${operation.log?.length?`<pre class="task-log">${esc(operation.log.join('\n'))}</pre>`:''}`)+
 panel('System',`<div class="row"><div><b>${esc(system.os)} ${esc(system.os_version)}</b><p class="muted">${esc(system.architecture)} / ${system.free_gb} GB free</p></div><div><span class="badge">${system.install_supported?'Installation available':'Installation unavailable'}</span><p class="muted">${esc(system.reason)}${system.validation_status==='pending-platform-verification'?' '+esc(system.platform_label)+': real-machine verification pending.':''}</p></div></div><details><summary>Storage location</summary><code>${esc(system.app_home)}</code></details>`);
 restore();
}
async function refreshEnvironments(){try{const next=await api('environments');const fingerprint=JSON.stringify(next);if(fingerprint!==envFingerprint||!$('#environment-content')?.querySelector('section')){envData=next;envFingerprint=fingerprint;drawEnvironments()}}catch(e){toast(e.message)}}
document.addEventListener('click',async e=>{
 const b=e.target.closest('button');if(!b)return;
 try{
  let body;
  if(b.hasAttribute('data-env-refresh'))return await refreshEnvironments();
  if(b.dataset.envInstall)body={action:'install',profile:b.dataset.envInstall};
  if(b.dataset.envCheck){const id=b.dataset.envCheck;body={action:'check',profile:id,python:document.querySelector('[data-existing-python="'+id+'"]').value}}
  if(b.dataset.envUse)body={action:'activate',id:b.dataset.envUse};
  if(b.hasAttribute('data-env-cancel'))body={action:'cancel'};
  if(!body)return;
  b.disabled=true;envData=await api('environments',body);drawEnvironments();
  if(body.action==='activate')toast('Environment selected. Other environments are unchanged.');
 }catch(e){b.disabled=false;toast(e.message)}
});
setInterval(()=>{if(location.hash==='#environments'&&connected)refreshEnvironments()},1800);

async function ensureEnvironment(op){
 const result=await api('environment-access?op='+encodeURIComponent(op));
 if(result.ready)return;
 show('Environment required',`<p>${esc(result.message)}</p><button class="primary" data-open-environments>Open Environments</button>`);
 const error=Error(result.message);error.environmentRequired=true;throw error;
}
document.addEventListener('click',e=>{if(e.target.closest('[data-open-environments]')){$('#modal').close();location.hash='environments'}});
