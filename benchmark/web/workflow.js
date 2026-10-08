'use strict';

/* The browser prepares and reviews work. Live model calls are CLI-only. */
(() => {
  const workflow = {data:null, rubricId:null, rubricDirty:false, selectedBatch:null, loadPromise:null, initialized:false, taskRequest:0};
  const action = (name, body = {}) => api('/api/workflow/action', {action:name,...body});
  const list = value => Array.isArray(value) ? value : [];
  const field = id => document.getElementById(id);
  const value = id => field(id).value.trim();
  const blank = text => `<p class="workflow-empty">${esc(text)}</p>`;
  const code = text => `<pre class="code-block">${esc(text)}</pre>`;
  function parseJSON(text, label) { try { return JSON.parse(text); } catch { throw new Error(`${label} must be valid JSON.`); } }
  function selectedFrames(scope) {
    if (scope === 'all') return [];
    const ids = scope === 'selected' ? (state.selected ? [state.selected] : []) : state.frames.map(f => f.id);
    if (!ids.length) throw new Error('Select a frame or a nonempty catalog filter first.');
    return ids;
  }
  function selectedRubric(id) { return field(id)?.value || null; }
  function latestRubric() { return list(workflow.data?.rubrics).at(-1); }
  function rubricOptions(selected) {
    return '<option value="">Choose a rubric version</option>' + list(workflow.data?.rubrics).slice().reverse().map(r => `<option value="${esc(r.id)}" ${r.id===selected?'selected':''}>${esc(r.label || r.content?.id || r.id)} · ${esc(r.status)} · ${esc(r.id)}</option>`).join('');
  }
  function updateRubricSelectors() {
    const preferred = list(workflow.data?.rubrics).filter(r => r.status==='frozen').at(-1)?.id || latestRubric()?.id;
    for (const id of ['batch-rubric','calibration-rubric','release-rubric']) {
      const select = field(id); if (!select) continue;
      const current = select.value; select.innerHTML = rubricOptions(current || preferred);
    }
  }
  function renderRubricPreview() {
    const selected=list(workflow.data?.rubrics).find(r=>r.id===workflow.rubricId);
    try {
      const content=parseJSON(value('rubric-json'),'Rubric');
      field('rubric-readable').innerHTML=window.MousebenchRubric.renderRubric(content,{id:selected?.id,status:selected?.status,unsaved:workflow.rubricDirty});
    } catch(error) { field('rubric-readable').innerHTML=`<p class="job-error" role="status">Readable preview unavailable: ${esc(error.message)} Your JSON edits are kept below.</p>`; }
  }
  function setRubric(rubric) {
    workflow.rubricId = rubric?.id || null; workflow.rubricDirty = false;
    field('rubric-json').value = json(rubric?.content || workflow.data?.default_rubric || {});
    field('rubric-state').textContent = rubric?.status || 'Proposed draft';
    field('rubric-state').className = `pill ${rubric?.status==='frozen'?'approved':'draft'}`;
    field('rubric-meta').textContent = rubric ? `${rubric.id} · ${rubric.status} · ${rubric.sha256 || ''}` : 'Save a reviewed draft to create the first rubric version.';
    field('rubric-freeze').disabled = !rubric || rubric.status === 'frozen';
    renderRubricPreview();
  }
  function renderRubrics() {
    field('rubric-versions').innerHTML = list(workflow.data.rubrics).length ? list(workflow.data.rubrics).slice().reverse().map(r => `<div class="version-card"><div class="batch-card-head"><h4>${esc(r.label || r.content?.id || r.id)}</h4><span class="pill ${r.status==='frozen'?'approved':'draft'}">${esc(r.status)}</span></div><p class="field-note">${esc(r.author || '')} · ${esc(r.created_at || '')}</p><p class="metadata-strip">${esc(r.sha256 || '')}</p><button type="button" class="text-button" data-edit-rubric="${esc(r.id)}">Inspect / edit this version ↗</button></div>`).join('') : blank('No rubric versions saved yet.');
    field('rubric-versions').querySelectorAll('[data-edit-rubric]').forEach(button => button.addEventListener('click', () => {
      if (workflow.rubricDirty && !window.confirm('Discard unsaved rubric edits and inspect this version?')) return;
      setRubric(workflow.data.rubrics.find(r => r.id === button.dataset.editRubric));
      field('rubric-form').scrollIntoView({behavior:'smooth',block:'center'});
    }));
  }
  function renderStatus() {
    const r = workflow.data.readiness || {};
    field('setup-status').replaceChildren();
    field('setup-input-status').textContent = `${fmt(r.input_contracts)} / ${fmt(r.frames)} frame inputs saved`;
    for(const id of ['bulk-input-scope','batch-scope'])field(id).querySelector('option[value="all"]').textContent=`All ${fmt(r.frames)} frames`;
    field('workflow-status').innerHTML = `<div class="workflow-status-items"><span><strong>${fmt(r.input_contracts)}</strong> / ${fmt(r.frames)} inputs frozen</span><span><strong>${fmt(r.gt_approved)}</strong> references approved</span><span><strong>${fmt(r.open_issues)}</strong> open issues</span><span><strong>${fmt(list(workflow.data.batches).length)}</strong> saved batches</span></div>`;
    if (!workflow.initialized) { setRubric(latestRubric()); workflow.initialized=true; }
    else if (!workflow.rubricDirty && workflow.rubricId) { const r = list(workflow.data.rubrics).find(r=>r.id===workflow.rubricId); if(r) setRubric(r); }
    renderRubrics(); updateRubricSelectors(); renderBatches(); renderIssues(); renderCalibration(); renderReleases(); updateRuns(); renderEvidenceLinks(); updateTaskControls();
  }
  function load() {
    if (workflow.loadPromise) return workflow.loadPromise;
    workflow.loadPromise = (async () => {
      try { workflow.data = await api('/api/workflow'); renderStatus(); return workflow.data; }
      catch(error) { showError(field('workflow-status'),error,load); showError(field('setup-status'),error,load); return null; }
      finally { workflow.loadPromise = null; }
    })();
    return workflow.loadPromise;
  }
  async function openTask(task) {
    if (!['ground_truth','gt_check','inference','judge'].includes(task)) return;
    const request = ++workflow.taskRequest;
    const perFrame = ['ground_truth','gt_check'].includes(task);
    if (perFrame && !state.selected) { toast('Select a frame in Ground truth first.',true); return; }
    switchPage('workflow');
    if (!await load() || state.page !== 'workflow' || request !== workflow.taskRequest) return;
    field('batch-task').value = task;
    field('batch-scope').value = perFrame ? 'selected' : 'filtered';
    updateTaskControls();
    field('workflow-batches').scrollIntoView({behavior:'smooth',block:'start'});
  }
  function updateRuns() {
    for(const id of ['batch-run','evidence-run','evidence-other-run','calibration-template-run']) {
      const current=field(id).value;
      field(id).innerHTML=`<option value="">${id==='evidence-other-run'?'No second output':'Choose a prediction run'}</option>`+state.runs.map(r=>`<option value="${esc(r.id)}" ${current===r.id?'selected':''}>${esc(r.name || r.id)} · ${esc(r.track)}</option>`).join('');
    }
    updateCalibrationTemplate();
  }
  function batchCounts(b) {
    const c = b.counts || {}, total=Object.values(c).reduce((sum,n)=>sum+(typeof n==='number'?n:0),0), done=(c.succeeded||0);
    return {c,total,done,percent:total?Math.round(done/total*100):0};
  }
  function renderBatches() {
    const batches = list(workflow.data.batches);
    field('batch-list').innerHTML = batches.length ? batches.slice().reverse().map(b => {
      const {c,total,done,percent}=batchCounts(b);
      return `<article class="batch-card"><div class="batch-card-head"><h4>${esc(b.name || b.id)}</h4><span class="pill neutral">${esc(b.task)} · ${esc(b.executor)}</span></div><div class="batch-meta"><span>${esc(b.model || '')}</span><span>${fmt(done)} / ${fmt(total)} complete</span><span>${fmt(c.failed||0)} failed · ${fmt(c.uncertain||0)} uncertain</span><span>${fmt(c.leased||0)} claimed · ${fmt(c.pending||0)} pending</span></div><progress class="batch-progress" aria-label="Completed jobs" max="100" value="${percent}">${percent}%</progress><button type="button" class="text-button" data-open-batch="${esc(b.id)}">Open jobs & results ↗</button></article>`;
    }).join('') : blank('No batches yet. Freeze the required inputs and create your first assisted draft batch.');
    field('batch-list').querySelectorAll('[data-open-batch]').forEach(b=>b.addEventListener('click',()=>openBatch(b.dataset.openBatch)));
  }
  async function openBatch(id) {
    workflow.selectedBatch=id; const target=field('batch-detail'); target.hidden=false; target.innerHTML=blank('Loading jobs…');
    try {
      const detail=await api(`/api/workflow/batches/${encodeURIComponent(id)}`); if(workflow.selectedBatch!==id)return;
      const b=detail.batch || {}, jobs=list(detail.jobs), failures=jobs.filter(j=>j.status==='failed'), uncertain=jobs.filter(j=>j.status==='uncertain'||j.status==='leased');
      const command=b.executor==='codex' ? `python3 benchmark/workflow.py work ${id} --worker codex --limit 10` : `python3 benchmark/workflow.py run ${id} --execute --limit 10`;
      target.className='batch-detail'; target.innerHTML=`<div class="panel-heading"><div><h3>${esc(b.name || id)}</h3><p class="field-note">${esc(id)}</p></div><button type="button" class="text-button" id="batch-detail-refresh">Refresh ↻</button></div><div class="action-row"><a href="/api/workflow/export/${encodeURIComponent(id)}" class="button secondary" download>Export Codex job packets ↓</a>${(failures.length||uncertain.length)?'<button type="button" class="button secondary" id="batch-retry-failed">Review retries</button>':''}<button type="button" class="button secondary" id="batch-ingest-receipts">Reingest saved receipts</button></div><details class="workflow-details"><summary>${b.executor==='codex'?'Claim a small batch in Codex':'Start API execution locally'}</summary><p class="field-note">${b.executor==='codex'?'Use the local command to claim jobs, then read each image and return results in the supplied schema.':'Set the configured environment variables in your local shell first. Only this explicit command can start live API requests; this page never starts them.'}</p>${code(command)}</details><details class="workflow-details"><summary>Jobs (${fmt(jobs.length)})</summary>${jobs.map((j,index)=>`<div class="job-row"><div class="job-row-head"><span>${esc(j.frame_id)}</span><span class="pill neutral">${esc(j.status)}</span></div><p class="metadata-strip">${esc(j.id)} · ${fmt(Array.isArray(j.attempts)?j.attempts.length:(j.attempts||0))} attempts</p>${j.error?`<p class="job-error">${esc(typeof j.error==='object'?json(j.error):j.error)}</p>`:''}<details data-job-index="${index}"><summary>Job packet</summary><pre class="code-block"></pre></details></div>`).join('')}</details><form id="batch-result-form" class="form-stack workflow-json-submit"><h4>Submit Codex / external results</h4><p class="field-note">Use the exported expected-result schema. Each record needs job_id, result, worker and model_version; include the lease_token when a job was claimed. Assisted GT remains a draft for human review. Inference submissions wrap the exact answer as result.output_text; they do not put a bare string in result.</p><details class="workflow-details"><summary>Submission envelope example</summary>${code(json({records:[{job_id:'from the job packet',worker:'your Codex worker name',model_version:'actual model/version',result:b.task==='inference'?{output_text:'Exact model answer'}:b.task==='ground_truth'?{content:'Use the complete GT content object from the expected schema',issues:[]}:b.task==='gt_check'?{issues:[]}:{claims:[],supported_fact_ids:[],structure_score:0,instruction_score:0,issues:[]}}]}))}<p class="field-note">This is a format example, not a completed assessment. Include lease_token for claimed jobs; use actual evidence and verdicts.</p></details><label>Result JSON / JSONL<input id="batch-result-file" type="file" accept=".json,.jsonl,.ndjson,application/json,text/plain"></label><details class="workflow-details"><summary>Or paste result records</summary><label class="sr-only" for="batch-result-json">Batch result records</label><textarea id="batch-result-json" rows="9" class="code-input" placeholder='{"records":[{"job_id":"…","worker":"codex","model_version":"…","result":{}}]}'></textarea></details><button type="submit" class="button primary">Validate & save results</button></form><div id="batch-submit-receipt" role="status"></div>`;
      target.querySelectorAll('[data-job-index]').forEach(el=>el.addEventListener('toggle',()=>{if(el.open && !el.dataset.loaded){el.querySelector('pre').textContent=json(jobs[Number(el.dataset.jobIndex)]);el.dataset.loaded='true';}}));
      field('batch-detail-refresh').addEventListener('click',()=>openBatch(id));
      field('batch-ingest-receipts').addEventListener('click',()=>busy(field('batch-ingest-receipts'),async()=>{const receipt=await action('ingest_receipts',{batch_id:id});toast('Saved receipts validated; no API request was made.');openDetails('Receipt ingestion',receipt);await Promise.allSettled([load(),loadSummary(),loadFrames(),loadRunsPage()]);await refreshSelectedFrameAfterMutation();await openBatch(id);}));
      field('batch-result-form').addEventListener('submit',event=>{event.preventDefault();busy(field('batch-result-form').querySelector('button[type="submit"]'),async()=>{
        const raw=await inputJSON('batch-result-file','batch-result-json',true);
        const records=Array.isArray(raw)?raw:raw.records; if(!Array.isArray(records)||!records.length)throw new Error('Supply a nonempty result records array.');
        const receipt=await action('submit_results',{batch_id:id,records});
        field('batch-submit-receipt').innerHTML=`<div class="workflow-result">Results saved.${code(json(receipt))}</div>`;
        toast('Results validated and saved.'); await Promise.allSettled([load(),loadSummary(),loadFrames(),loadRunsPage()]); await refreshSelectedFrameAfterMutation();
      });});
      field('batch-retry-failed')?.addEventListener('click',()=>{
        openDetails('Review failed / uncertain jobs',{});
        field('dialog-body').innerHTML=`<p class="field-note">${fmt(failures.length)} failed jobs can return to pending. ${fmt(uncertain.length)} jobs are uncertain or claimed by a worker. First verify that any claimed worker has stopped and inspect its saved receipts. A retry may repeat an earlier API call and incur cost. This action only queues work.</p><form id="retry-form" class="form-stack"><label>Reviewer<input id="retry-reviewer" required></label>${uncertain.length?'<label class="check-line"><input id="retry-uncertain" type="checkbox">I verified the claimed workers have stopped, inspected the attempts, and accept potential duplicate execution for these uncertain or interrupted jobs.</label>':''}<button type="submit" class="button primary">Queue selected retry jobs</button></form>`;
        field('retry-form').addEventListener('submit',event=>{event.preventDefault();busy(field('retry-form').querySelector('button'),async()=>{await action('retry_jobs',{batch_id:id,job_ids:[...failures,...(field('retry-uncertain')?.checked?uncertain:[])].map(j=>j.id),reviewer:value('retry-reviewer'),acknowledge_uncertain:!!field('retry-uncertain')?.checked});field('detail-dialog').close();toast('Retry jobs queued.');await load();await openBatch(id);});});
      });
    } catch(error){showError(target,error,()=>openBatch(id));}
  }
  function renderIssues() {
    const filter=value('issue-filter'), issues=list(workflow.data.issues).filter(i=>filter==='all'||i.status===filter);
    field('issue-list').innerHTML=issues.length?issues.map(i=>`<article class="issue-card ${i.severity==='critical'?'severity-high':''}"><div class="issue-card-head"><h4>${esc(i.category || 'Review observation')}</h4><span class="pill ${i.severity==='critical'?'conflict':'neutral'}">${esc(i.severity)} · ${esc(i.status)}</span></div><p>${esc(i.message)}</p><div class="batch-meta"><button class="text-button" type="button" data-issue-frame="${esc(i.frame_id)}">${esc(i.frame_id)} ↗</button><span class="pill neutral">${i.target==='candidate'?'Candidate output':'Reference evidence'}</span><span>GT revision ${esc(i.gt_revision ?? '—')}</span></div>${i.resolution?`<p class="field-note">Decision: ${esc(typeof i.resolution==='object'?json(i.resolution):i.resolution)}</p>`:''}<details><summary class="field-note">Source and audit record</summary>${code(json(i))}</details>${i.status==='open'?`<form data-issue-id="${esc(i.id)}" class="form-stack"><div class="issue-controls"><label>Reviewer<input name="reviewer" required></label><label>Decision<select name="status"><option value="resolved">Resolved · corrected or verified</option><option value="dismissed">Dismissed · not applicable</option></select></label><button type="submit" class="button secondary">Save decision</button></div><label>Reason / evidence<input name="resolution" required placeholder="Explain the correction or why this flag does not apply"></label></form>`:''}</article>`).join(''):blank(filter==='open'?'No open review issues. This does not certify factual correctness.':'No issues in this state.');
    field('issue-list').querySelectorAll('[data-issue-frame]').forEach(b=>b.addEventListener('click',async()=>{switchPage('frames');await selectFrame(b.dataset.issueFrame);}));
    field('issue-list').querySelectorAll('[data-issue-id]').forEach(form=>form.addEventListener('submit',event=>{event.preventDefault();busy(form.querySelector('button'),async()=>{const f=new FormData(form);await action('resolve_issue',{id:form.dataset.issueId,status:f.get('status'),reviewer:f.get('reviewer'),resolution:f.get('resolution')});toast('Review decision saved.');await load();await refreshSelectedFrameAfterMutation();});}));
  }
  function renderCalibration() {
    const reports=list(workflow.data.calibrations);
    field('calibration-report').innerHTML=reports.length?reports.slice().reverse().map(r=>{const report=r.report||r;return `<article class="version-card"><div class="batch-card-head"><h4>${esc(report.judge?.model || 'Judge calibration')}</h4><span class="pill draft">${esc(report.status || 'Recorded evidence')}</span></div><p class="field-note">${fmt(report.n ?? r.records_count)} observations · ${fmt(report.frame_count)} frames · ${fmt(report.recording_count)} recordings. This report does not certify judge reliability.</p><div class="table-wrap"><table><thead><tr><th>Split</th><th>Observations</th><th>Recordings</th></tr></thead><tbody>${Object.entries(report.splits||{}).map(([split,metrics])=>`<tr><td>${esc(split)}</td><td>${fmt(metrics.n)}</td><td>${fmt(metrics.recording_count)}</td></tr>`).join('')}</tbody></table></div><details class="workflow-details"><summary>Agreement, disagreements & limitations</summary>${code(json(report))}</details></article>`;}).join(''):blank('No calibration evidence imported. Automated scores remain provisional.');
  }
  function renderReleases() {
    field('release-list').innerHTML=list(workflow.data.releases).length?list(workflow.data.releases).slice().reverse().map(r=>`<article class="version-card"><div class="batch-card-head"><h4>${esc(r.name || r.id)}</h4><span class="pill ${r.complete?'approved':'draft'}">${r.complete?'Complete reference set':'Partial snapshot'}</span></div><p class="field-note">${fmt(r.frame_count)} frames · ${esc(r.created_at || '')}</p><p class="metadata-strip">SHA-256 ${esc(r.sha256 || '')}</p><a class="text-button" href="/api/workflow/releases/${encodeURIComponent(r.id)}" download>Download release manifest ↓</a><details class="workflow-details"><summary>Export a portable archive</summary><p class="field-note">This local command packages the snapshot and its image objects into a ZIP archive.</p>${code(`python3 benchmark/workflow.py export-release ${r.id} --output ${String(r.id).replace(/[^a-zA-Z0-9_-]/g,'_')}.zip`)}</details><details class="workflow-details"><summary>Snapshot metadata</summary>${code(json(r))}</details></article>`).join(''):blank('No release snapshots yet.');
  }
  function renderEvidenceLinks() {
    field('evidence-frame').value=state.selected || '';
    const links=list(workflow.data.evidence_links).filter(link=>!state.selected || link.frame_id===state.selected);
    field('evidence-link-list').innerHTML=links.length?links.map(link=>`<details class="workflow-details"><summary>${esc(link.frame_id)} · ${fmt(list(link.rating_ids).length)} ratings linked · ${link.input_verified?'input verified':'output link only'}</summary>${code(json(link))}</details>`).join(''):blank('No verified evidence links for the selected frame.');
  }
  function updateCalibrationTemplate() {
    const run=value('calibration-template-run'),link=field('calibration-template-export');link.hidden=!run;
    if(run)link.href=`/api/workflow/calibration-template?run_id=${encodeURIComponent(run)}`;else link.removeAttribute('href');
  }
  async function inputJSON(fileId,textId,allowJSONL=false) {
    const file=field(fileId).files[0]; if(file?.size>24_000_000)throw new Error('Choose a file smaller than 24 MB.');
    const text=file?await file.text():value(textId); if(!text)throw new Error('Choose a file or paste JSON.');
    try{return JSON.parse(text);}catch{if(allowJSONL){try{return text.split(/\r?\n/).filter(v=>v.trim()).map(v=>JSON.parse(v));}catch{}}throw new Error('The supplied result is not valid JSON'+(allowJSONL?' or JSONL.':'.'));}
  }
  function updateTaskControls() {
    const task = value('batch-task'), scope = value('batch-scope');
    field('batch-run-label').hidden=task!=='judge';
    field('batch-evidence-label').hidden=task!=='ground_truth';
    field('batch-scope').querySelector('option[value="selected"]').textContent = state.selected ? `Selected frame · ${state.selected}` : 'Selected frame · none selected';
    field('batch-scope').querySelector('option[value="filtered"]').textContent = `Current catalog filter · ${fmt(state.frames.length)} frames`;
    const scopeText = scope==='selected' ? (state.selected || 'No frame selected') : scope==='filtered' ? `${fmt(state.frames.length)} frames in the current catalog filter` : `All ${fmt(workflow.data?.readiness?.frames ?? state.frames.length)} frames`;
    const taskText = field('batch-task').selectedOptions[0].textContent;
    field('batch-context-note').textContent = `${taskText}: ${scopeText}. Review the settings below before creating the batch. Creating a batch prepares work; it does not call a model.`;
  }
  function updateExecutorControls() {
    const executor=value('batch-executor');field('batch-api-fields').hidden=executor==='codex';
    field('batch-key-env').value=executor==='azure'?'AZURE_OPENAI_API_KEY':'OPENAI_API_KEY';
    field('batch-base-url').value='';field('batch-base-url').placeholder=executor==='azure'?'https://YOUR-RESOURCE.openai.azure.com/openai/v1':'https://api.openai.com/v1';
  }
  function addRubricSelect(beforeId,id,label) {(field(beforeId).closest('label') || field(beforeId)).insertAdjacentHTML('beforebegin',`<label>${esc(label)}<select id="${id}" required><option value="">Choose a rubric version</option></select></label>`);}
  function initWorkflow() {
    addRubricSelect('batch-run-label','batch-rubric','Rubric version');
    addRubricSelect('calibration-file','calibration-rubric','Rubric being calibrated');
    addRubricSelect('release-name','release-rubric','Frozen rubric for this release');
    field('workflow-refresh').addEventListener('click',async()=>{await Promise.allSettled([load(),loadSummary(),loadRuns()]);updateRuns();});
    field('rubric-json').addEventListener('input',()=>{workflow.rubricDirty=true;field('rubric-state').textContent='Unsaved edits';field('rubric-freeze').disabled=true;renderRubricPreview();});
    field('bulk-input-form').addEventListener('submit',event=>{event.preventDefault();if(!confirmDiscard())return;busy(event.submitter,async()=>{const receipt=await action('freeze_recorded_inputs',{frame_ids:selectedFrames(value('bulk-input-scope')),reviewer:value('bulk-input-reviewer'),overwrite:field('bulk-input-replace').checked});field('bulk-input-result').textContent=`${fmt(receipt.saved)} input versions saved; ${fmt(receipt.skipped)} existing inputs kept.`;toast('Input versions saved.');await Promise.allSettled([load(),loadFrames(),loadSummary()]);if(state.selected)await selectFrame(state.selected,true);});});
    field('rubric-form').addEventListener('submit',event=>{event.preventDefault();busy(event.submitter,async()=>{const content=parseJSON(value('rubric-json'),'Rubric');const r=await action('save_rubric',{content,author:value('rubric-reviewer'),label:value('rubric-notes')||content.id});workflow.rubricDirty=false;workflow.rubricId=r.id || r.rubric?.id || null;await load();setRubric(list(workflow.data.rubrics).find(v=>v.id===workflow.rubricId)||latestRubric());toast('New rubric draft saved.');});});
    field('rubric-freeze').addEventListener('click',async()=>{await busy(field('rubric-freeze'),async()=>{if(!value('rubric-reviewer'))throw new Error('Enter the rubric reviewer name.');if(!workflow.rubricId||workflow.rubricDirty)throw new Error('Save the exact rubric draft before freezing.');await action('freeze_rubric',{id:workflow.rubricId,reviewer:value('rubric-reviewer')});await load();toast('Rubric version frozen.');});const current=list(workflow.data?.rubrics).find(r=>r.id===workflow.rubricId);field('rubric-freeze').disabled=!current||current.status==='frozen'||workflow.rubricDirty;});
    field('batch-task').addEventListener('change',updateTaskControls);field('batch-executor').addEventListener('change',updateExecutorControls);
    field('batch-form').addEventListener('submit',event=>{event.preventDefault();busy(event.submitter,async()=>{
      const executor=value('batch-executor'),model=value('batch-model'),settings=parseJSON(value('batch-settings')||'{}','Generation settings');
      if(!settings||Array.isArray(settings)||typeof settings!=='object')throw new Error('Generation settings must be a JSON object.');
      const config={provider:executor,model,settings};
      if(executor!=='codex'){if(!/^[A-Za-z_][A-Za-z0-9_]*$/.test(value('batch-key-env')))throw new Error('Credential field accepts an environment variable name only.');config.api_key_env=value('batch-key-env');if(value('batch-base-url'))config.base_url=value('batch-base-url');if(executor==='azure'&&!config.base_url)throw new Error('Enter your Azure v1 API base URL.');}
      const body={task:value('batch-task'),frame_ids:selectedFrames(value('batch-scope')),executor,model,name:value('batch-name'),reviewer:value('batch-operator'),rubric_id:selectedRubric('batch-rubric'),provider_config:config};
      if(body.task==='ground_truth')body.evidence_mode=value('batch-evidence-mode');
      if(body.task==='judge'){body.run_id=value('batch-run');if(!body.run_id)throw new Error('Choose a prediction run to judge.');}
      const result=await action('create_batch',body),id=result.id||result.batch?.id;
      const summary=result.batch || result;field('batch-create-result').innerHTML=`<div class="workflow-result">Batch created. No model call was made.${code(json({id,name:summary.name,task:summary.task,executor:summary.executor,model:summary.model,counts:summary.counts,job_count:summary.job_count || result.job_count}))}</div>`;toast('Batch created.');await load();if(id)await openBatch(id);
    });});
    field('calibration-template-run').addEventListener('change',updateCalibrationTemplate);
    field('evidence-link-form').addEventListener('submit',event=>{event.preventDefault();busy(event.submitter,async()=>{
      const frame_id=value('evidence-frame');if(!frame_id)throw new Error('Select a frame in Ground truth first.');
      const text=value('evidence-rating-ids'), rating_ids=text.startsWith('[')?parseJSON(text,'Rating IDs'):text.split(/[\s,]+/).filter(Boolean);
      if(!Array.isArray(rating_ids)||!rating_ids.length||rating_ids.some(id=>typeof id!=='string'))throw new Error('Supply rating IDs separated by commas/newlines, or a JSON string array.');
      const body={frame_id,rating_ids:[...new Set(rating_ids)],run_id:value('evidence-run'),reviewer:value('evidence-reviewer'),reason:value('evidence-reason'),human_confirmed:field('evidence-human-confirmed').checked,input_verified:field('evidence-input-verified').checked};
      if(value('evidence-other-run'))body.other_run_id=value('evidence-other-run');
      await action('verify_evidence_link',body);field('evidence-human-confirmed').checked=false;field('evidence-input-verified').checked=false;toast('Verified evidence link saved.');await load();
    });});
    field('issue-filter').addEventListener('change',()=>{if(workflow.data)renderIssues();});
    field('calibration-form').addEventListener('submit',event=>{event.preventDefault();busy(event.submitter,async()=>{const raw=await inputJSON('calibration-file','calibration-json');const records=Array.isArray(raw)?raw:raw.records;if(!Array.isArray(records)||!records.length)throw new Error('Calibration JSON needs a nonempty records array.');await action('import_calibration',{...(Array.isArray(raw)?{}:raw),rubric_id:selectedRubric('calibration-rubric'),records});toast('Calibration evidence imported.');await load();});});
    field('release-form').addEventListener('submit',event=>{event.preventDefault();busy(event.submitter,async()=>{await action('create_release',{name:value('release-name'),rubric_id:selectedRubric('release-rubric'),reviewer:value('release-reviewer'),notes:value('release-notes'),frame_ids:[],require_complete:field('release-complete').checked});toast('Release snapshot created locally.');await load();});});
    window.addEventListener('beforeunload',event=>{if(workflow.rubricDirty){event.preventDefault();event.returnValue='';}});
  }
  window.MousebenchWorkflow={load,openTask}; initWorkflow();
  field('batch-scope').addEventListener('change',updateTaskControls);
})();
