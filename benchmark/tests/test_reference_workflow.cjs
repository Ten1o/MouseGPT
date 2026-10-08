const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const referenceEdit = require('../web/reference-edit.js');

const fields = ['Overall', 'Head', 'Limb', 'Torso', 'Others', 'Keywords'];
const source = fs.readFileSync(path.join(__dirname, '../web/app.js'), 'utf8');
assert.match(source, /\ninit\(\);\s*$/);

function node(value = '') {
  return {value, textContent:'', innerHTML:'', checked:false, disabled:false, isConnected:true,
    focus() {}, addEventListener() {}};
}

function harness() {
  const elements = new Map([
    ['#gt-status', node('draft')], ['#gt-reviewer', node()],
    ['#gt-notes', node('Saved review note')],
    ['#save-state', node()], ['#save-gt', node()], ['#gt-fields', node()],
    ['#edit-reference', node()], ['#editor-body', node()], ['#frame-review-issues', node()],
    ['#manage-frame-issues', node()], ['#retry-frame-issues', node()],
    ['#gt-gate-note', node()], ['#approve-gt', node()],
    ['#frame-search', node()], ['#status-filter', node('draft')], ['#collection-filter', node('v1.0')]
  ]);
  const sectionInputs = fields.map(field => ({...node(`${field} detail.`), dataset:{section:field}}));
  const uncertaintyInputs = [node('A saved uncertainty.')];
  const sectionButtons = fields.map(() => node());
  const querySelectorAll = selector => selector === '[data-section]' ? sectionInputs
    : selector === '[data-edit-reference]' ? sectionButtons
    : selector === '[data-uncertainty]' ? uncertaintyInputs : [];
  const document = {
    querySelector:selector => selector === '[data-section]' ? sectionInputs[0] : elements.get(selector) || null,
    querySelectorAll
  };
  elements.get('#editor-body').querySelectorAll = querySelectorAll;
  const toasts = [], requests = [];
  const context = vm.createContext({document, structuredClone, console,
    window:{MousebenchReferenceEdit:referenceEdit},
    recordToast:(...args) => toasts.push(args),
    recordRequest:(...args) => requests.push(args),
    fetch:() => { throw new Error('Network access is forbidden in this test.'); }});
  // Keep the real state/capture/transition functions, replacing rendering only for
  // edit-state tests. Issue rendering below still runs the production renderer.
  vm.runInContext(source.replace(/\ninit\(\);\s*$/, '') + `
    toast = (...args) => recordToast(...args);
    renderEditor = () => {};
    api = async (...args) => { recordRequest(...args); throw new Error('Unexpected API request'); };
    globalThis.referenceWorkflow = {state, startReferenceEdit, finishReferenceEdit,
      captureEditor, renderFrameReviewIssues, saveGroundTruth, referenceActionReason, referenceSubmitStatus, humanReviewerName, renderGate};
  `, context, {filename:'app.js'});
  const controls = context.referenceWorkflow, state = controls.state;
  const content = {sections:Object.fromEntries(sectionInputs.map(input => [input.dataset.section, input.value])),
    facts:[{id:'fixture-fact',text:'Observable fact',field:'Overall',evidence:'image',required:true}], uncertainties:uncertaintyInputs.map(input => input.value)};
  content.reference_description = referenceEdit.summaryFromSections(content.sections);
  state.selected = 'fixture-frame';
  state.detail = {ground_truth:{revision:2, status:'draft', content:structuredClone(content), input_hash:'input-fixture', content_sha256:'content-fixture'}, input_contract:{input_hash:'input-fixture'}};
  state.editContent = structuredClone(content);
  state.reviewIssues = {frameId:'fixture-frame', revision:2, contentHash:'content-fixture', inputHash:'input-fixture', items:[]};
  return {controls, state, elements, sectionInputs, sectionButtons, toasts, requests, context};
}

function approvalHarness(frames, {failSave=false, failRefresh=false, onSave=()=>{}}={}) {
  const h=harness(), selected=[], focused=[];
  h.state.page='frames';
  h.state.frames=[{id:'a',gt_status:'draft'},{id:'fixture-frame',gt_status:'draft'},{id:'z',gt_status:'draft'}];
  Object.assign(h.context, {
    saveAPI:async(url,payload)=>{
      h.requests.push([url,payload]);
      if (failSave) throw new Error('Approval rejected');
      onSave(h);
      return {};
    },
    refreshCatalog:async()=>{if(failRefresh)return; h.state.frames=frames; return frames;},
    selectSavedFrame:async id=>{selected.push(id); h.state.selected=id; h.sectionButtons.forEach(button=>button.disabled=h.state.savingReference); return true;},
    focusSavedFrame:where=>focused.push(where)
  });
  vm.runInContext(`api=saveAPI; loadFrames=refreshCatalog; loadSummary=async()=>{};
    selectFrame=selectSavedFrame; focusReferenceStart=()=>focusSavedFrame('start');
    focusReferenceApproval=()=>focusSavedFrame('approval');`,h.context);
  h.submit=async(status='approved')=>h.controls.saveGroundTruth({preventDefault(){},submitter:{...node(),dataset:{gtStatus:status}}});
  return {...h,selected,focused};
}

test('approval advances to the next fresh draft and starts its review at the top',async()=>{
  const h=approvalHarness([{id:'a',gt_status:'draft'},{id:'fixture-frame',gt_status:'approved'},{id:'z',gt_status:'draft'}]);
  await h.submit();
  assert.deepEqual(h.selected,['z']);
  assert.deepEqual(h.focused,['start']);
  assert.equal(h.requests.length,1);
  assert.equal(h.state.dirty,false);
  assert.ok(h.sectionButtons.every(button=>!button.disabled));
  assert.ok(h.toasts.some(([message])=>message.includes('Opened next draft: z')));
});

test('approval wraps to an earlier draft, skipping approved or input-stale references',async()=>{
  const h=approvalHarness([{id:'a',gt_status:'draft'},{id:'z',gt_status:'needs_review'}]);
  await h.submit();
  assert.deepEqual(h.selected,['a']);
});

test('the last draft stays visible with a completion message',async()=>{
  const h=approvalHarness([]);
  await h.submit();
  assert.deepEqual(h.selected,['fixture-frame']);
  assert.deepEqual(h.focused,['approval']);
  assert.ok(h.toasts.some(([message])=>message.includes('No drafts remain in the current filters')));
});

test('failed approval never navigates or reports success, and draft save does not advance',async()=>{
  const failed=approvalHarness([{id:'z',gt_status:'draft'}],{failSave:true});
  failed.state.dirty=true;
  await failed.submit();
  assert.deepEqual(failed.selected,[]);
  assert.equal(failed.state.dirty,true);
  assert.ok(failed.toasts.some(([message])=>message==='Approval rejected'));
  const draft=approvalHarness([{id:'z',gt_status:'draft'}]);
  await draft.submit('draft');
  assert.deepEqual(draft.selected,['fixture-frame']);
  assert.deepEqual(draft.focused,[]);
});

test('a failed queue refresh does not advance using previously loaded drafts',async()=>{
  const h=approvalHarness([],{failRefresh:true});
  await h.submit();
  assert.deepEqual(h.selected,['fixture-frame']);
  assert.ok(h.toasts.some(([message])=>message.includes('catalog could not be refreshed')));
});

test('navigation or filter changes during approval take precedence over automatic advance',async t=>{
  for(const change of ['frame','page','filter','collection'])await t.test(change,async()=>{
    const h=approvalHarness([{id:'z',gt_status:'draft'}],{onSave:h=>{
      if(change==='frame'){h.state.selected='a'; h.state.generation++; h.state.dirty=true;}
      if(change==='page')h.state.page='runs';
      if(change==='filter')h.elements.get('#status-filter').value='approved';
      if(change==='collection')h.elements.get('#collection-filter').value='v1.1';
    }});
    await h.submit();
    assert.ok(!h.selected.includes('z'));
    if(change==='frame'){assert.deepEqual(h.selected,[]);assert.equal(h.state.dirty,true);}
  });
});

test('input edits made on another page while approval is saving survive its refresh',async()=>{
  const h=approvalHarness([{id:'z',gt_status:'draft'}],{onSave:h=>{
    h.state.page='protocol'; h.state.contractDirty=true;
  }});
  await h.submit();
  assert.deepEqual(h.selected,[]);
  assert.equal(h.state.contractDirty,true);
  assert.equal(h.state.detailStale,true);
});

test('Apply stays local, and a later Cancel restores the previously applied unsaved content', () => {
  const h = harness(), saved = JSON.stringify(h.state.detail.ground_truth.content);
  h.controls.startReferenceEdit();
  h.sectionInputs[0].value = 'First applied correction.';
  h.controls.finishReferenceEdit(true);
  const applied = JSON.stringify(h.state.editContent);
  assert.equal(h.state.editContent.sections.Overall, 'First applied correction.');
  assert.equal(h.state.dirty, true);
  assert.equal(h.state.editingReference, false);
  assert.equal(JSON.stringify(h.state.detail.ground_truth.content), saved);
  assert.equal(h.requests.length, 0);

  h.controls.startReferenceEdit();
  h.sectionInputs[0].value = 'Second correction to cancel.';
  h.controls.captureEditor(); // Add/remove controls can capture before Cancel.
  h.elements.get('#gt-notes').value = 'A new review note remains.';
  h.controls.finishReferenceEdit(false);
  assert.equal(JSON.stringify(h.state.editContent), applied);
  assert.equal(h.state.dirty, true);
  assert.equal(h.state.editSession, null);
  assert.equal(h.elements.get('#gt-notes').value, 'A new review note remains.');
  assert.equal(h.elements.get('#save-state').textContent, 'Unsaved changes');
  assert.equal(h.requests.length, 0);
});

test('Cancel preserves footer-only changes while a clean Cancel restores clean state', () => {
  const h = harness(), saved = JSON.stringify(h.state.editContent);
  h.controls.startReferenceEdit();
  h.sectionInputs[0].value = 'Discard this local change.';
  h.controls.captureEditor();
  h.controls.finishReferenceEdit(false);
  assert.equal(JSON.stringify(h.state.editContent), saved);
  assert.equal(h.state.dirty, false);
  assert.equal(h.elements.get('#save-state').textContent, 'Saved reference loaded');

  h.controls.startReferenceEdit();
  h.elements.get('#gt-reviewer').value = 'Another reviewer';
  h.elements.get('#gt-notes').value = 'Footer-only correction';
  h.controls.finishReferenceEdit(false);
  assert.equal(JSON.stringify(h.state.editContent), saved);
  assert.equal(h.state.dirty, true);
  assert.equal(h.elements.get('#gt-reviewer').value, 'Another reviewer');
  assert.equal(h.elements.get('#gt-notes').value, 'Footer-only correction');
});

test('critical and annotated uncertainty issues retain visible severity and details', () => {
  const h = harness();
  const common = {category:'uncertainty', severity:'warning', message:'A saved uncertainty.'};
  h.state.reviewIssues = {items:[
    {...common}, {...common, severity:'critical'},
    {...common, suggested_fix:'Check the second view.'},
    {...common, evidence:'The paw is occluded.'},
    {...common, fact_ids:['fact-7']}
  ]};
  h.controls.renderFrameReviewIssues();
  const html = h.elements.get('#frame-review-issues').innerHTML;
  assert.equal((html.match(/<article class="reference-issue">/g) || []).length, 4);
  assert.ok(html.includes('review-issue-severity critical'));
  assert.ok(html.includes('Check the second view.'));
  assert.ok(html.includes('The paw is occluded.'));
  assert.ok(html.includes('fact-7'));
  assert.ok(html.includes('1 issue is the uncertainty note shown above.'));
});

test('editing, in-flight saving, unavailable issues and stale evidence guard saves before any request', async t => {
  const cases = [
    {name:'editing', state:{editingReference:true}, message:'Apply or cancel'},
    {name:'saving', state:{savingReference:true}},
    {name:'unsaved input', state:{contractDirty:true}, message:'input contract edits'},
    {name:'approval issues loading', state:{reviewIssues:null}, status:'approved', message:'Load the current review issues'},
    {name:'approval issues failed', state:{reviewIssues:{error:'Unavailable'}}, status:'approved', message:'Load the current review issues'},
    {name:'stale approval', state:{detailStale:true}, status:'approved', message:'Reload the latest saved evidence'}
  ];
  for (const scenario of cases) await t.test(scenario.name, async () => {
    const h = harness();
    Object.assign(h.state, scenario.state);
    if (scenario.status) h.elements.get('#gt-status').value = scenario.status;
    let prevented = false;
    await h.controls.saveGroundTruth({preventDefault() { prevented = true; }, submitter:{dataset:{gtStatus:scenario.status || 'draft'}}});
    assert.equal(prevented, true);
    assert.equal(h.requests.length, 0);
    if (scenario.message) assert.ok(h.toasts.some(([text]) => text.includes(scenario.message)));
  });
});

test('an in-flight save prevents starting or finishing an edit session', () => {
  const h = harness();
  h.state.savingReference = true;
  h.controls.startReferenceEdit();
  assert.equal(h.state.editingReference, false);
  assert.equal(h.state.editSession, null);
  const before = JSON.stringify(h.state.editContent);
  h.controls.finishReferenceEdit(true);
  assert.equal(JSON.stringify(h.state.editContent), before);
  assert.equal(h.requests.length, 0);
});


test('submit buttons select explicit transitions; implicit submissions can only save a draft', () => {
  const {controls} = harness();
  for (const status of ['draft','approved']) {
    assert.equal(controls.referenceSubmitStatus({submitter:{dataset:{gtStatus:status}}}),status);
  }
  for (const event of [{}, {submitter:{}}, {submitter:{dataset:{gtStatus:'unexpected'}}}, {submitter:{dataset:{gtStatus:'reviewed'}}}]) {
    assert.equal(controls.referenceSubmitStatus(event),'draft');
  }
});

test('approval checks issues bound to the loaded saved reference and input', async t => {
  const cases = [
    ['changed input without reloaded issues', h=>h.state.detail.input_contract.input_hash='new-input', 'Load the current review issues'],
    ['wrong frame issues', h=>h.state.reviewIssues.frameId='other-frame', 'Load the current review issues'],
    ['wrong revision issues', h=>h.state.reviewIssues.revision=1, 'Load the current review issues'],
    ['wrong content issues', h=>h.state.reviewIssues.contentHash='old-content', 'Load the current review issues'],
    ['wrong input issues', h=>h.state.reviewIssues.inputHash='old-input', 'Load the current review issues'],
    ['critical issue', h=>h.state.reviewIssues.items.push({severity:'critical'}), 'Resolve the open critical'],
    ['already approved', h=>h.state.detail.ground_truth.status='approved', 'already approved']
  ];
  for (const [name, mutate, expected] of cases) await t.test(name,async()=>{
    const h=harness(); mutate(h);
    assert.ok(h.controls.referenceActionReason('approved').includes(expected));
    await h.controls.saveGroundTruth({preventDefault(){},submitter:{dataset:{gtStatus:'approved'}}});
    assert.equal(h.requests.length,0);
  });
  const h=harness();
  assert.equal(h.controls.referenceActionReason('approved'),'');
  h.state.reviewIssues.items.push({severity:'warning'});
  assert.equal(h.controls.referenceActionReason('approved'),'');
});

test('approval readiness requires six sections, required facts and a saved input', () => {
  const h=harness();
  assert.equal(h.controls.referenceActionReason('approved'),'');
  h.state.editContent.sections.Head='   ';
  assert.match(h.controls.referenceActionReason('approved'), /six sections/);
  h.state.editContent.sections.Head='Head detail';
  h.state.editContent.facts=[];
  assert.match(h.controls.referenceActionReason('approved'), /required grading fact/);
  h.state.detail.input_contract=null;
  assert.match(h.controls.referenceActionReason('approved'), /recorded input/);
  assert.equal(h.controls.referenceActionReason('draft'),'');
});

test('a saved draft and applied local edits can be approved without a separate review step', () => {
  const h=harness();
  assert.equal(h.state.detail.ground_truth.status,'draft');
  assert.equal(h.elements.get('#gt-reviewer').value,'');
  h.controls.renderGate();
  assert.equal(h.elements.get('#approve-gt').disabled,false);
  assert.match(h.elements.get('#gt-gate-note').textContent,/one click/);
  h.state.editingReference=true;
  h.controls.renderGate();
  assert.equal(h.elements.get('#approve-gt').disabled,true);
  assert.equal(h.elements.get('#save-gt').disabled,true);
  h.state.editingReference=false;
  h.state.editContent.sections.Head='Applied correction for approval.';
  h.state.dirty=true;
  h.controls.renderGate();
  assert.equal(h.elements.get('#approve-gt').disabled,false);
  assert.equal(h.elements.get('#save-gt').disabled,false);
});

test('assistant reviewer labels are not reused as a human signature', () => {
  const {controls}=harness();
  for (const reviewer of ['Codex','Codex GPT-6 Astra','Assistant draft','GPT-6 Astra']) {
    assert.equal(controls.humanReviewerName({reviewer}), '');
  }
  assert.equal(controls.humanReviewerName({reviewer:'Teng Xu',provenance:{actor_type:'assistant'}}),'Teng Xu');
  assert.equal(controls.humanReviewerName(null),'');
});

test('one explicit approval request records the click with optional attribution and current input', async t => {
  for (const status of ['draft','approved']) await t.test(status,async()=>{
    const h=harness();
    const submitter={...node(status),dataset:{gtStatus:status}};
    await h.controls.saveGroundTruth({preventDefault(){},submitter});
    assert.equal(h.requests.length,1);
    const [url,payload]=h.requests[0];
    assert.equal(url,'/api/frames/fixture-frame/ground-truth');
    assert.equal(payload.status,status);
    assert.equal(payload.expected_revision,2);
    assert.equal(payload.actor_type,'human');
    assert.equal(payload.reviewer,'');
    assert.equal(payload.human_confirmed,status==='approved');
    assert.equal(payload.approval_action,status==='approved'?'approve_reference':undefined);
    assert.equal(payload.expected_input_hash,status==='approved'?'input-fixture':undefined);
    assert.equal(JSON.stringify(payload.content),JSON.stringify(h.state.detail.ground_truth.content));
    assert.equal(h.state.savingReference,false);
  });
});

test('approval submits applied content and optional notes without silently altering attribution', async () => {
  const h=harness();
  h.state.editContent.sections.Head='A human correction.';
  h.elements.get('#gt-reviewer').value='  Teng Xu  ';
  h.elements.get('#gt-notes').value='Verified head direction.';
  await h.controls.saveGroundTruth({preventDefault(){},submitter:{...node(),dataset:{gtStatus:'approved'}}});
  assert.equal(h.requests.length,1);
  assert.equal(h.requests[0][1].content.sections.Head,'A human correction.');
  assert.equal(h.requests[0][1].reviewer,'Teng Xu');
  assert.equal(h.requests[0][1].notes,'Verified head direction.');
});

test('implicit submission can only save a draft even when a prior explicit state was approved', async () => {
  const h=harness();
  h.elements.get('#gt-status').value='approved';
  await h.controls.saveGroundTruth({preventDefault(){}});
  assert.equal(h.requests.length,1);
  const payload=h.requests[0][1];
  assert.equal(payload.status,'draft');
  assert.equal(payload.human_confirmed,false);
  assert.equal(payload.approval_action,undefined);
});
