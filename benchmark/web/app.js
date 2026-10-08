'use strict';

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const FIELDS = ['Overall', 'Head', 'Limb', 'Torso', 'Others', 'Keywords'];
const state = {token: null, frames: [], detail: null, selected: null, page: 'overview', runs: [], runTrack: 'controlled', scorecards: null, summary: null, editingReference: false, savingReference: false, editSession: null, reviewIssues: null, evidenceTab: 'descriptions', dirty: false, contractDirty: false, generation: 0, searchGeneration: 0, preview: null, detailStale: false};
const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[char]));
const json = value => JSON.stringify(value, null, 2);
const arr = value => Array.isArray(value) ? value : [];
const fmt = value => typeof value === 'number' ? new Intl.NumberFormat('en').format(value) : '—';
const pct = value => typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '—';
const display = value => value == null ? '—' : typeof value === 'object' ? JSON.stringify(value) : String(value);
const statusName = value => ({not_started:'No reference', unreviewed:'No reference', draft:'Draft', reviewed:'Reviewed', approved:'Approved', needs_review:'Input changed — review needed'}[value] || value || 'No reference');
const badge = value => `<span class="pill ${['draft','reviewed','approved'].includes(value) ? value : 'neutral'}">${esc(statusName(value))}</span>`;
const modelImageAssets = detail => arr(detail?.assets).filter(asset => asset.role === 'model_output_image');
const materializedAsset = asset => Boolean(asset?.path && asset?.sha256);
const collectionBadges = frame => arr(frame?.tags).map(tag => `<span class="pill collection-tag" title="Dataset collection tag">${esc(tag)}</span>`).join('');
const empty = text => `<p class="empty compact">${esc(text)}</p>`;
let toastTimer;

function toast(message, error = false) {
  const el = $('#toast'); el.textContent = message; el.classList.toggle('error', error); el.hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { el.hidden = true; }, error ? 11000 : 5000);
}

async function api(path, body) {
  // Snapshot the request before any await so reconnecting preserves the exact
  // content, revision and approval action the reviewer submitted.
  const serializedBody = body === undefined ? undefined : JSON.stringify(body);
  async function renewSession() {
    const session = await api('/api/session');
    if (typeof session.token !== 'string' || !session.token.trim()) throw new Error('Could not reconnect to the workbench. Your edits are still here; try saving again.');
    state.token = session.token;
  }
  if (body !== undefined && !state.token) await renewSession();
  const options = {headers: {'Accept':'application/json'}};
  if (body !== undefined) { options.method = 'POST'; options.headers['Content-Type'] = 'application/json'; options.body = serializedBody; }
  for (let attempt = 0; attempt < 2; attempt++) {
    if (body !== undefined) options.headers['X-Mousebench-Token'] = state.token;
    const response = await fetch(path, options);
    const text = await response.text();
    let result; try { result = text ? JSON.parse(text) : {}; } catch { throw new Error(`The server returned an unreadable response (${response.status}).`); }
    const sessionExpired = body !== undefined && response.status === 403 && (
      result.code === 'session_token_expired' ||
      // Existing server processes use this exact pre-write rejection message.
      (!result.code && result.error === 'Reload the workbench before saving.')
    );
    if (sessionExpired && attempt === 0) {
      // This guard rejects before dispatch. Never replay an ambiguous failure
      // such as a disconnected response, server error or revision conflict.
      state.token = null;
      await renewSession();
      continue;
    }
    if (!response.ok) {
      const error = new Error(sessionExpired ? 'Could not reconnect to the workbench. Your edits are still here; try saving again.' : result.error || `Request failed (${response.status}).`);
      error.status = response.status; throw error;
    }
    return result;
  }
}

async function busy(button, action) {
  const original = button.textContent; button.disabled = true; button.textContent = 'Working…';
  try { return await action(); } catch (error) { toast(error.message, true); }
  finally { if (button.isConnected) { button.disabled = false; button.textContent = original; } }
}

function showError(target, error, retry) {
  target.innerHTML = `<div class="error-card"><h3>Unable to load this view</h3><p>${esc(error.message)}</p><button class="button secondary">Try again</button></div>`;
  $('button', target).addEventListener('click', retry);
}

async function loadSummary() {
  try {
    const data = await api('/api/summary'); state.summary = data;
    renderCollectionControls(data);
    const metrics = [
      ['Canonical frames · 样本', data.frames, 'Capture · group · frame'],
      ['Local images', data.local_images, 'Original image bytes'],
      ['Expert source rows', data.ratings, 'Includes archived copies'],
      ['Reference drafts', data.gt_drafts, `${fmt(data.gt_reviewed)} reviewed`],
      ['Approved references', data.gt_approved, 'Eligible for assessment'],
      ['Prediction runs', data.runs, 'Historical + controlled']
    ];
    $('#metrics').innerHTML = metrics.map(([label, value, note], i) => `<div class="metric ${i === 4 ? 'highlight' : ''}"><span>${esc(label)}</span><strong>${fmt(value)}</strong><small>${esc(note)}</small></div>`).join('');
    $('#provider-status').textContent = 'Codex assisted · API execution by explicit command';
  } catch (error) { $('#metrics').innerHTML = `<div class="metric"><span>Overview unavailable</span><strong>—</strong><small>${esc(error.message)}</small></div>`; }
}

function renderCollectionControls(data) {
  const filter = $('#collection-filter');
  const selected = filter.value;
  const tags = arr(data.collection_tags);
  filter.innerHTML = `<option value="">All collections (${fmt(data.frames)})</option>` + tags.map(item => `<option value="${esc(item.tag)}">${esc(item.tag)} · ${fmt(item.frames)} frames</option>`).join('');
  // A summary refresh must not silently change an active review filter.
  if (selected && !tags.some(item => item.tag === selected)) filter.insertAdjacentHTML('beforeend', `<option value="${esc(selected)}">${esc(selected)}</option>`);
  filter.value = selected;
  for (const id of ['bulk-input-scope','batch-scope']) {
    const option = $(`#${id} option[value="all"]`);
    if (option) option.textContent = `All ${fmt(data.frames)} frames · all collections`;
  }
}

async function browseCollection(tag) {
  $('#frame-search').value = '';
  $('#status-filter').value = '';
  $('#collection-filter').value = tag;
  switchPage('frames');
  // Only the catalog changes. Keep any open reference and its unsaved edits intact.
  await loadFrames();
}

function renderFrameList() {
  for (const id of ['bulk-input-scope','batch-scope']) {
    const option = $(`#${id} option[value="filtered"]`);
    if (option) option.textContent = `Current catalog filter · ${$('#collection-filter').value || 'all collections'} · ${fmt(state.frames.length)} frames`;
  }
  $('#frame-count').textContent = fmt(state.frames.length);
  $('#frame-picker-count').textContent = `(${fmt(state.frames.length)})`;
  $('#input-frame-select').innerHTML = state.frames.length ? state.frames.map(f=>`<option value="${esc(f.id)}">${esc(f.id)}</option>`).join('') : '<option value="">No frames match the catalog filter</option>';
  if (state.selected && state.frames.some(f=>f.id===state.selected)) $('#input-frame-select').value = state.selected;
  else $('#input-frame-select').value = '';
  $('#frame-list').innerHTML = state.frames.length ? state.frames.map(f => `<button type="button" class="frame-item ${f.id === state.selected ? 'active' : ''}" data-id="${esc(f.id)}" ${f.id === state.selected ? 'aria-current="true"' : ''}><span class="frame-id">${esc(f.id)}</span><span class="frame-meta">${collectionBadges(f)}${badge(f.gt_status)}<span>${fmt(f.ratings_count)} ratings · ${f.has_image ? 'image ready' : 'image pending'}</span></span></button>`).join('') : empty('No frames match this filter.');
  $$('.frame-item').forEach(button => button.addEventListener('click', () => selectFrame(button.dataset.id)));
}

async function loadFrames(selectFirst = false) {
  const generation = ++state.searchGeneration;
  try {
    const data = await api(`/api/frames?q=${encodeURIComponent($('#frame-search').value.trim())}&status=${encodeURIComponent($('#status-filter').value)}&tag=${encodeURIComponent($('#collection-filter').value)}`);
    if (generation !== state.searchGeneration) return;
    state.frames = arr(data); renderFrameList();
    if (selectFirst && !state.selected && state.frames.length) await selectFrame(state.frames[0].id);
    return state.frames;
  } catch (error) { if (generation === state.searchGeneration) showError($('#frame-list'), error, () => loadFrames(selectFirst)); }
}

function confirmDiscard() {
  return (!state.dirty && !state.contractDirty) || window.confirm('You have unsaved reference or input edits. Discard them and continue?');
}

async function selectFrame(id, force = false) {
  if (id === state.selected && !force && !state.detailStale) return;
  if (!force && !confirmDiscard()) return;
  const generation = ++state.generation; state.selected = id; state.dirty = false; state.contractDirty = false;
  renderFrameList(); $('#frame-workspace').innerHTML = empty('Loading frame evidence…');
  try {
    const detail = await api(`/api/frames/${encodeURIComponent(id)}`);
    if (generation !== state.generation) return;
    state.detail = detail; state.detailStale = false; state.editingReference = false; state.editSession = null; state.reviewIssues = null; state.evidenceTab = 'descriptions';
    renderFrame(); renderContract();
    return true;
  } catch (error) { if (generation === state.generation) showError($('#frame-workspace'), error, () => selectFrame(id, true)); }
}

async function refreshSelectedFrameAfterMutation() {
  if (!state.selected) return;
  state.detailStale = true; renderGate();
  if (!state.dirty && !state.contractDirty) {
    await selectFrame(state.selected,true);
    return;
  }
  // Preserve local edits; the backend still rejects saving against an old revision.
  if (!$('#frame-stale-note')) {
    $('#frame-workspace').insertAdjacentHTML('afterbegin','<div id="frame-stale-note" class="callout gate" role="status"><div><strong>New saved evidence is available</strong><p>Your unsaved reference or input edits have been kept. Reload the saved frame before reviewing or approving the updated evidence.</p><button type="button" class="text-button" id="reload-stale-frame">Review latest saved frame ↻</button></div></div>');
    $('#reload-stale-frame').addEventListener('click',()=>selectFrame(state.selected));
  }
  toast('Results were saved. Your unsaved frame edits were kept; reload the saved frame before continuing its review.');
}

function emptyContent() { return {reference_description:'', facts:[], uncertainties:[], sections:Object.fromEntries(FIELDS.map(f => [f,'']))}; }

function renderFrame() {
  const d = state.detail, f = d.frame, gt = d.ground_truth;
  const {logged, reconstructed, adapted, paired} = inputProvenance(d.input_contract);
  $('#frame-workspace').innerHTML = `
    <div class="frame-title"><div><p class="eyebrow">FRAME EVIDENCE · 证据</p><h2>${esc(f.id)}</h2></div><div class="frame-title-actions">${collectionBadges(f)}${badge(gt && d.input_contract && gt.input_hash !== d.input_contract.input_hash ? 'needs_review' : gt?.status)}<button class="text-button" id="frame-protocol">Frame input ↗</button></div></div>
    <div class="frame-columns"><section class="panel evidence-panel"><div class="panel-heading"><h3>${reconstructed ? 'Reconstructed input images' : logged ? 'Recorded input images' : 'Archived frame preview'}</h3><span class="mini-label">SOURCE · 原图</span></div>${paired ? recordedImages(d.input_contract,f.id) : `<div class="image-stage" id="image-stage"><img src="/api/image/${encodeURIComponent(f.id)}" alt="Archived mouse frame ${esc(f.id)}" id="frame-image"></div>`}<div class="image-caption">${adapted ? (reconstructed ? 'Source-reconstructed images and speed text are unchanged. ' : 'Recorded image bytes, view order and speed text are unchanged. ') + 'The system instruction is standardized to v1.0: six output fields, with the tail in Limb. This saved input is used for new GT work and model runs.' : reconstructed ? 'Reconstructed from source recording frames and pose-derived speeds, using the recovered request template. These are the frozen images supplied to GT assistance and candidate models; no original request log was recovered for this frame.' : logged ? 'Original image bytes and view order from the request log. These are the images supplied to GT assistance and candidate models.' : f.input_provenance_status === 'unverified_archived_input' && !d.input_contract ? esc(f.provenance_note || 'Archived image. The historical request and exact model inputs have not been verified; prepare a verified input before ground-truth approval or controlled scoring.') : 'Archived source image. Import the recorded request to verify its actual model input images.'}${paired ? `<details class="workflow-details"><summary>${adapted ? 'Standardized' : reconstructed ? 'Reconstructed' : 'Recorded'} prompt & speed information</summary><p class="recorded-prompt">${esc(d.input_contract.prompt)}</p><details><summary>Current system instruction</summary><p class="recorded-prompt">${esc(d.input_contract.system_prompt)}</p></details></details><details class="workflow-details"><summary>Archived source preview</summary><img class="source-preview" src="/api/image/${encodeURIComponent(f.id)}" alt="Archived source preview"></details>` : ''}</div><div class="evidence-tabs" role="tablist" aria-label="Frame evidence"><button role="tab" data-evidence="descriptions">Model outputs (${arr(d.descriptions).length}${modelImageAssets(d).length ? ` + ${modelImageAssets(d).length} images` : ''})</button><button role="tab" data-evidence="ratings">Ratings (${arr(d.ratings).length})</button><button role="tab" data-evidence="assets">Assets (${arr(d.assets).length})</button></div><div class="evidence-content" id="evidence-content"></div></section>
    <section class="panel editor-panel"><div class="editor-heading"><h3>Ground truth · 参考答案</h3><span class="revision">${gt ? `Revision ${esc(gt.revision)}` : 'No saved reference'}</span></div><div class="reference-toolbar" id="reference-toolbar"></div><form id="gt-form"><fieldset id="gt-fields"><div class="editor-body" id="editor-body"></div><section class="editor-footer reference-approval" id="reference-approval" aria-labelledby="reference-approval-title" tabindex="-1"><div class="reference-approval-heading"><h4 id="reference-approval-title">Approve reference</h4><span class="reference-saved-state">Saved: ${badge(gt?.status)}${gt ? ` · revision ${esc(gt.revision)}` : ''}</span></div><input type="hidden" id="gt-status" value="draft"><p class="field-note" id="gt-gate-note" role="status"></p><div class="reference-save-actions"><button type="submit" class="button secondary" id="save-gt" data-gt-status="draft">Save draft</button><button type="submit" class="button primary" id="approve-gt" data-gt-status="approved" aria-describedby="gt-gate-note">Approve reference</button></div><details class="workflow-details reference-review-details"><summary>Optional review details</summary><label>Your name (optional)<input id="gt-reviewer" autocomplete="name" placeholder="Name" value="${esc(humanReviewerName(gt))}"></label><label>Review notes (optional)<textarea id="gt-notes" rows="2" placeholder="Add a note if helpful">${esc(gt?.notes || '')}</textarea></label></details><details class="workflow-details"><summary>Draft provenance</summary><pre class="code-block">${esc(json(gt?.provenance || {origin:"No saved draft"}))}</pre></details><span class="save-state" id="save-state">${gt ? 'Saved reference loaded' : 'No reference has been created'}</span></section></fieldset></form></section></div>`;
  state.editContent = structuredClone(gt?.content || emptyContent());
  $('#frame-image')?.addEventListener('error', () => { $('#image-stage').innerHTML = '<div class="empty compact"><span class="empty-symbol">▧</span><p>The canonical image has not been downloaded locally yet.</p></div>'; });
  $('#frame-protocol').addEventListener('click', () => switchPage('protocol'));
  $$('[data-evidence]').forEach(b => b.addEventListener('click', () => { state.evidenceTab = b.dataset.evidence; renderEvidence(); }));
  $('#gt-form').addEventListener('input', markDirty);
  $('#gt-form').addEventListener('change', () => { markDirty(); renderGate(); });
  $('#gt-form').addEventListener('submit', saveGroundTruth);
  renderEvidence(); renderEditor(); renderGate(); loadFrameReviewIssues();
}

function markDirty() { state.dirty = true; const el = $('#save-state'); if (el) el.textContent = 'Unsaved changes'; renderGate(); }

function renderEvidence() {
  const d = state.detail, target = $('#evidence-content');
  $$('[data-evidence]').forEach(b => { b.classList.toggle('active', b.dataset.evidence === state.evidenceTab); b.setAttribute('aria-selected', b.dataset.evidence === state.evidenceTab); });
  if (state.evidenceTab === 'descriptions') {
    target.innerHTML = '<div class="callout"><p>Read these outputs alongside the image and expert ratings during reconciliation. They are evidence, not ground truth. Different run copies are retained; no copy is designated the original evaluation output.</p>' + (inputProvenance(d.input_contract).adapted ? '<p><strong>Historical output format:</strong> these archived answers retain their original request requirements. They are not judged against the new six-field standardized prompt. New model runs use Overall, Head, Limb, Torso, Others and Keywords, with tail descriptions in Limb.</p>' : '') + '</div>' + (arr(d.descriptions).length ? d.descriptions.map((v,i) => {
      const format = modelOutputStatus(v);
      return `<details class="candidate candidate-model-output"><summary><span>${esc(v.model_label || v.model || v.model_directory || `Saved description ${i+1}`)}</span><span class="output-format-badge ${esc(format.kind)}">${esc(format.label)}</span></summary>${renderModelOutput(v)}</details>`;
    }).join('') : empty('No model description text is available locally for this frame.')) + renderModelImageAssets(d);
  } else if (state.evidenceTab === 'ratings') {
    const outputIndex = new Map([...arr(d.descriptions), ...arr(d.rating_reference_outputs)].map(output => [output.id, output]));
    target.innerHTML = '<div class="callout"><p>Expand a rating, then its output reference to read the answer in context. Single-model rows label sections; comparisons preserve A/B order. Possible matches stay marked until the exact source correspondence is verified.</p></div>' + (arr(d.ratings).length ? d.ratings.map(v => renderRatingEvidence(v, d.rating_output_references?.[v.id], outputIndex)).join('') : empty('No rating evidence is indexed for this frame.'));
    $$('[data-rating-copy]',target).forEach(b=>b.addEventListener('click',()=>copyText(b.dataset.ratingCopy)));
  } else {
    const records = arr(d.raw_records);
    target.innerHTML = (records.length ? `<h4>Materialized frame records</h4><p class="field-note">Generic frame records; these do not establish native execution logs for every model.</p>${records.map(v => `<div class="asset-row"><div class="asset-row-head"><strong>${esc(v.role || v.log_kind || 'Frame record')}</strong><a class="text-link" href="/api/assets/${encodeURIComponent(v.id)}" target="_blank" rel="noopener">Open JSON ↗</a></div><p class="asset-path">${esc(v.source_path || v.nas_path || v.path || '')}</p></div>`).join('')}<h4 class="asset-heading">Indexed NAS assets</h4>` : '') + (arr(d.assets).length ? d.assets.map((v,i) => `<div class="asset-row"><div class="asset-row-head"><span>${esc(v.role || v.asset_role || v.relation_level || 'Indexed asset')}</span>${materializedAsset(v) ? `<a class="text-link" href="/api/assets/${encodeURIComponent(v.id)}" target="_blank" rel="noopener">Open file ↗</a>` : ''}<button type="button" class="copy-button" data-asset-copy="${i}">Copy path</button></div><p class="asset-path">${esc(v.nas_path || v.source_path || v.path || '')}</p>${v.frame_membership_verified === false ? '<p class="field-note">Recording/container association; per-frame membership unverified.</p>' : ''}</div>`).join('') : empty('No assets are indexed for this frame.'));
    $$('[data-asset-copy]', target).forEach(b => b.addEventListener('click', async () => { const a = d.assets[Number(b.dataset.assetCopy)]; await copyText(a.nas_path || a.source_path || a.path || ''); }));
  }
}

function renderModelImageAssets(detail) {
  const assets = modelImageAssets(detail);
  if (!assets.length) return '';
  return `<h4 class="asset-heading">Archived rendered outputs</h4><p class="field-note">These images preserve the saved model responses. They have not been transcribed into description text or treated as ground truth.</p>${assets.map(asset => `<details class="candidate model-output-image"><summary>${esc(asset.model_label || asset.model_directory || 'Archived model output')} <span class="output-format-badge">Rendered image</span></summary>${materializedAsset(asset) ? `<a class="text-link" href="/api/assets/${encodeURIComponent(asset.id)}" target="_blank" rel="noopener">Open full image ↗</a><img class="source-preview" loading="lazy" src="/api/assets/${encodeURIComponent(asset.id)}" alt="Saved output from ${esc(asset.model_label || asset.model_directory || 'archived model')}">` : '<p class="field-note">Indexed on the NAS; image has not been materialized locally.</p>'}<p class="asset-path">${esc(asset.nas_path || asset.source_path || asset.path || '')}</p></details>`).join('')}`;
}

function factMarkup(f) {
  const fields = FIELDS.includes(f.field) ? FIELDS : [...FIELDS, f.field || ''];
  const evidence = ['image','velocity','unknown'].includes(f.evidence) ? ['image','velocity','unknown'] : ['image','velocity','unknown',f.evidence || ''];
  return `<div class="fact" data-fact-id="${esc(f.id)}"><label>Fact ${esc(f.id)}<textarea data-fact-text rows="2" placeholder="One observable, verifiable fact">${esc(f.text)}</textarea></label><div class="fact-controls"><select data-fact-field aria-label="Section for ${esc(f.id)}">${fields.map(v => `<option value="${esc(v)}" ${v === f.field ? 'selected' : ''}>${esc(v || 'Unassigned')}</option>`).join('')}</select><select data-fact-evidence aria-label="Evidence for ${esc(f.id)}">${evidence.map(v => `<option value="${esc(v)}" ${v === f.evidence ? 'selected' : ''}>${esc(v === 'image' ? 'Image' : v === 'velocity' ? 'Speed' : v || 'Unknown')}</option>`).join('')}</select><label class="checkbox-label"><input type="checkbox" data-fact-required ${f.required ? 'checked' : ''}>Required</label><button type="button" class="remove-button" data-remove-fact aria-label="Remove fact ${esc(f.id)}">×</button></div>${Object.keys(f).some(k=>!["id","text","field","evidence","required"].includes(k))?`<details class="fact-evidence-note"><summary>Evidence links & additional annotation</summary><pre class="code-block">${esc(json(Object.fromEntries(Object.entries(f).filter(([k])=>!["id","text","field","evidence","required"].includes(k)))))}</pre></details>`:""}</div>`;
}

function captureEditor() {
  if (!state.editingReference || !state.editContent || !$('[data-section]')) return;
  const edits = {
    sections:Object.fromEntries($$('[data-section]').map(el => [el.dataset.section, el.value])),
    facts:$$('.fact', $('#editor-body')).map(el => ({id:el.dataset.factId, text:$('[data-fact-text]',el).value, field:$('[data-fact-field]',el).value, evidence:$('[data-fact-evidence]',el).value, required:$('[data-fact-required]',el).checked})),
    uncertainties:$$('[data-uncertainty]').map(el => el.value)
  };
  if ($('#reference-description')) edits.reference_description = $('#reference-description').value;
  state.editContent = window.MousebenchReferenceEdit.capture(state.editContent, edits);
}

function referenceReviewMeta() {
  return json(['gt-status','gt-reviewer','gt-notes'].map(id => $(`#${id}`).value));
}

function focusReferenceRegion(region, editing) {
  if (!region) return;
  const selector = editing ? `[data-reference-region="${region}"]` : FIELDS.includes(region)
    ? `[data-ref-section="${region}"]` : {uncertainties:'.ref-uncertainties',summary:'.ref-summary',unassigned:'.ref-unassigned'}[region];
  const container = selector && $(selector, $('#editor-body'));
  if (!container) return;
  container.scrollIntoView({block:'start',behavior:'auto'});
  $(editing ? 'textarea' : '[data-edit-reference]', container)?.focus({preventScroll:true});
}

function referenceEditActions(region) {
  return `<div class="reference-actions reference-section-actions"><button type="button" class="text-button" data-cancel-reference data-return-region="${esc(region)}">Cancel edits</button><button type="button" class="button secondary" data-apply-reference data-return-region="${esc(region)}">Apply edits</button></div>`;
}

function startReferenceEdit(region = null) {
  if (state.savingReference) return;
  region = [...FIELDS,'uncertainties','summary','unassigned'].includes(region) ? region : null;
  state.editSession = {content:structuredClone(state.editContent), dirty:state.dirty, meta:referenceReviewMeta(), region};
  state.editingReference = true; renderEditor();
  if (region) focusReferenceRegion(region,true); else $('[data-section]')?.focus({preventScroll:true});
}

function finishReferenceEdit(apply, returnRegion = null) {
  if (state.savingReference) return;
  const session = state.editSession;
  if (apply) captureEditor(); else state.editContent = structuredClone(session.content);
  const changed = json(state.editContent) !== json(session.content);
  state.dirty = session.dirty || changed || referenceReviewMeta() !== session.meta;
  state.editingReference = false; state.editSession = null;
  $('#save-state').textContent = state.dirty ? 'Unsaved changes' : state.detail.ground_truth ? 'Saved reference loaded' : 'No reference has been created';
  renderEditor();
  const region = returnRegion || session.region;
  if (region) focusReferenceRegion(region,false); else $('#edit-reference')?.focus({preventScroll:true});
}

function renderReviewHistory() {
  return `<details class="reference-history"><summary>Review history · ${arr(state.detail.history).length} revisions</summary>${arr(state.detail.history).length ? [...state.detail.history].reverse().map(h => `<details class="history-entry"><summary>Revision ${esc(h.revision)} · ${esc(statusName(h.status))}</summary><p class="history-meta">${esc(h.reviewer)} · ${esc(h.created_at || '')}</p><p class="field-note">${esc(h.notes || 'No revision note')}</p><pre class="code-block">${esc(json(h.content))}</pre></details>`).join('') : empty('No saved revisions yet.')}</details>`;
}

function renderEditor() {
  const c = state.editContent, target = $('#editor-body');
  $('#gt-fields').disabled = state.savingReference;
  $('#reference-toolbar').innerHTML = state.editingReference
    ? '<p>Editing locally. Apply edits to return to review, then save your revision.</p><div class="reference-actions"><button type="button" class="button secondary" data-cancel-reference>Cancel edits</button><button type="button" class="button primary" data-apply-reference>Apply edits</button></div>'
    : '<p>Review each section and its grading facts together. Required facts determine recall; additional facts support correctness checks. Then review uncertainties and open issues below.</p><div class="reference-actions"><button type="button" class="text-button" id="review-reference">Approve ↓</button><button type="button" class="button secondary" id="edit-reference">Edit reference</button></div>';
  if (!state.editingReference) {
    target.innerHTML = window.MousebenchReferenceView.render(c) + '<section id="frame-review-issues" class="reference-issues" aria-label="Open reference issues"></section>' + renderReviewHistory();
    $('#edit-reference').disabled = state.savingReference;
    $('#edit-reference').addEventListener('click', () => startReferenceEdit());
    $('#review-reference').addEventListener('click', focusReferenceApproval);
    $$('[data-edit-reference]',target).forEach(button=>{
      button.disabled = state.savingReference;
      button.addEventListener('click',()=>startReferenceEdit(button.dataset.editReference));
    });
    renderFrameReviewIssues(); renderGate();
    return;
  }
  const generatedSummary = !c.reference_description || c.reference_description === window.MousebenchReferenceView.summaryFromSections(c.sections);
  const extraFacts = arr(c.facts).filter(f => !FIELDS.includes(f.field));
  target.innerHTML = `<p class="editor-note">Keep descriptions, facts and uncertainties consistent when editing. Changing prose does not rewrite its grading facts.</p><div class="reference-edit-sections">${FIELDS.map(field => `<section class="reference-edit-section" data-reference-region="${field}">${referenceEditActions(field)}<label>${field}<textarea data-section="${field}" rows="3" placeholder="Evidence-grounded ${field.toLowerCase()} description">${esc(c.sections?.[field] || '')}</textarea></label><div class="field-heading"><h4>Grading facts</h4><button type="button" class="text-button" data-add-fact="${field}">+ Add fact</button></div><div class="fact-list">${arr(c.facts).filter(f=>f.field===field).map(factMarkup).join('')}</div></section>`).join('')}${extraFacts.length ? `<section class="reference-edit-section" data-reference-region="unassigned">${referenceEditActions('unassigned')}<h4>Unassigned grading facts</h4>${extraFacts.map(factMarkup).join('')}</section>` : ''}</div>${generatedSummary ? '<p class="field-note">The combined reference summary follows the six sections automatically.</p>' : `<section class="reference-edit-section" data-reference-region="summary">${referenceEditActions('summary')}<label>Additional reference summary<textarea id="reference-description" rows="5">${esc(c.reference_description)}</textarea></label><p class="field-note">This saved summary differs from the six sections and is also sent to the judge. Keep it consistent with your edits.</p></section>`}<section class="reference-edit-section" data-reference-region="uncertainties">${referenceEditActions('uncertainties')}<div class="field-heading"><h4>Uncertainties & alternatives</h4><button type="button" class="text-button" id="add-uncertainty">+ Add</button></div><div id="uncertainty-list">${arr(c.uncertainties).map((v,i) => `<div class="uncertainty-row"><textarea data-uncertainty rows="2" aria-label="Uncertainty ${i+1}">${esc(v)}</textarea><button type="button" class="remove-button" data-remove-uncertainty aria-label="Remove uncertainty ${i+1}">×</button></div>`).join('')}</div></section><div class="reference-actions reference-edit-end"><button type="button" class="button secondary" data-cancel-reference>Cancel edits</button><button type="button" class="button primary" data-apply-reference>Apply edits</button></div>`;
  $$('[data-apply-reference]').forEach(b=>b.addEventListener('click',()=>finishReferenceEdit(true,b.dataset.returnRegion)));
  $$('[data-cancel-reference]').forEach(b=>b.addEventListener('click',()=>finishReferenceEdit(false,b.dataset.returnRegion)));
  $$('[data-add-fact]').forEach(b=>b.addEventListener('click',()=>{
    captureEditor(); const id = `fact_${crypto.randomUUID().slice(0,8)}`;
    state.editContent.facts.push({id,text:'',field:b.dataset.addFact,evidence:'image',required:false});
    markDirty(); renderEditor(); $(`[data-fact-id="${id}"] [data-fact-text]`)?.focus();
  }));
  $$('[data-remove-fact]').forEach(b=>b.addEventListener('click',()=>{ b.closest('.fact').remove(); captureEditor(); markDirty(); }));
  $('#add-uncertainty').addEventListener('click',()=>{ captureEditor(); state.editContent.uncertainties.push(''); markDirty(); renderEditor(); $$('[data-uncertainty]').at(-1)?.focus(); });
  $$('[data-remove-uncertainty]').forEach(b=>b.addEventListener('click',()=>{ b.closest('.uncertainty-row').remove(); captureEditor(); markDirty(); }));
  renderGate();
}

async function loadFrameReviewIssues() {
  const generation = state.generation, id = state.selected, d = state.detail;
  state.reviewIssues = null; renderFrameReviewIssues(); renderGate();
  try {
    const records = await api('/api/workflow/issues');
    if (generation !== state.generation || id !== state.selected) return;
    state.reviewIssues = {frameId:id, revision:d.ground_truth?.revision, contentHash:d.ground_truth?.content_sha256, inputHash:d.input_contract?.input_hash, items:arr(records).filter(i=>i.frame_id===id && i.status==='open' && i.current && (i.target || 'reference')==='reference' && i.content_sha256===d.ground_truth?.content_sha256 && i.input_sha256===d.input_contract?.input_hash)};
  } catch (error) {
    if (generation !== state.generation || id !== state.selected) return;
    state.reviewIssues = {error:error.message};
  }
  renderFrameReviewIssues(); renderGate();
}

function renderFrameReviewIssues() {
  const target = $('#frame-review-issues'); if (!target) return;
  const issues = state.reviewIssues;
  if (!issues) { target.innerHTML = '<h4>Open review issues</h4><p class="field-note">Loading saved reference checks…</p>'; return; }
  if (issues.error) {
    target.innerHTML = `<h4>Open review issues</h4><p class="review-issue-error">Could not load review issues: ${esc(issues.error)}</p><button type="button" class="text-button" id="retry-frame-issues">Retry loading issues</button>`;
    $('#retry-frame-issues').addEventListener('click',loadFrameReviewIssues); return;
  }
  const uncertainties = arr(state.editContent.uncertainties);
  const repeated = issues.items.filter(i=>i.category==='uncertainty' && i.severity!=='critical' && !i.suggested_fix && !i.evidence && !arr(i.fact_ids).length && uncertainties.includes(i.message));
  const remaining = issues.items.filter(i=>!repeated.includes(i));
  const modified = json(state.editContent) !== json(state.detail.ground_truth?.content || emptyContent());
  target.innerHTML = `<div class="field-heading"><h4>Open review issues <span class="count">${issues.items.length}</span></h4><button type="button" class="text-button" id="manage-frame-issues">Manage issues ↗</button></div><p class="field-note">${modified ? 'These checks apply to the saved reference. Your unsaved edits have not been rechecked.' : 'Checks for the saved reference and recorded input.'}${repeated.length ? ` ${repeated.length} ${repeated.length===1?'issue is':'issues are'} the uncertainty ${repeated.length===1?'note':'notes'} shown above.` : ''}</p>${remaining.length ? remaining.map(i=>`<article class="reference-issue"><div class="ref-fact-labels"><span class="review-issue-severity ${esc(i.severity)}">${esc(i.severity)}</span>${arr(i.fact_ids).length ? `<span>${esc(i.fact_ids.join(', '))}</span>` : ''}</div><p>${esc(i.message)}</p>${i.suggested_fix ? `<p class="review-issue-fix"><strong>Suggested next step:</strong> ${esc(display(i.suggested_fix))}</p>` : ''}${i.evidence ? `<details><summary>Evidence details</summary><p>${esc(display(i.evidence))}</p></details>` : ''}</article>`).join('') : !issues.items.length ? '<p class="field-note">No open issues are recorded. This does not replace human review.</p>' : ''}`;
  $('#manage-frame-issues').addEventListener('click',()=>switchPage('workflow'));
}

function humanReviewerName(gt) {
  const name = gt?.reviewer || '';
  return /\b(codex|assistant|gpt(?:[-\s]?\d)?)\b/i.test(name) ? '' : name;
}

function focusReferenceApproval() {
  const footer = $('#reference-approval');
  footer?.scrollIntoView({block:'start',behavior:'smooth'});
  footer?.focus({preventScroll:true});
}

function referenceActionReason(status) {
  const gt = state.detail?.ground_truth, contract = state.detail?.input_contract, content = state.editContent;
  if (state.savingReference) return 'A reference revision is being saved.';
  if (state.editingReference) return 'Apply or cancel your reference edits before saving a revision.';
  if (state.contractDirty) return 'Save the input contract edits first, or refresh to discard them, before saving this reference.';
  if (state.detailStale) return 'Reload the latest saved evidence before approving this reference.';
  if (status === 'draft') return '';
  if (!contract) return 'Save the recorded input before approving this reference.';
  if (!arr(content?.facts).some(f => f.required)) return 'Add at least one required grading fact before approving.';
  if (!FIELDS.every(field => typeof content?.sections?.[field] === 'string' && content.sections[field].trim())) return 'Complete all six sections before approving.';
  if (arr(content?.facts).some(f => f.evidence === 'velocity') && !contract.velocity) return 'Speed facts need verified speed information in the saved input.';
  if (gt?.status === 'approved' && json(content) === json(gt.content) && gt.input_hash === contract.input_hash) return 'This saved reference is already approved.';
  const issues = state.reviewIssues;
  if (!issues || issues.error || issues.frameId !== state.selected || issues.revision !== gt?.revision || issues.contentHash !== gt?.content_sha256 || issues.inputHash !== contract.input_hash) return 'Load the current review issues successfully before approval. Use Retry loading issues if needed.';
  if (issues.items.some(issue => issue.severity === 'critical')) return 'Resolve the open critical reference issues before approval.';
  return '';
}

function renderGate() {
  if (!$('#gt-gate-note') || !state.detail) return;
  const draftReason = referenceActionReason('draft'), approvalReason = referenceActionReason('approved');
  $('#save-gt').disabled = Boolean(draftReason);
  $('#approve-gt').disabled = Boolean(approvalReason);
  $('#gt-gate-note').textContent = approvalReason || 'Approve the descriptions and grading facts in one click. Name and notes are optional.';
}

function referenceSubmitStatus(event) {
  const status = event.submitter?.dataset?.gtStatus;
  // Save draft is the first submit button, so implicit Enter submissions never approve.
  return status === 'approved' ? 'approved' : 'draft';
}

function nextDraftFrame(frames, previousOrder, approvedId) {
  const drafts = new Set(frames.filter(f => f.gt_status === 'draft' && f.id !== approvedId).map(f => f.id));
  const pivot = previousOrder.indexOf(approvedId);
  const order = [...previousOrder.slice(pivot + 1), ...previousOrder.slice(0, pivot + 1)];
  return order.find(id => drafts.has(id)) || frames.find(f => drafts.has(f.id))?.id || null;
}

function focusReferenceStart() {
  const title = $('#frame-workspace .frame-title');
  if (!title) return;
  title.setAttribute('tabindex','-1');
  title.focus({preventScroll:true});
  title.scrollIntoView({block:'start',behavior:'auto'});
}

async function saveGroundTruth(event) {
  event.preventDefault();
  if (state.savingReference) return;
  const status = referenceSubmitStatus(event), reason = referenceActionReason(status);
  if (reason) return toast(reason,true);
  const d = state.detail, fid = state.selected;
  $('#gt-status').value = status;
  const payload = {expected_revision:d.ground_truth?.revision || 0,status,reviewer:$('#gt-reviewer').value.trim(),notes:$('#gt-notes').value,content:structuredClone(state.editContent),actor_type:'human',human_confirmed:status === 'approved'};
  if (status === 'approved') { payload.approval_action = 'approve_reference'; payload.expected_input_hash = d.input_contract.input_hash; }
  const generation = state.generation, previousOrder = state.frames.map(f => f.id);
  const search = $('#frame-search').value, filter = $('#status-filter').value, collection = $('#collection-filter').value;
  const button = event.submitter || $('#save-gt');
  state.savingReference = true; $('#gt-fields').disabled = true; $('#edit-reference').disabled = true; renderGate();
  try { await busy(button, async () => {
    await api(`/api/frames/${encodeURIComponent(fid)}/ground-truth`,payload);
    // Refresh the queue only after a confirmed save; never advance from stale statuses.
    if (state.selected === fid && state.generation === generation) { state.dirty = false; state.detailStale = true; }
    const [frames] = await Promise.all([loadFrames(),loadSummary()]);
    let message = status === 'approved' ? `Reference approved for ${fid}.` : 'Draft saved. Click Approve reference when ready.';
    // A user navigation or filter change during the request takes precedence.
    if (state.selected === fid && state.generation === generation && !state.dirty && !state.contractDirty) {
      const advance = status === 'approved' && state.page === 'frames' && frames !== undefined && search === $('#frame-search').value && filter === $('#status-filter').value && collection === $('#collection-filter').value;
      const next = advance ? nextDraftFrame(frames,previousOrder,fid) : null;
      const loaded = await selectFrame(next || fid,true);
      if (loaded && state.page === 'frames') {
        if (next) { focusReferenceStart(); message += ` Opened next draft: ${next}.`; }
        else if (status === 'approved') {
          focusReferenceApproval();
          if (advance) message += ' No drafts remain in the current filters.';
        }
      }
      if (status === 'approved' && frames === undefined) message += ' The catalog could not be refreshed; choose the next draft after retrying.';
    }
    toast(message);
  }); } finally {
    state.savingReference = false;
    if ($('#gt-fields')) $('#gt-fields').disabled = false;
    if ($('#edit-reference')) $('#edit-reference').disabled = false;
    $$('[data-edit-reference]').forEach(button => { button.disabled = false; });
    renderGate();
  }
}

function switchPage(page) {
  state.page = page; $('#metrics').hidden = page === 'overview'; $$('.page').forEach(el => el.classList.toggle('active',el.id === `page-${page}`));
  window.scrollTo({top:0,behavior:'auto'});
  $$('.primary-tab').forEach(el => { const active = el.dataset.page === page; el.classList.toggle('active',active); if (active) el.setAttribute('aria-current','page'); else el.removeAttribute('aria-current'); });
  if (page === 'frames' && state.detailStale && !state.dirty && !state.contractDirty) selectFrame(state.selected,true);
  if (page === 'overview') window.MousebenchOverview?.load();
  if (page === 'runs') loadRunsPage();
  if (page === 'scoring-review') window.MousebenchScoringReview?.load();
  if (page === 'workflow' || page === 'protocol') window.MousebenchWorkflow?.load();
}

function inputProvenance(contract) {
  const adapted = contract?.protocol_version === 'mousebench.input.v3.adapted';
  let sourceInput = contract;
  for (let depth = 0; depth < 20 && sourceInput?.protocol_version === 'mousebench.input.v3.adapted'; depth++) {
    sourceInput = sourceInput.adaptation?.parent_input;
  }
  const logged = sourceInput?.protocol_version === 'mousebench.input.v3.logged';
  const reconstructed = sourceInput?.protocol_version === 'mousebench.input.v3.reconstructed';
  return {adapted, logged, reconstructed, paired:adapted || logged || reconstructed, sourceInput};
}

function recordedImages(contract, frameId) {
  return `<div class="recorded-view-grid">${arr(contract?.input_images).map((img,i) => `<figure><img src="/api/input-image/${encodeURIComponent(frameId)}/${encodeURIComponent(img.attachment_id)}" alt="${inputProvenance(contract).reconstructed ? 'Reconstructed' : 'Recorded'} input view ${i+1} for ${esc(frameId)}"><figcaption>View ${i+1} · ${esc(img.image_detail || 'default')} detail</figcaption></figure>`).join('')}</div>`;
}

function renderContract() {
  const d = state.detail, c = d?.input_contract, r = d?.recorded_input;
  const {logged, reconstructed, adapted, paired, sourceInput} = inputProvenance(c);
  const ready = r?.status === 'ready', shown = paired ? c : ready ? r : c;
  $('#contract-frame').textContent = d?.frame.id || 'No frame selected';
  $('#contract-prompt').value = shown?.prompt || ''; $('#contract-velocity').value = shown?.velocity ? json(shown.velocity) : 'null';
  $('#contract-system-prompt').value = shown?.system_prompt || ''; $('#contract-image-detail').value = shown?.image_detail || 'high';
  $('#contract-notes').value = c?.notes || ''; $('#contract-reviewer').value = c?.reviewer || '';
  $('#save-contract').disabled = adapted || reconstructed || !ready;
  $('#save-contract').textContent = adapted ? 'Standardized input saved' : reconstructed ? 'Reconstructed input saved' : logged ? 'Save recorded input version' : 'Import recorded input';
  $('#contract-notes').readOnly = adapted || reconstructed; $('#contract-reviewer').readOnly = adapted || reconstructed;
  $('#contract-system-note').textContent = adapted ? 'Standardized to the exact v1.0 system instruction: Overall, Head, Limb, Torso, Others and Keywords. Tail descriptions belong in Limb. The frame-specific user prompt, speeds and ordered image bytes are unchanged from the prior input version.' : reconstructed ? 'The instruction is preserved from the recovered historical template. Images and speeds were reconstructed from documented source data.' : 'Imported unchanged from this frame’s request log. Editing these fields would create a different protocol.';
  $('#recorded-input-status').innerHTML = adapted ? '<p><strong>Standardized input saved.</strong> This new input version uses the v1.0 six-field system instruction, with the tail in Limb. The prior input and its historical output requirements remain preserved. New GT work and model runs use this standardized version.</p>' : reconstructed ? '<p><strong>Reconstructed input saved.</strong> The historical request template is reused with source-derived images and velocity values. No original request log was found for this frame. Review the derivation and its limitations below; this is the frozen input for new GT and model runs.</p>' : logged ? '<p><strong>Original request imported.</strong> The system instruction, frame-specific user text and ordered image bytes come from this frame’s source log. Speed is already in the user text; it is not appended again.</p>' : ready ? '<p><strong>Original request found.</strong> Review the recorded input below and import it to use for this frame.</p>' : `<p><strong>Recorded input unavailable.</strong> ${esc(r?.error || r?.status || 'Select a frame first.')} No replacement prompt will be generated.</p>`;
  if (reconstructed && arr(sourceInput.reconstruction?.quality_flags).length) $('#recorded-input-status').innerHTML += `<p><strong>Review speed estimates:</strong></p><ul>${sourceInput.reconstruction.quality_flags.map(note => `<li>${esc(note)}</li>`).join('')}</ul>`;
  if (logged && !adapted && !ready) $('#recorded-input-status').innerHTML += `<p class="job-error">Source recheck: ${esc(r?.error || r?.status)}. Re-import is blocked.</p>`;
  $('#recorded-input-images').innerHTML = paired ? recordedImages(c,d.frame.id) : ready ? `<p class="field-note">${fmt(arr(r.input_images).length)} original image attachments will be imported in recorded order.</p>` : '';
  const labels = {overall:'Overall',head:'Head',left_forelimb:'Left forelimb',right_forelimb:'Right forelimb',left_hind_limb:'Left hind limb',right_hind_limb:'Right hind limb'};
  const speeds = shown?.velocity?.value;
  $('#recorded-input-speeds').innerHTML = speeds && typeof speeds === 'object' ? `<h4>${reconstructed ? 'Derived' : 'Recorded'} speed estimates · ${esc(shown.velocity.units)}</h4><dl class="recorded-speed-grid">${Object.entries(speeds).map(([key,v])=>`<div><dt>${esc(labels[key] || key)}</dt><dd>${esc(typeof v === 'number' ? v.toFixed(2) : v)}</dd></div>`).join('')}</dl><p class="field-note">The original instruction treats these as noisy auxiliary estimates and asks for image-led descriptions.</p>` : '';
  const source = logged ? sourceInput.source_request : ready ? {record_id:r.source_record_id,source_path:r.source_path,sha256:r.source_sha256,generation_settings:r.generation_settings} : null;
  $('#recorded-input-source').innerHTML = reconstructed ? `<p class="field-note">The reconstruction manifest preserves the template sources, velocity calculation, image extraction and validation evidence. It is not an archived inference log.</p>${sourceInput.reconstruction?.manifest_id ? `<a class="text-link" href="/api/assets/${encodeURIComponent(sourceInput.reconstruction.manifest_id)}" target="_blank" rel="noopener">Open reconstruction manifest ↗</a>` : ''}<details class="workflow-details"><summary>Reconstruction details & limitations</summary><pre class="code-block">${esc(json(sourceInput.reconstruction || {}))}</pre></details>` : source ? `<p class="asset-path">${esc(source.source_path)}</p><a class="text-link" href="/api/assets/${encodeURIComponent(source.record_id)}" target="_blank" rel="noopener">Open original request log ↗</a><p class="metadata-strip">Source SHA-256: ${esc(source.sha256)}</p><p class="field-note">Recorded generation settings: ${esc(json(source.generation_settings))}. Candidate-model settings are configured separately. This source does not by itself verify every historical model’s execution.</p>` : '';
  if (adapted) $('#recorded-input-source').innerHTML = `<p class="field-note">Only the system instruction was standardized. Image and speed provenance below belongs to the preserved source input.</p><details class="workflow-details"><summary>Prompt standardization provenance</summary><pre class="code-block">${esc(json(c.adaptation || {}))}</pre></details>` + $('#recorded-input-source').innerHTML;
  $('#contract-meta').textContent = c ? `Revision ${c.revision} · ${c.track || ''}\nInput SHA-256: ${c.input_hash || 'unavailable'}` : 'No saved input yet. Import preserves the recorded request without rewriting it.';
  state.contractDirty = false;
}

async function saveContract(event) {
  event.preventDefault(); if (!state.detail) return toast('Select a frame first.',true);
  if (inputProvenance(state.detail.input_contract).adapted) return toast('This frame has a standardized input version. Its original source remains available in provenance.',true);
  const source = state.detail.recorded_input;
  if (source?.status !== 'ready') return toast('A valid source request log is required.',true);
  const fid = state.selected, payload = {expected_revision:state.detail.input_contract?.revision || 0,source_record_id:source.source_record_id,notes:$('#contract-notes').value,reviewer:$('#contract-reviewer').value.trim()};
  await busy($('#save-contract'), async () => { const result = await api(`/api/frames/${encodeURIComponent(fid)}/input-contract`,payload); if (state.selected === fid) { state.detail.input_contract = result; renderContract(); renderGate(); } toast(`Recorded input saved for ${fid}.`); loadSummary(); });
}

async function readRecords(input, status) {
  const file = input.files[0]; if (!file) throw new Error('Choose a JSONL file first.');
  if (file.size > 24_000_000) throw new Error('Use a file smaller than 24 MB.');
  const text = (await file.text()).trim(); if (!text) throw new Error('The selected file is empty.');
  let rows;
  if (text.startsWith('[')) { try { rows = JSON.parse(text); } catch { throw new Error('Invalid JSON array.'); } }
  else rows = text.split(/\r?\n/).flatMap((line,i) => { if (!line.trim()) return []; try { return [JSON.parse(line)]; } catch { throw new Error(`Invalid JSON on line ${i+1}.`); } });
  if (!Array.isArray(rows) || !rows.length || rows.some(r => !r || typeof r !== 'object' || Array.isArray(r))) throw new Error('Each record must be a JSON object.');
  if (status) status.textContent = `${fmt(rows.length)} records read from ${file.name}. The server will validate each record before import.`;
  return rows;
}

function openDetails(title, value) { $('#dialog-title').textContent = title; $('#dialog-body').innerHTML = `<pre class="code-block">${esc(json(value))}</pre>`; $('#detail-dialog').showModal(); }

async function loadRunsPage() { await Promise.allSettled([loadRuns(),loadScorecards()]); }

function matchesRunTrack(run) {
  return state.runTrack === 'all' || run.track === state.runTrack;
}

function setRunTrack(track, {reloadReview = true} = {}) {
  if (!['all','controlled','historical'].includes(track)) return;
  state.runTrack = track;
  try { localStorage.setItem('mousebench.runs.track', track); } catch {}
  for (const id of ['runs-track-filter','score-review-track']) {
    const select = document.getElementById(id);
    if (select) select.value = track;
  }
  renderRuns();
  renderScorecards();
  if (reloadReview && state.page === 'scoring-review') window.MousebenchScoringReview?.load();
}

function renderRuns() {
  const runs = state.runs.filter(matchesRunTrack);
  $('#runs-filter-count').textContent = `${fmt(runs.length)} of ${fmt(state.runs.length)} runs shown · Applies to runs and scorecards`;
    $('#runs-list').innerHTML = runs.length ? runs.map(r => `<article class="run-card"><div class="run-card-head"><h4>${esc(r.name || r.id)}</h4><span class="pill ${r.track === 'controlled' ? 'reviewed' : 'neutral'}">${esc(r.track || 'Unknown track')}</span></div><p class="run-model">${esc(r.model)}</p><div class="run-details"><span>${fmt(r.predictions_count)} predictions</span><span>${pct(r.coverage)} catalog coverage</span><span>${esc(r.eligibility || 'Eligibility unknown')}</span></div>${r.track !== 'controlled' ? '<p class="field-note">Historical provenance may be incomplete; not equivalent to a controlled run.</p>' : '<p class="field-note">Input contract matched. Model identity is declared by the importer.</p>'}<button class="text-button" data-run-details="${esc(r.id)}">Inspect manifest ↗</button></article>`).join('') : empty(state.runs.length ? `No ${state.runTrack} runs yet. Choose All to see other saved runs.` : 'No prediction runs imported yet. Import actual historical or controlled outputs to begin.');
  $$('[data-run-details]').forEach(button => button.addEventListener('click', () => {
    const run = state.runs.find(item => item.id === button.dataset.runDetails);
    if (run) openDetails('Run manifest', run);
  }));
}

async function loadRuns() {
  try {
    state.runs = arr(await api('/api/runs'));
    renderRuns();
    const previous = $('#preview-run').value;
    $('#preview-run').innerHTML = '<option value="">Select a run</option>' + state.runs.map(r => `<option value="${esc(r.id)}">${esc(r.name)} · ${esc(r.model)}</option>`).join('');
    if (state.runs.some(r => r.id === previous)) $('#preview-run').value = previous;
  } catch(error) { showError($('#runs-list'),error,loadRuns); }
}
let scorecardRequestGeneration = 0;
async function loadScorecards() {
  const generation = ++scorecardRequestGeneration;
  try {
    const data = await api('/api/scorecards');
    if(generation !== scorecardRequestGeneration) return;
    state.scorecards = data;
    renderScorecards();
  } catch(error) {
    if(generation !== scorecardRequestGeneration) return;
    window.MousebenchComparison.unmount($('#runs-comparison'));
    showError($('#scorecards'),error,loadScorecards);
    showError($('#runs-comparison'),error,loadScorecards);
  }
}

let comparisonView = {};
function renderRunComparison(data) {
  const models = arr(data.models).filter(matchesRunTrack);
  window.MousebenchComparison.mount($('#runs-comparison'), {
    data:{...data,models,groups:arr(data.groups).filter(g=>models.some(m=>m.group_id===g.group_id))},
    state:comparisonView,
    onStateChange:view=>{comparisonView=view;},
    onReview:(runId,groupId)=>window.MousebenchScoringReview?.open(runId,groupId)
  });
}

function openRunComparison(groupId, data=state.scorecards) {
  comparisonView={groupId};
  const group=arr(data?.groups).find(g=>g.group_id===groupId);
  const track=group?.track || arr(data?.models).find(m=>m.group_id===groupId)?.track;
  if(track) setRunTrack(track,{reloadReview:false});
  // Changing the run filter may remount the chart. Restore the requested group.
  comparisonView={groupId};
  if(data) state.scorecards=data;
  switchPage('runs');
  if(data) renderScorecards();
  $('#runs-comparison-panel')?.scrollIntoView({behavior:'smooth',block:'start'});
}

function renderScorecards() {
  const data = state.scorecards;
  if (!data) return;
  renderRunComparison(data);
  const models = arr(data.models).filter(matchesRunTrack);
    const indexValue = value => typeof value === 'number' && Number.isFinite(value) ? value.toFixed(1) : '—';
    const methods=models.map(m=>window.MousebenchRubric.scoreLabels(m));
    const allGraded=methods.length>0&&methods.every(m=>m.graded), someGraded=methods.some(m=>m.graded);
    const columns = [['factual_precision',allGraded?'Graded correctness':someGraded?'Correctness / precision':'Fact precision'],['required_fact_recall',allGraded?'Graded required coverage':someGraded?'Coverage / recall':'Required recall'],['factual_f1',allGraded?'Graded factual F1':'Factual F1'],['structure_score','Structure'],['instruction_score','Instructions']];
    if(models.some(m=>Number.isFinite(m.dimension_means?.major_error_rate)))columns.push(['major_error_rate','Major factual error rate']);
    const note = `<div class="callout"><p>${data.provisional ? 'Provisional scorecards. ' : ''}Judge calibration: ${esc(data.calibration_status || 'not established')}. ${fmt(data.stale_judgments_excluded || 0)} stale judgments excluded; ${fmt(data.critical_issue_judgments_excluded || 0)} judgments excluded for open critical reference issues. Groups preserve protocol, rubric, judge and comparison track; there is no pooled leaderboard across incompatible inputs.</p></div><p class="field-note score-explainer">Factual quality and required-fact coverage determine F1. Each row identifies its scoring version: graded scoring allows partial correctness and coverage; legacy scoring uses binary support. These versions are not interchangeable. The optional 0–100 index is an exploratory composite with unvalidated weights; structure and prompt following modify that score using the frozen rubric. The completion-adjusted index treats unevaluated catalog frames as zero utility; it does not fabricate semantic judgments. Compare the component scores, coverage and calibration evidence alongside the index.</p>`;
    $('#scorecards').innerHTML = note + (models.length ? `<div class="table-wrap"><table><thead><tr><th>Model / run</th><th>Track · group</th><th>Evaluable</th><th>Coverage</th>${columns.map(([,label]) => `<th>${label}</th>`).join('')}<th>Exploratory index / 100</th><th>Exploratory adjusted index</th><th>Details</th></tr></thead><tbody>${models.map((m,i) => `<tr><td><strong>${esc(m.model)}</strong><br>${esc(m.name || m.run_id)}</td><td>${esc(m.track)}<br><span class="field-note">${esc(m.group_id || 'Unspecified group')}</span><span class="score-method-label">${esc(window.MousebenchRubric.scoreLabels(m).versionLabel)}</span></td><td>${fmt(m.evaluable_count)} / ${fmt(m.attempted_count)}<br><span class="field-note">${fmt(m.failed_count)} failed</span></td><td class="numeric">${pct(m.benchmark_coverage)}</td>${columns.map(([key]) => `<td class="numeric">${pct(m.dimension_means?.[key])}</td>`).join('')}<td class="numeric">${indexValue(m.benchmark_index)}<br><span class="field-note">${fmt(m.index_count)} scored</span></td><td class="numeric">${indexValue(m.completion_adjusted_index)}</td><td><button class="text-button" data-review-score="${i}">Review frames →</button><br><button class="text-button" data-score="${i}">Statistics ↗</button></td></tr>`).join('')}</tbody></table></div>` : empty(state.runTrack === 'all' ? 'No eligible judgments yet. Approve references, import predictions and then import assessments against the saved versions.' : `No ${state.runTrack} scores yet. Saved model outputs remain available above; scores appear after grading.`));
    $$('[data-score]').forEach(b => b.addEventListener('click',()=>openDetails('Scorecard, common-frame and recording-balanced results',models[Number(b.dataset.score)])));
    $$('[data-review-score]').forEach(b => b.addEventListener('click',()=>{const m=models[Number(b.dataset.reviewScore)];window.MousebenchScoringReview?.open(m.run_id,m.group_id);}));

}

async function loadProvider() {
  try { const p = await api('/api/providers'); $('#provider-name').textContent = 'Codex & API execution'; $('#provider-description').textContent = 'Prepare Codex-assisted work or configure an API batch in Jobs & validation. Live API calls require an explicit local command.'; $('#provider-details').innerHTML = `<span class="pill neutral">Browser makes no model calls</span>`; }
  catch(error) { $('#provider-description').textContent = `Provider configuration unavailable. ${error.message}`; }
}

async function previewJob(event) {
  event.preventDefault(); const filtered = $('#preview-scope').value === 'filtered';
  const ids = filtered ? state.frames.map(f => f.id) : state.selected ? [state.selected] : [];
  if (!ids.length) return toast('Select a frame or a nonempty filtered catalog.',true);
  const payload = {task:$('#preview-task').value,frame_ids:ids};
  if (payload.task === 'judge') { if (!$('#preview-run').value) return toast('Select a run for the assessment preview.',true); payload.run_id = $('#preview-run').value; }
  await busy($('#preview-form button'),async () => {
    state.preview = await api('/api/jobs/preview',payload);
    $('#preview-result').innerHTML = `<div class="callout"><p>Request preview prepared for ${fmt(ids.length)} selected frame${ids.length === 1 ? '' : 's'}. No live model call was made. Inspect validation and payload details before execution elsewhere.</p></div><button type="button" class="text-button" id="download-preview">Download request JSON ↓</button><details class="candidate"><summary>Inspect preview response</summary><pre>${esc(json(state.preview))}</pre></details>`;
    $('#download-preview').addEventListener('click',()=>downloadJSON(state.preview,`mousebench-${payload.task}-preview.json`));
    toast('Request preview is ready. No model call made.');
  });
}

function downloadJSON(value,name) { const url=URL.createObjectURL(new Blob([json(value)+'\n'],{type:'application/json'})); const a=document.createElement('a'); a.href=url; a.download=name; a.click(); setTimeout(()=>URL.revokeObjectURL(url),1000); }
async function copyText(text) { try { await navigator.clipboard.writeText(text); toast('Copied to clipboard.'); } catch { openDetails('NAS path',text); } }

function wire() {
  try {
    const savedTrack = localStorage.getItem('mousebench.runs.track');
    if (['all','controlled','historical'].includes(savedTrack)) state.runTrack = savedTrack;
  } catch {}
  $('#runs-track-filter').value = state.runTrack;
  $('#runs-track-filter').addEventListener('change', event => setRunTrack(event.target.value));
  $('#toggle-frame-catalog').addEventListener('click',()=>{ const open = $('#page-frames .frame-layout').classList.toggle('catalog-open'); $('#toggle-frame-catalog').setAttribute('aria-expanded',String(open)); });
  $$('.primary-tab').forEach(b=>b.addEventListener('click',()=>switchPage(b.dataset.page)));
  $$('[data-go-page]').forEach(b=>b.addEventListener('click',()=>{switchPage(b.dataset.goPage); if(b.dataset.goAnchor)document.getElementById(b.dataset.goAnchor)?.scrollIntoView({behavior:'smooth',block:'start'}); else window.scrollTo({top:0,behavior:'smooth'});}));
  $$('[data-open-job]').forEach(b=>b.addEventListener('click',()=>window.MousebenchWorkflow?.openTask(b.dataset.openJob)));
  $('#input-frame-select').addEventListener('change',async event=>{const id=event.target.value;if(id)await selectFrame(id);event.target.value=state.selected || '';});
  let timer; $('#frame-search').addEventListener('input',()=>{clearTimeout(timer); timer=setTimeout(()=>loadFrames(),180);});
  $('#status-filter').addEventListener('change',()=>loadFrames());
  $('#collection-filter').addEventListener('change',()=>loadFrames());
  $('#refresh-all').addEventListener('click',async()=>{ if (!confirmDiscard()) return; state.dirty=false; state.contractDirty=false; await Promise.allSettled([loadSummary(),loadFrames(),state.page==='runs' ? loadRunsPage() : loadRuns(),loadProvider(),state.page==='overview' ? window.MousebenchOverview?.load() : Promise.resolve(),state.page==='scoring-review' ? window.MousebenchScoringReview?.load() : Promise.resolve(),state.selected ? selectFrame(state.selected,true) : Promise.resolve()]); toast('Workspace refreshed.'); });
  $('#provider-workflow').addEventListener('click',()=>switchPage('workflow'));
  $('#contract-form').addEventListener('submit',saveContract); $('#contract-form').addEventListener('input',()=>{state.contractDirty=true; renderGate();});
  $('#refresh-scores').addEventListener('click',loadScorecards);
  $('#run-track').addEventListener('change',()=>{$('#track-note').textContent=$('#run-track').value==='controlled' ? 'Every record must match its frozen input hash and include model_id, generation_settings and a native_execution request log. Image-only and image-plus-velocity tracks stay separate.' : 'Missing prompts, checkpoints or settings remain unknown. A saved output alone does not establish comparability.';});
  $('#run-file').addEventListener('change',async()=>{try{await readRecords($('#run-file'),$('#run-file-status'));}catch(e){$('#run-file-status').textContent=e.message;}});
  $('#run-import-form').addEventListener('submit',async event=>{event.preventDefault();await busy($('#run-import-form button[type="submit"]'),async()=>{const records=await readRecords($('#run-file'),$('#run-file-status'));const form=new FormData(event.target);const result=await api('/api/runs/import',{name:form.get('name'),model:form.get('model'),track:form.get('track'),records});toast(`Run imported: ${result.name || result.id || 'saved'}.`);event.target.reset();$('#run-file-status').textContent='';await Promise.allSettled([loadRunsPage(),loadSummary()]);});});
  $('#judgment-file').addEventListener('change',async()=>{try{await readRecords($('#judgment-file'),$('#judgment-file-status'));}catch(e){$('#judgment-file-status').textContent=e.message;}});
  $('#judgment-import-form').addEventListener('submit',async event=>{event.preventDefault();await busy($('#judgment-import-form button'),async()=>{const records=await readRecords($('#judgment-file'),$('#judgment-file-status'));await api('/api/judgments/import',{records});toast('Judgments imported and validated.');event.target.reset();$('#judgment-file-status').textContent='';await Promise.allSettled([loadScorecards(),loadSummary()]);});});
  $('#preview-task').addEventListener('change',()=>{$('#preview-run-label').hidden=$('#preview-task').value!=='judge';});
  $('#preview-form').addEventListener('submit',previewJob);
  $('#close-dialog').addEventListener('click',()=>$('#detail-dialog').close());
  window.addEventListener('beforeunload',event=>{if(state.dirty||state.contractDirty){event.preventDefault();event.returnValue='';}});
}

async function init() {
  wire(); await Promise.allSettled([loadSummary(),loadFrames(true),loadRuns(),loadProvider()]);
}
init();
