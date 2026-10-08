(function () {
  'use strict';
  const list = value => Array.isArray(value) ? value : [];
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  function filterFrames(frames, query = '', status = 'judged', order = 'frame') {
    const search = query.trim().toLowerCase();
    return list(frames).filter(row => (!search || `${row.frame_id} ${row.recording_id || ''} ${list(row.tags).join(' ')}`.toLowerCase().includes(search)) &&
      (status === 'all' || (status === 'excluded' ? ['stale', 'ineligible'].includes(row.status) : row.status === status)))
      .slice().sort((a, b) => {
        if (order !== 'frame') {
          const av = a.benchmark_index, bv = b.benchmark_index;
          if (finite(av) !== finite(bv)) return finite(av) ? -1 : 1;
          if (finite(av) && av !== bv) return order === 'lowest' ? av - bv : bv - av;
        }
        return String(a.frame_id).localeCompare(String(b.frame_id));
      });
  }
  function adjacentFrame(rows, selected, step) {
    const index = rows.findIndex(row => row.frame_id === selected);
    return index < 0 ? null : rows[index + step] || null;
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = {filterFrames, adjacentFrame};
  if (typeof window === 'undefined' || typeof document === 'undefined') return;

  const review = {runId: '', groupId: '', selected: '', data: null, visible: [], request: 0, detailRequest: 0, detail: null, explicitRun: false};
  const el = id => document.getElementById(id);
  const view = () => window.MousebenchScoringReviewView;
  function saveLocation() {
    if (state.page !== 'scoring-review') return;
    const params = new URLSearchParams();
    if (review.runId) params.set('run', review.runId);
    if (review.groupId) params.set('group', review.groupId);
    if (review.selected) params.set('frame', review.selected);
    if (state.runTrack) params.set('track', state.runTrack);
    params.set('filter', el('score-review-filter').value);
    params.set('order', el('score-review-sort').value);
    history.replaceState(null, '', `#scoring-review${params.size ? '?' + params : ''}`);
  }
  function navigation() {
    const index = review.visible.findIndex(row => row.frame_id === review.selected);
    el('score-review-prev').disabled = !adjacentFrame(review.visible, review.selected, -1);
    el('score-review-next').disabled = !adjacentFrame(review.visible, review.selected, 1);
    el('score-review-position').textContent = index < 0 ? '' : `${index + 1} of ${review.visible.length} in this view`;
    el('score-review-frame-title').textContent = review.selected || 'Select a frame';
  }
  function bottomNavigation() {
    return `<div class="score-review-bottom score-review-nav"><button type="button" class="button secondary" data-score-step="-1" ${adjacentFrame(review.visible,review.selected,-1)?'':'disabled'}>← Previous frame</button><button type="button" class="button primary" data-score-step="1" ${adjacentFrame(review.visible,review.selected,1)?'':'disabled'}>Next frame →</button></div>`;
  }
  function clearDetail(message) {
    review.detailRequest++;
    review.detail = null;
    el('score-review-detail').innerHTML = empty(message);
    navigation();
  }
  async function select(row) {
    if (!row) return;
    review.selected = row.frame_id;
    const request = ++review.detailRequest;
    navigation();
    el('score-review-list').innerHTML = review.visible.map(item => view().renderListItem(item, item.frame_id === review.selected)).join('');
    el('score-review-detail').innerHTML = empty('Loading the saved answer and grading evidence…');
    saveLocation();
    const path = row.judgment_id ? `/api/scoring-review/${encodeURIComponent(row.judgment_id)}` :
      `/api/scoring-review/prediction?${new URLSearchParams({run_id: review.runId, frame_id: row.frame_id})}`;
    try {
      const detail = await api(path);
      if (request !== review.detailRequest) return;
      review.detail = detail;
      el('score-review-detail').innerHTML = view().renderDetail(detail) + bottomNavigation();
      el('score-review-detail').classList.toggle('score-show-problems', el('score-review-problems').checked);
      // Each newly selected frame starts its evidence pane at the top.
      el('score-review-detail').querySelectorAll('.score-evidence').forEach(node => { node.scrollTop = 0; });
    } catch (error) {
      if (request === review.detailRequest) showError(el('score-review-detail'), error, () => select(row));
    }
  }
  function renderList() {
    review.visible = filterFrames(review.data?.frames, el('score-review-search').value, el('score-review-filter').value, el('score-review-sort').value);
    el('score-review-count').textContent = `${review.visible.length} frames shown`;
    el('score-review-list').innerHTML = review.visible.length ? review.visible.map(row => view().renderListItem(row, row.frame_id === review.selected)).join('') : empty('No matching frames. Try another filter.');
    const selected = review.visible.find(row => row.frame_id === review.selected) || review.visible[0];
    if (selected) select(selected);
    else { review.selected = ''; clearDetail('No frames in this view. Choose All saved outputs or another run.'); saveLocation(); }
  }
  async function load() {
    const request = ++review.request;
    el('score-review-track').value = state.runTrack || 'all';
    review.data = null; review.visible = [];
    el('score-review-group').disabled = true;
    clearDetail('Loading scoring review…');
    el('score-review-prev').disabled = el('score-review-next').disabled = true;
    el('score-review-list').innerHTML = empty('Loading frames…');
    el('score-review-status').textContent = 'Loading saved assessments…';
    try {
      const params = new URLSearchParams();
      if (review.runId) params.set('run_id', review.runId);
      if (review.groupId) params.set('group_id', review.groupId);
      let data = await api('/api/scoring-review?' + params);
      if (request !== review.request) return;
      const allRuns = list(data.runs);
      const requestedRun = allRuns.find(run => run.id === review.runId);
      // An explicit link to a run takes priority over the last browsing filter.
      if (review.explicitRun && requestedRun && state.runTrack && state.runTrack !== 'all' && requestedRun.track !== state.runTrack) {
        setRunTrack(['controlled','historical'].includes(requestedRun.track) ? requestedRun.track : 'all', {reloadReview: false});
      }
      review.explicitRun = false;
      const runs = allRuns.filter(run => !state.runTrack || state.runTrack === 'all' || run.track === state.runTrack);
      el('score-review-track-count').textContent = `${runs.length} of ${allRuns.length} runs shown · Shared with Step 03`;
      el('score-review-run').innerHTML = runs.length ? runs.map(run => `<option value="${esc(run.id)}">${esc(run.name || run.model || run.id)} · ${esc(run.track || 'Unknown track')} · ${run.judgment_count || 0} assessments</option>`).join('') : '<option value="">No matching runs</option>';
      el('score-review-run').disabled = !runs.length;
      if (!runs.length) {
        review.runId = ''; review.groupId = ''; review.selected = '';
        el('score-review-group').innerHTML = '<option value="">No assessments</option>';
        el('score-review-count').textContent = '0 frames shown';
        el('score-review-status').textContent = 'No runs match this run type. Choose All to see other saved runs.';
        el('score-review-list').innerHTML = empty('No matching runs.');
        clearDetail('Choose another run type to review saved outputs.');
        saveLocation();
        return;
      }
      if (!runs.some(run => run.id === data.selected_run_id)) {
        review.runId = runs[0].id; review.groupId = ''; review.selected = '';
        el('score-review-filter').value = 'all';
        data = await api('/api/scoring-review?' + new URLSearchParams({run_id: review.runId}));
        if (request !== review.request) return;
        if (!runs.some(run => run.id === data.selected_run_id)) throw new Error('Selected run does not match the run type. Refresh review to try again.');
      }
      review.data = data;
      review.runId = data.selected_run_id || '';
      review.groupId = data.selected_group_id || '';
      el('score-review-run').value = review.runId;
      const groups = list(data.groups).filter(group => !group.run_ids || group.run_ids.includes(review.runId));
      el('score-review-group').innerHTML = groups.length ? groups.map((group, index) => `<option value="${esc(group.group_id)}">${esc(group.rubric_name || group.protocol_version || 'Rubric')} · ${esc(group.judge?.model || 'Judge')} · ${index + 1}</option>`).join('') : '<option value="">No assessments for this run</option>';
      el('score-review-group').value = review.groupId;
      el('score-review-group').disabled = !groups.length;
      if (!groups.length && el('score-review-filter').value === 'judged') el('score-review-filter').value = 'all';
      const frames = list(data.frames), count = name => frames.filter(frame => frame.status === name).length;
      el('score-review-status').textContent = `${count('judged')} scored · ${count('unjudged')} not yet scored · ${count('stale') + count('ineligible')} excluded / stale · ${frames.length} saved frame outputs. Automated grades remain provisional until judge calibration.`;
      renderList();
    } catch (error) {
      if (request !== review.request) return;
      review.visible = []; review.selected = ''; review.data = null;
      navigation(); el('score-review-status').textContent = 'Scoring review could not be loaded.';
      showError(el('score-review-detail'), error, load);
      el('score-review-list').innerHTML = empty('Frame list unavailable.');
    }
  }
  function open(runId, groupId) {
    review.runId = runId || ''; review.groupId = groupId || ''; review.selected = '';
    review.explicitRun = Boolean(runId);
    el('score-review-search').value = ''; el('score-review-filter').value = 'judged';
    switchPage('scoring-review');
  }
  function move(step) {
    const row = adjacentFrame(review.visible, review.selected, step);
    if (row) {
      select(row);
      el('score-review-frame-title').focus({preventScroll: true});
      el('score-review-frame-title').scrollIntoView({block: 'start', behavior: 'auto'});
    }
  }
  function init() {
    el('score-review-track').value = state.runTrack || 'all';
    el('score-review-track').addEventListener('change', () => {
      review.explicitRun = false; review.selected = '';
      el('score-review-filter').value = 'all';
      setRunTrack(el('score-review-track').value);
    });
    el('score-review-refresh').addEventListener('click', load);
    el('score-review-run').addEventListener('change', () => {
      review.runId = el('score-review-run').value; review.groupId = ''; review.selected = '';
      el('score-review-search').value = ''; el('score-review-filter').value = 'all'; load();
    });
    el('score-review-group').addEventListener('change', () => { review.groupId = el('score-review-group').value; load(); });
    for (const id of ['score-review-search', 'score-review-filter', 'score-review-sort']) el(id).addEventListener(id === 'score-review-search' ? 'input' : 'change', renderList);
    el('score-review-list').addEventListener('click', event => {
      const button = event.target.closest('[data-score-frame]');
      if (button) select(review.visible.find(row => row.frame_id === button.dataset.scoreFrame));
    });
    el('score-review-prev').addEventListener('click', () => move(-1));
    el('score-review-next').addEventListener('click', () => move(1));
    el('score-review-problems').addEventListener('change', () => el('score-review-detail').classList.toggle('score-show-problems', el('score-review-problems').checked));
    el('score-review-detail').addEventListener('click', async event => {
      const step = event.target.closest('[data-score-step]');
      if (step) { move(Number(step.dataset.scoreStep)); return; }
      const claim = event.target.closest('[data-score-claim]');
      if (claim) {
        event.preventDefault();
        el('score-review-problems').checked = false;
        el('score-review-detail').classList.remove('score-show-problems');
        document.getElementById(`score-claim-${encodeURIComponent(claim.dataset.scoreClaim)}`)?.scrollIntoView({block: 'center', behavior: 'smooth'});
        return;
      }
      const button = event.target.closest('[data-review-gt]');
      if (button && confirmDiscard()) { switchPage('frames'); await selectFrame(button.dataset.reviewGt); }
    });
    if (location.hash.startsWith('#scoring-review')) {
      const params = new URLSearchParams(location.hash.split('?')[1] || '');
      review.runId = params.get('run') || ''; review.groupId = params.get('group') || ''; review.selected = params.get('frame') || '';
      const track = params.get('track');
      if (['all','controlled','historical'].includes(track)) setRunTrack(track, {reloadReview: false});
      else review.explicitRun = Boolean(review.runId);
      const filter = params.get('filter');
      el('score-review-filter').value = ['judged','all','unjudged','excluded'].includes(filter) ? filter : review.selected ? 'all' : 'judged';
      const order = params.get('order');
      if (['frame','lowest','highest'].includes(order)) el('score-review-sort').value = order;
      switchPage('scoring-review');
    }
  }
  window.MousebenchScoringReview = {load, open};
  init();
})();
