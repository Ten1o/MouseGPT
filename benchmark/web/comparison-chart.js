/* Shared, read-only comparison figures for the overview and run workspace. */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.MousebenchComparison = api;
})(typeof window !== 'undefined' ? window : null, function () {
  'use strict';
  const METRICS = Object.freeze({
    benchmark_index: {label: 'Overall benchmark index', unit: '/ 100'},
    factual_f1: {label: 'Factual F1', unit: '%'},
    factual_precision: {label: 'Factual precision', unit: '%'},
    required_fact_recall: {label: 'Required fact recall', unit: '%'},
    structure_score: {label: 'Description structure', unit: '%'},
    instruction_score: {label: 'Prompt following', unit: '%'},
  });
  const mounts = new WeakMap();
  let nextId = 0;
  const list = value => Array.isArray(value) ? value : [];
  const number = value => typeof value === 'number' && Number.isFinite(value);
  const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
  const validMetric = value => Object.hasOwn(METRICS, value) ? value : 'benchmark_index';
  const time = value => {const n = Date.parse(value || ''); return Number.isFinite(n) ? n : 0;};

  function orderedGroups(data) {
    const groups = list(data?.groups).map((group, index) => ({...group, comparison_number: group.comparison_number ?? index + 1}));
    const known = new Set(groups.map(group => group.group_id));
    for (const model of list(data?.models)) {
      if (known.has(model.group_id)) continue;
      known.add(model.group_id);
      groups.push({group_id: model.group_id, comparison_number: groups.length + 1, common_frame_ids: [],
        track: model.track, input_track: model.input_track, scoring_version: model.scoring_version});
    }
    return groups.sort((a, b) => time(b.created_at) - time(a.created_at)
      || Number(b.comparison_number) - Number(a.comparison_number)
      || String(a.group_id).localeCompare(String(b.group_id)));
  }

  function effort(model) {
    const status = model?.reasoning_effort_status;
    if (['not_requested', 'unrecorded', 'mixed'].includes(status)) return status;
    if (typeof model?.reasoning_effort === 'string' && model.reasoning_effort.trim()) return model.reasoning_effort.trim().toLowerCase();
    const match = String(model?.name || '').match(/\beffort\s*=\s*([\w-]+)/i);
    return match ? match[1].toLowerCase() : 'unspecified';
  }

  function label(model) {
    const original = String(model?.model || model?.name || model?.run_id || 'Unnamed model');
    const name = original.replace(/^gpt-/i, 'GPT-').replace(/-(sol|luna|astra)$/i, (_, word) => ' ' + word[0].toUpperCase() + word.slice(1).toLowerCase());
    const setting = effort(model);
    if (setting === 'unspecified' || setting === 'not_applicable' || setting === 'not_requested') return name;
    if (setting === 'unrecorded') return `${name} · effort unrecorded`;
    if (setting === 'mixed') return `${name} · mixed efforts`;
    return `${name} · ${setting}`;
  }

  function effortLabel(value) {
    return ({not_requested: 'Not requested', unrecorded: 'Not recorded', mixed: 'Mixed efforts', unspecified: 'Not recorded', not_applicable: 'Not applicable'})[value] || value;
  }

  function metricValue(model, metric = 'benchmark_index') {
    metric = validMetric(metric);
    // A missing matched-frame score must never silently become zero or an unmatched average.
    const value = metric === 'benchmark_index' ? model?.common_benchmark_index : model?.common_frame_dimension_means?.[metric];
    if (!number(value)) return null;
    const scaled = metric === 'benchmark_index' ? value : value * 100;
    return scaled >= 0 && scaled <= 100 ? scaled : null;
  }

  function groupModels(data, groupId) {
    return list(data?.models).filter(model => model.group_id === groupId);
  }

  function chartRows(data, groupId, options = {}) {
    const metric = validMetric(options.metric);
    const query = String(options.query || '').trim().toLocaleLowerCase();
    const selection = Array.isArray(options.selectedRuns) ? new Set(options.selectedRuns) : null;
    const rows = groupModels(data, groupId).filter(model => (!selection || selection.has(model.run_id))
      && (!options.effort || options.effort === 'all' || effort(model) === options.effort)
      && (!query || `${label(model)} ${model.name || ''} ${model.run_id || ''}`.toLocaleLowerCase().includes(query)))
      .map(model => ({model, label: label(model), value: metricValue(model, metric), rank: null}));
    const alphabetical = (a, b) => a.label.localeCompare(b.label, undefined, {numeric: true}) || String(a.model.run_id).localeCompare(String(b.model.run_id));
    rows.sort((a, b) => (a.value === null) - (b.value === null) || (b.value ?? 0) - (a.value ?? 0) || alphabetical(a, b));
    let previous = null;
    let rank = null;
    rows.forEach((row, index) => {
      if (row.value !== null) {if (row.value !== previous) rank = index + 1; row.rank = rank; previous = row.value;}
    });
    if (options.sort === 'asc') rows.sort((a, b) => (a.value === null) - (b.value === null) || (a.value ?? 0) - (b.value ?? 0) || alphabetical(a, b));
    if (options.sort === 'name') rows.sort(alphabetical);
    return rows;
  }

  function csvCell(value) {
    let text = value === null || value === undefined ? '' : String(value);
    // Keep model names safe when a CSV is opened directly in a spreadsheet.
    if (typeof value === 'string' && /^[\s]*[=+@-]/.test(text)) text = "'" + text;
    return /[",\r\n]/.test(text) ? '"' + text.replace(/"/g, '""') + '"' : text;
  }

  function csv(data, group, rows, metric = 'benchmark_index') {
    metric = validMetric(metric);
    const count = list(group?.common_frame_ids).length;
    const header = ['comparison_group', 'group_id', 'judge', 'run_id', 'model', 'reasoning_effort', 'metric', 'score_0_to_100', 'rank_among_visible_runs', 'common_frames', 'index_frames', 'all_scored_frames', 'benchmark_frames', 'scope'];
    const body = rows.map(row => [group.comparison_number, group.group_id, group.judge?.model, row.model.run_id, row.model.model, effort(row.model), metric, row.value, row.rank, count, row.model.common_index_count, row.model.evaluable_count, data.total_frames, 'fixed common frames of all runs in comparison group']);
    return [header, ...body].map(cells => cells.map(csvCell).join(',')).join('\r\n') + '\r\n';
  }

  function groupLabel(group) {
    const judge = String(group.judge?.model || 'Unknown judge');
    const conciseJudge = judge.startsWith('Codex current session') ? 'Codex' : judge;
    return `Comparison group ${group.comparison_number} · ${conciseJudge} · ${list(group.common_frame_ids).length} frames`;
  }

  function figureSVG(data, group, rows, metric, compact, interactive, id, displayWidth = 1000) {
    const narrow = number(displayWidth) && displayWidth > 0 && displayWidth < 700;
    const width = narrow ? Math.max(220, Math.round(displayWidth)) : 1000;
    const rowHeight = narrow ? (compact ? 52 : 56) : (compact ? 38 : 43);
    const top = narrow ? 98 : 83, height = top + 21 + rows.length * rowHeight;
    const x = narrow ? 28 : 294, plotWidth = narrow ? width - 30 : 604;
    const metricInfo = METRICS[validMetric(metric)];
    const title = `${metricInfo.label} ${metricInfo.unit}`;
    const commonCount = list(group.common_frame_ids).length;
    const judgeCaption = `${groupLabel(group)}${group.judge_reasoning_effort ? ' · judge effort ' + group.judge_reasoning_effort : ''}`;
    const description = `Comparison group ${group.comparison_number}. ${rows.length} visible runs, scored on the same ${commonCount} frames. Higher scores are better. Bars start at zero. Values use the stored matched-frame scores.`;
    const tickMarkup = [0, 20, 40, 60, 80, 100].map(tick => {
      const tickX = x + tick / 100 * plotWidth;
      return `${narrow ? '' : `<line x1="${tickX}" x2="${tickX}" y1="${top - 9}" y2="${height - 19}" stroke="${tick === 0 ? '#a6b5aa' : '#e6ebe4'}" stroke-width="1"/>`}<text x="${tickX}" y="${top - 13}" text-anchor="${narrow && tick === 100 ? 'end' : 'middle'}" fill="#69786e" font-size="11">${tick}</text>`;
    }).join('');
    const marks = rows.map((row, i) => {
      const y = top + i * rowHeight, labelY = y + (narrow ? 13 : 17);
      const available = row.value !== null;
      const value = available ? row.value.toFixed(2) : '—';
      const titleText = `${row.label}: ${available ? value + ' ' + metricInfo.unit : 'no matched score'}. ${commonCount} matched frames; ${row.model.evaluable_count ?? 0} total scored. ${interactive ? 'Open individual scoring review.' : ''}`;
      const actionable = interactive ? ` role="button" tabindex="0" data-chart-run="${escape(row.model.run_id)}" aria-label="${escape(titleText)}"` : '';
      const maxLabelLength = narrow ? Math.max(15, Math.floor((width - 91) / 7)) : 31;
      const visibleLabel = row.label.length > maxLabelLength ? row.label.slice(0, maxLabelLength - 1) + '…' : row.label;
      return `<g class="comparison-bar-row"${actionable}><title>${escape(titleText)}</title>
        <rect class="comparison-row-hit" x="0" y="${y - 4}" width="${width}" height="${rowHeight - 2}" rx="5" fill="transparent"/>
        <text x="${narrow ? 10 : 22}" y="${labelY}" text-anchor="middle" font-size="11" fill="#728078">${row.rank ?? '—'}</text>
        <text class="comparison-model-label" x="${narrow ? x : 43}" y="${labelY}" font-size="13" fill="#223733">${escape(visibleLabel)}</text>
        ${narrow ? `<rect x="${x}" y="${y + 23}" width="${plotWidth}" height="18" rx="3" fill="#edf1eb"/>` : ''}
        ${available ? `<rect x="${x}" y="${y + (narrow ? 23 : 0)}" width="${row.value / 100 * plotWidth}" height="${narrow ? 18 : 25}" rx="3" fill="#2e5b4c"/>` : ''}
        <text x="${narrow ? width - 2 : available ? x + row.value / 100 * plotWidth + 9 : x + 10}" y="${labelY}" text-anchor="${narrow ? 'end' : 'start'}" font-size="${narrow ? 13 : 12}" font-weight="600" fill="#244c3f">${value}</text>
      </g>`;
    }).join('');
    const source = {group_id: group.group_id, comparison_number: group.comparison_number, judge: group.judge,
      judge_reasoning_effort: group.judge_reasoning_effort || null, scoring_version: group.scoring_version,
      rubric_sha256: group.rubric_sha256, metric, common_frame_count: commonCount,
      common_frame_ids: list(group.common_frame_ids), scope: 'fixed group common frames', source: '/api/scorecards',
      generated_at: data.generated_at || null};
    const captionLimit = narrow ? Math.floor(width / 5.8) : 140;
    return `<svg class="comparison-svg${narrow ? ' comparison-svg-narrow' : ''}" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" role="group" aria-labelledby="${id}-title ${id}-description" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif">
      <title id="${id}-title">${escape(title)}</title><desc id="${id}-description">${escape(description)}</desc><metadata>${escape(JSON.stringify(source))}</metadata>
      <rect width="${width}" height="${height}" fill="#ffffff"/>
      <text x="0" y="19" font-size="14" font-weight="600" fill="#244c3f">${escape(title)}</text>
      <text x="${narrow ? 0 : width - 1}" y="${narrow ? 58 : 19}" text-anchor="${narrow ? 'start' : 'end'}" font-size="11" fill="#728078">${commonCount} matched frames · higher is better</text>
      <text x="0" y="39" font-size="11" fill="#728078">${escape(judgeCaption.length > captionLimit ? judgeCaption.slice(0, captionLimit - 1) + '…' : judgeCaption)}</text>
      ${tickMarkup}${marks}</svg>`;
  }

  function unmount(container) {
    const mounted = mounts.get(container);
    if (mounted) mounted.destroy();
  }

  function mount(container, options = {}) {
    if (!container) return {destroy() {}, getState() {return {};}};
    unmount(container);
    const data = options.data || {}, groups = orderedGroups(data), compact = Boolean(options.compact);
    const saved = options.state || {};
    const requestedGroup = options.groupId || saved.groupId;
    const firstGroup = groups.find(group => group.group_id === requestedGroup) || groups[0];
    const sameGroup = saved.groupId === firstGroup?.group_id;
    const allRuns = firstGroup ? groupModels(data, firstGroup.group_id).map(model => model.run_id) : [];
    const state = {
      groupId: firstGroup?.group_id || '', metric: compact ? 'benchmark_index' : validMetric(saved.metric),
      sort: compact ? 'desc' : ['desc', 'asc', 'name'].includes(saved.sort) ? saved.sort : 'desc',
      effort: compact || !sameGroup ? 'all' : saved.effort || 'all', query: compact || !sameGroup ? '' : String(saved.query || ''),
      selectedRuns: !compact && sameGroup && Array.isArray(saved.selectedRuns)
        ? saved.selectedRuns.filter(run => allRuns.includes(run)) : allRuns,
    };
    const id = `comparison-${++nextId}`;
    let destroyed = false, currentRows = [], resizeObserver = null;
    let displayWidth = number(container.clientWidth) && container.clientWidth > 0 ? container.clientWidth : 1000;
    const getState = () => ({...state, selectedRuns: [...state.selectedRuns]});
    const emit = () => {if (typeof options.onStateChange === 'function') options.onStateChange(getState());};
    const selected = value => value ? ' selected' : '';
    const group = () => groups.find(item => item.group_id === state.groupId);
    const figure = () => currentRows.length ? figureSVG(data, group(), currentRows, state.metric, compact, typeof options.onReview === 'function', id, displayWidth) : '<div class="empty compact">No runs match these filters.</div>';

    function resizeFigure() {
      const measured = Math.round(container.clientWidth || 0);
      if (destroyed || measured <= 0 || measured === displayWidth) return;
      displayWidth = measured;
      const host = container.querySelector('.comparison-figure-scroll');
      if (!host || !group()) return;
      // Resizing must not recreate controls or interrupt an in-progress model search.
      const active = typeof document !== 'undefined' ? document.activeElement : null;
      const focusedRun = active && host.contains(active) ? active.dataset?.chartRun : null;
      host.innerHTML = figure();
      if (focusedRun) [...host.querySelectorAll('[data-chart-run]')].find(item => item.dataset.chartRun === focusedRun)?.focus();
    }

    function draw() {
      if (destroyed) return;
      const current = group();
      if (!current) {container.innerHTML = '<div class="empty compact">No scored comparison group is available.</div>'; return;}
      const models = groupModels(data, state.groupId);
      const efforts = [...new Set(models.map(effort))];
      if (state.effort !== 'all' && !efforts.includes(state.effort)) state.effort = 'all';
      const commonCount = list(current.common_frame_ids).length;
      currentRows = chartRows(data, state.groupId, state);
      if (!commonCount) currentRows = currentRows.map(row => ({...row, value: null, rank: null}));
      const judge = current.judge?.model || 'Unknown judge';
      const controls = compact ? '' : `<div class="comparison-controls">
        <label class="comparison-group-control">Comparison group<select data-comparison-control="groupId">${groups.map(item => `<option value="${escape(item.group_id)}"${selected(item.group_id === state.groupId)}>${escape(groupLabel(item))}</option>`).join('')}</select></label>
        <label>Score<select data-comparison-control="metric">${Object.entries(METRICS).map(([value, metric]) => `<option value="${value}"${selected(state.metric === value)}>${escape(metric.label)}</option>`).join('')}</select></label>
        <label>Order<select data-comparison-control="sort"><option value="desc"${selected(state.sort === 'desc')}>Score: high to low</option><option value="asc"${selected(state.sort === 'asc')}>Score: low to high</option><option value="name"${selected(state.sort === 'name')}>Model name</option></select></label>
        <label>Reasoning effort<select data-comparison-control="effort"><option value="all">All efforts</option>${efforts.sort().map(value => `<option value="${escape(value)}"${selected(state.effort === value)}>${escape(effortLabel(value))}</option>`).join('')}</select></label>
        <label class="comparison-search-control">Find model<input type="search" data-comparison-control="query" value="${escape(state.query)}" placeholder="Model name or run" autocomplete="off"></label>
      </div>
      <div class="comparison-actions">
        <details class="comparison-model-picker"><summary>Choose models <span>${state.selectedRuns.length} / ${models.length}</span></summary>
          <div class="comparison-model-options"><div class="comparison-picker-actions"><button type="button" data-comparison-action="all" class="text-button">Select all</button><button type="button" data-comparison-action="none" class="text-button">Clear selection</button></div>
          ${models.slice().sort((a, b) => label(a).localeCompare(label(b))).map(model => `<label><input type="checkbox" data-comparison-model="${escape(model.run_id)}"${state.selectedRuns.includes(model.run_id) ? ' checked' : ''}><span>${escape(label(model))}</span></label>`).join('')}</div>
        </details>
        <button class="text-button" type="button" data-comparison-action="reset">Reset filters</button>
        <div class="comparison-export-actions"><button class="button secondary" type="button" data-comparison-action="csv"${currentRows.length ? '' : ' disabled'}>Export CSV</button><button class="button secondary" type="button" data-comparison-action="svg"${currentRows.length ? '' : ' disabled'}>Save figure</button></div>
      </div>`;
      const scope = `${commonCount} common frames${number(data.total_frames) ? ' / ' + data.total_frames + ' in the benchmark' : ''}`;
      const mismatch = state.metric === 'benchmark_index' && currentRows.some(row => row.value !== null && number(row.model.common_index_count) && row.model.common_index_count !== commonCount);
      container.innerHTML = `<section class="comparison-chart${compact ? ' comparison-chart-compact' : ''}" aria-label="Ranked model comparison">
        ${controls}
        <div class="comparison-caption"><p><strong>${escape(scope)}</strong> · ${currentRows.length} of ${models.length} runs shown</p><p>Judge: ${escape(judge)}${current.judge_reasoning_effort ? ' · ' + escape(current.judge_reasoning_effort) : ''}${current.created_at ? ' · Started ' + escape(String(current.created_at).slice(0, 10)) : ''}</p></div>
        ${!commonCount ? '<p class="comparison-caution">These runs have no common scored frames yet. A matched ranking is not available.</p>' : ''}
        ${mismatch ? '<p class="comparison-caution">Some runs have fewer valid index scores than common frames; inspect their coverage before comparing.</p>' : ''}
        <div class="comparison-figure-scroll">${figure()}</div>
        <div class="comparison-footnote"><p>${compact ? 'Select a model to review its individual scores. ' : 'Select a bar to review its individual scores. Model filters keep the same common-frame basis. '}Point estimates; differences alone do not establish statistical significance.</p>${compact && typeof options.onExplore === 'function' ? '<button type="button" class="text-button" data-comparison-action="explore">Explore comparison →</button>' : ''}</div>
        ${compact ? '' : `<details class="comparison-method"><summary>How to read this figure</summary><p>All scores use the ${commonCount} scored frames shared by every run in this comparison group. This fixed set stays the same when you hide models. Overall index is the mean of the stored per-frame indices, not a formula applied to averaged dimensions. The other scores are frame means shown as percentages. Ranking uses unrounded values; exact ties share a rank.</p><p>Comparison groups keep judge configuration, rubric and input track separate. The figure shows point estimates; an interval for the overall index is not available here. Paired uncertainty remains in the comparison details on the dashboard.</p><p class="comparison-source">Scoring: ${escape(current.scoring_version || 'not specified')} · Group ${escape(current.group_id)}</p></details>`}
      </section>`;
    }

    function updateControl(event) {
      const control = event.target.closest('[data-comparison-control]');
      if (!control || !container.contains(control)) return;
      const key = control.dataset.comparisonControl;
      if (key === 'query' && event.type === 'change') return;
      if (key !== 'query' && event.type === 'input') return;
      const focus = key === 'query' ? {start: control.selectionStart, end: control.selectionEnd} : null;
      state[key] = control.value;
      if (key === 'groupId') {state.selectedRuns = groupModels(data, state.groupId).map(model => model.run_id); state.effort = 'all'; state.query = '';}
      draw(); emit();
      container.querySelector(`[data-comparison-control="${key}"]`)?.focus();
      if (focus) {
        const input = container.querySelector('[data-comparison-control="query"]');
        input?.focus();
        // Search inputs do not support selection ranges in every browser.
        try {if (focus.start !== null) input?.setSelectionRange(focus.start, focus.end);} catch (_) {}
      }
    }

    function changeModel(event) {
      const input = event.target.closest('[data-comparison-model]');
      if (!input || !container.contains(input)) return;
      const chosen = new Set(state.selectedRuns), run = input.dataset.comparisonModel;
      if (input.checked) chosen.add(run); else chosen.delete(run);
      state.selectedRuns = [...chosen];
      draw(); emit();
      const picker = container.querySelector('.comparison-model-picker');
      if (picker) picker.open = true;
      const next = [...container.querySelectorAll('[data-comparison-model]')].find(item => item.dataset.comparisonModel === run);
      next?.focus();
    }

    function download(content, suffix, type) {
      const url = URL.createObjectURL(new Blob([content], {type}));
      const link = document.createElement('a');
      link.href = url; link.download = `mousegpt-group-${group().comparison_number}-${state.metric}.${suffix}`;
      document.body.appendChild(link); link.click(); link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    }

    function click(event) {
      const bar = event.target.closest('[data-chart-run]');
      if (bar && container.contains(bar)) {options.onReview?.(bar.dataset.chartRun, state.groupId); return;}
      const action = event.target.closest('[data-comparison-action]');
      if (!action || !container.contains(action)) return;
      const value = action.dataset.comparisonAction;
      if (value === 'explore') {options.onExplore?.(state.groupId); return;}
      if (value === 'csv') {download(csv(data, group(), currentRows, state.metric), 'csv', 'text/csv;charset=utf-8'); return;}
      if (value === 'svg') {download(figureSVG(data, group(), currentRows, state.metric, compact, false, id + '-export'), 'svg', 'image/svg+xml;charset=utf-8'); return;}
      if (value === 'none') state.selectedRuns = [];
      if (value === 'all' || value === 'reset') state.selectedRuns = groupModels(data, state.groupId).map(model => model.run_id);
      if (value === 'reset') {state.query = ''; state.effort = 'all'; state.sort = 'desc'; state.metric = 'benchmark_index';}
      draw(); emit();
      if (value === 'all' || value === 'none') container.querySelector('.comparison-model-picker').open = true;
    }

    function keydown(event) {
      const bar = event.target.closest('[data-chart-run]');
      if (bar && (event.key === 'Enter' || event.key === ' ')) {event.preventDefault(); options.onReview?.(bar.dataset.chartRun, state.groupId);}
    }

    container.addEventListener('input', updateControl);
    container.addEventListener('change', updateControl);
    container.addEventListener('change', changeModel);
    container.addEventListener('click', click);
    container.addEventListener('keydown', keydown);
    const instance = {getState, destroy() {
      if (destroyed) return;
      destroyed = true;
      container.removeEventListener('input', updateControl);
      container.removeEventListener('change', updateControl);
      container.removeEventListener('change', changeModel);
      container.removeEventListener('click', click);
      container.removeEventListener('keydown', keydown);
      resizeObserver?.disconnect();
      mounts.delete(container);
    }};
    mounts.set(container, instance);
    draw(); emit();
    if (typeof ResizeObserver !== 'undefined') {
      resizeObserver = new ResizeObserver(resizeFigure);
      resizeObserver.observe(container);
    }
    return instance;
  }

  return {METRICS, orderedGroups, effort, label, metricValue, chartRows, csv, csvCell, groupLabel, figureSVG, mount, unmount};
});
