const test = require('node:test');
const assert = require('node:assert/strict');
const chart = require('../web/comparison-chart.js');

const data = {
  total_frames: 435,
  groups: [
    {group_id: 'old', comparison_number: 1, created_at: '2026-10-07T12:00:00Z', common_frame_ids: ['a'], judge: {model: 'Codex'}},
    {group_id: 'latest', comparison_number: 3, created_at: '2026-10-08T14:00:00Z', common_frame_ids: ['a', 'b'], judge: {model: 'GPT judge'}},
    {group_id: 'pilot', comparison_number: 2, created_at: '2026-10-08T10:00:00Z', common_frame_ids: ['a'], judge: {model: 'Codex'}},
  ],
  models: [
    {group_id: 'latest', run_id: 'a', model: 'gpt-6-sol', reasoning_effort: 'max', common_benchmark_index: 82.54, benchmark_index: 90, common_index_count: 2, common_frame_dimension_means: {factual_f1: 0.87}, evaluable_count: 435},
    {group_id: 'latest', run_id: 'b', model: 'gpt-6-sol', name: 'Azure · effort=low · now', common_benchmark_index: 75.25, common_index_count: 2, common_frame_dimension_means: {factual_f1: 0.81}, evaluable_count: 435},
    {group_id: 'old', run_id: 'old', model: 'old-model', common_benchmark_index: 100},
    {group_id: 'latest', run_id: 'missing', model: 'missing', common_benchmark_index: null, benchmark_index: 95},
  ],
};

test('groups use creation order, stable comparison numbers and no mutation', () => {
  assert.deepEqual(chart.orderedGroups(data).map(g => g.comparison_number), [3, 2, 1]);
  assert.deepEqual(data.groups.map(g => g.comparison_number), [1, 3, 2]);
  assert.deepEqual(chart.orderedGroups({groups: [{group_id: 'one'}, {group_id: 'two'}]}).map(g => g.group_id), ['two', 'one']);
  assert.deepEqual(chart.orderedGroups({groups: [], models: [{group_id: 'legacy'}]}).map(g => g.group_id), ['legacy']);
});

test('labels distinguish reasoning efforts with metadata precedence', () => {
  assert.equal(chart.label(data.models[0]), 'GPT-6 Sol · max');
  assert.equal(chart.label(data.models[1]), 'GPT-6 Sol · low');
  assert.equal(chart.label({model: 'gpt-4o'}), 'GPT-4o');
  assert.equal(chart.effort({reasoning_effort: 'high', name: 'effort=low'}), 'high');
  assert.equal(chart.effort({reasoning_effort_status: 'unrecorded', name: 'effort=low'}), 'unrecorded');
  assert.equal(chart.label({model: 'gpt-5', reasoning_effort_status: 'unrecorded'}), 'GPT-5 · effort unrecorded');
  assert.equal(chart.label({model: 'gpt-4o', reasoning_effort_status: 'not_requested'}), 'GPT-4o');
});

test('index comes from stored matched index, fractional dimensions scale once', () => {
  assert.equal(chart.metricValue(data.models[0]), 82.54);
  assert.equal(chart.metricValue(data.models[0], 'factual_f1'), 87);
  assert.equal(chart.metricValue({common_benchmark_index: 0}), 0);
  assert.equal(chart.metricValue(data.models[3]), null);
  assert.equal(chart.metricValue({common_benchmark_index: '90'}), null);
  assert.equal(chart.metricValue({common_benchmark_index: Infinity}), null);
  assert.equal(chart.metricValue({common_frame_dimension_means: {factual_f1: 1.1}}, 'factual_f1'), null);
});

test('rank isolates judge groups, sorts descending, and keeps missing last', () => {
  const rows = chart.chartRows(data, 'latest');
  assert.deepEqual(rows.map(r => r.model.run_id), ['a', 'b', 'missing']);
  assert.deepEqual(rows.map(r => r.rank), [1, 2, null]);
  assert.deepEqual(chart.chartRows(data, 'latest', {sort: 'asc'}).map(r => r.model.run_id), ['b', 'a', 'missing']);
});

test('filters do not recalculate scores or substitute unmatched scores', () => {
  const rows = chart.chartRows(data, 'latest', {selectedRuns: ['b'], query: 'sol', effort: 'low'});
  assert.equal(rows.length, 1);
  assert.equal(rows[0].value, 75.25);
  assert.equal(chart.chartRows(data, 'latest', {selectedRuns: []}).length, 0);
  assert.equal(chart.chartRows(data, 'latest', {effort: 'medium'}).length, 0);
  assert.equal(chart.chartRows(data, 'nonexistent').length, 0);
});

test('ties use competition ranks from full precision', () => {
  const tied = {models: [70, 80, 80, 79.999].map((v, i) => ({model: `m${i}`, run_id: `${i}`, group_id: 'g', common_benchmark_index: v}))};
  assert.deepEqual(chart.chartRows(tied, 'g').map(r => r.rank), [1, 1, 3, 4]);
});

test('CSV uses full precision and includes denominator, group and judge', () => {
  const result = chart.csv(data, data.groups[1], chart.chartRows(data, 'latest'));
  assert.match(result, /score_0_to_100/);
  assert.match(result, /82\.54,1,2,2,435,435/);
  assert.match(result, /3,latest,GPT judge,a,gpt-6-sol,max/);
  assert.equal(chart.csvCell('a,"b"\nc'), '"a,""b""\nc"');
  assert.equal(chart.csvCell('=HYPERLINK("x")'), '"\'=HYPERLINK(""x"")"');
  assert.equal(chart.csvCell(null), '');
});

test('SVG has zero baseline and escaped model text with independent export styling', () => {
  const rows = chart.chartRows(data, 'latest');
  rows[0].label = '<script>alert(1)</script>';
  const svg = chart.figureSVG(data, data.groups[1], rows, 'benchmark_index', false, true, 'test');
  assert.match(svg, />0<\/text>/);
  assert.match(svg, /&lt;script&gt;/);
  assert.doesNotMatch(svg, /<script>/);
  assert.match(svg, /role="button" tabindex="0"/);
  assert.match(svg, /fixed group common frames/);
  assert.match(svg, /fill="#2e5b4c"/);
  assert.match(svg, /82\.54/);
});

test('narrow figure keeps physical font size, ranks, values and bars within the viewport', () => {
  const rows = chart.chartRows(data, 'latest');
  const svg = chart.figureSVG(data, data.groups[1], rows, 'benchmark_index', false, true, 'small', 400);
  assert.match(svg, /viewBox="0 0 400 /);
  assert.match(svg, /comparison-svg-narrow/);
  assert.match(svg, /class="comparison-model-label" x="28" y="111" font-size="13"/);
  assert.match(svg, /<text x="398" y="111" text-anchor="end" font-size="13"/);
  assert.match(svg, /x="28" y="121" width="370" height="18"/);
  assert.match(svg, /82\.54/);
  assert.equal((svg.match(/class="comparison-bar-row"/g) || []).length, 3);
  const exported = chart.figureSVG(data, data.groups[1], rows, 'benchmark_index', false, false, 'export');
  assert.match(exported, /viewBox="0 0 1000 /);
  assert.doesNotMatch(exported, /comparison-svg-narrow/);
});

test('mount preserves filter state, explicit navigation resets selection, and remount removes listeners', () => {
  const listeners = new Map();
  const container = {
    innerHTML: '',
    addEventListener(name, fn) {if (!listeners.has(name)) listeners.set(name, new Set()); listeners.get(name).add(fn);},
    removeEventListener(name, fn) {listeners.get(name)?.delete(fn);},
  };
  let emitted;
  const first = chart.mount(container, {data, state: {groupId: 'latest', metric: 'factual_f1', sort: 'asc', effort: 'low', query: 'sol', selectedRuns: ['b']}, onStateChange: state => emitted = state});
  assert.equal(first.getState().metric, 'factual_f1');
  assert.deepEqual(first.getState().selectedRuns, ['b']);
  assert.equal(emitted.query, 'sol');
  const second = chart.mount(container, {data, groupId: 'old', state: first.getState()});
  assert.equal(second.getState().groupId, 'old');
  assert.deepEqual(second.getState().selectedRuns, ['old']);
  assert.equal(second.getState().query, '');
  assert.equal(listeners.get('click').size, 1);
  assert.equal(listeners.get('change').size, 2);
  first.destroy();
  assert.equal(listeners.get('click').size, 1);
  chart.unmount(container);
  assert.equal([...listeners.values()].reduce((sum, set) => sum + set.size, 0), 0);
});
