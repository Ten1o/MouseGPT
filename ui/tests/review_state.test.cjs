const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function fixture() {
  const elements = {'#review-facts': {innerHTML: ''}, '#review-count': {textContent: ''}};
  const inputs = [];
  const sandbox = {
    state: {fields: {descriptions: '/mouse-a', out2: '/results'}, reviews: {'00000000': {'Overall-revised': 'unscoped legacy'}}},
    document: {querySelectorAll: () => inputs},
    $: selector => elements[selector],
    esc: value => String(value ?? ''),
    save() {}, toast() {}, download() {},
    val: name => sandbox.state.fields[name] || '',
    Object, Date, console
  };
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../review_state.js'), 'utf8'), sandbox);
  const data = (letter, text, hash=letter.repeat(64)) => ({session_id: letter.repeat(64),
    description_directory: '/mouse-' + letter, frame_id: '00000000', source_file: '00000000.json',
    source_sha256: hash, description: {Overall: text, Keywords: 'resting'}});
  const load = (letter, text, hash) => {
    sandbox.state.fields.descriptions = '/mouse-' + letter;
    sandbox.acceptReviewFrame(data(letter, text, hash), {session_id: letter.repeat(64),
      description_directory: '/mouse-' + letter, inputDirectory: '/mouse-' + letter});
    sandbox.renderFacts();
  };
  return {sandbox, elements, inputs, load};
}

test('same frame IDs retain separate corrections when switching sessions', () => {
  const {sandbox, elements, inputs, load} = fixture();
  load('a', 'Mouse A jumping');
  inputs.push({dataset: {review: 'Overall-revised'}, value: 'Corrected A'});
  sandbox.persistReview();
  sandbox.invalidateReview();
  inputs.length = 0;
  load('b', 'Mouse B resting');
  assert.ok(!elements['#review-facts'].innerHTML.includes('Corrected A'));
  inputs.push({dataset: {review: 'Overall-revised'}, value: 'Corrected B'});
  sandbox.persistReview();
  sandbox.invalidateReview();
  inputs.length = 0;
  load('a', 'Mouse A jumping');
  assert.ok(elements['#review-facts'].innerHTML.includes('Corrected A'));
  assert.ok(!elements['#review-facts'].innerHTML.includes('Corrected B'));
  assert.equal(sandbox.state.reviewSessions['b'.repeat(64)].reviews['00000000']['Overall-revised'], 'Corrected B');
  assert.equal(sandbox.state.reviews, undefined);
  assert.equal(sandbox.state.legacyReviews['00000000']['Overall-revised'], 'unscoped legacy');
});

test('saving posts only the currently selected source session', async () => {
  const {sandbox, inputs, load} = fixture();
  load('a', 'Mouse A');
  inputs.push({dataset: {review: 'Overall-revised'}, value: 'Corrected A'});
  sandbox.persistReview(); sandbox.invalidateReview(); inputs.length = 0;
  load('b', 'Mouse B');
  inputs.push({dataset: {review: 'Overall-revised'}, value: 'Corrected B'});
  let submitted;
  sandbox.api = async (route, body) => {
    if (route === 'review-source') return {session_id: 'b'.repeat(64), description_directory: '/mouse-b'};
    submitted = body;
    return {session_id: 'b'.repeat(64), reviewed_descriptions: '/results/b-snapshot/descriptions', frames: 1};
  };
  await sandbox.saveCurrentReview();
  assert.equal(submitted.session_id, 'b'.repeat(64));
  assert.equal(submitted.reviews['00000000'].original.Overall, 'Mouse B');
  assert.ok(!JSON.stringify(submitted).includes('Corrected A'));
});

test('changed source content archives the previous draft without applying it', () => {
  const {sandbox, elements, inputs, load} = fixture();
  load('a', 'Old content');
  inputs.push({dataset: {review: 'Overall-revised'}, value: 'Correction for old content'});
  sandbox.persistReview(); inputs.length = 0;
  load('a', 'New content', 'f'.repeat(64));
  assert.ok(!elements['#review-facts'].innerHTML.includes('Correction for old content'));
  assert.equal(sandbox.state.reviewSessions['a'.repeat(64)].previousDrafts.length, 1);
});

test('late preview responses cannot replace the selected frame image', async () => {
  const {sandbox, elements} = fixture();
  Object.assign(elements, {'#frame':{value:0}, '#preview-image':{}, '#preview-empty':{},
    '.preview-label':{}, '#velocity-metrics':{}});
  sandbox.location = {hash:'#stage2'};
  sandbox.ensureEnvironment = async () => {};
  let descriptionFrame;
  sandbox.loadReviewFrame = async frame => { descriptionFrame = frame; return {velocity:{}}; };
  const pending = new Map();
  sandbox.api = async (_route,body) => new Promise(resolve => pending.set(body.frame,resolve));
  const first = sandbox.loadFrame();
  await new Promise(setImmediate);
  elements['#frame'].value = 12;
  const second = sandbox.loadFrame();
  await new Promise(setImmediate);
  pending.get(12)({image:'frame-12'}); await second;
  pending.get(0)({image:'frame-0'}); await first;
  assert.equal(descriptionFrame,12);
  assert.equal(elements['#preview-image'].src,'frame-12');
  assert.equal(elements['.preview-label'].textContent,'Frame 00000012');
});
