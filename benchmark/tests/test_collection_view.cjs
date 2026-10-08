const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../web/app.js'), 'utf8');

function harness() {
  const elements = new Map([
    ['#frame-search', {value:' G4 '}], ['#status-filter', {value:'draft'}],
    ['#collection-filter', {value:'v1.1'}]
  ]);
  const requests = [], rendered = [], pages = [];
  const context = vm.createContext({
    document:{querySelector:selector=>elements.get(selector)||null,querySelectorAll:()=>[]},
    structuredClone, window:{},
    fetch:()=>{throw new Error('No network calls in this test.');},
    request:async url=>{requests.push(url);return [{id:'selected-legacy-frame',tags:['v1.1']}];},
    recordRender:frames=>rendered.push(JSON.parse(JSON.stringify(frames))), recordPage:page=>pages.push(page)
  });
  vm.runInContext(source.replace(/\ninit\(\);\s*$/, '')+`
    api=request;
    renderFrameList=()=>recordRender(state.frames);
    switchPage=recordPage;
    globalThis.controls={state,loadFrames,browseCollection,collectionBadges,renderModelImageAssets};
  `,context);
  return {elements,requests,rendered,pages,...context.controls};
}

test('catalog reads pass collection, status and text together and preserve unsaved review content',async()=>{
  const h=harness();
  h.state.selected='open-v1-frame'; h.state.dirty=true; h.state.contractDirty=true;
  h.state.editContent={sections:{Overall:'Unsaved human correction.'}};
  await h.loadFrames();
  const url=new URL(h.requests[0],'http://fixture');
  assert.equal(url.searchParams.get('q'),'G4');
  assert.equal(url.searchParams.get('status'),'draft');
  assert.equal(url.searchParams.get('tag'),'v1.1');
  assert.equal(h.state.selected,'open-v1-frame');
  assert.equal(h.state.dirty,true); assert.equal(h.state.contractDirty,true);
  assert.equal(h.state.editContent.sections.Overall,'Unsaved human correction.');
  assert.deepEqual(h.rendered,[[{id:'selected-legacy-frame',tags:['v1.1']}]]);
});

test('overview collection shortcut changes only the catalog scope, retaining active review edits',async()=>{
  const h=harness();
  h.state.selected='open-frame'; h.state.dirty=true;
  await h.browseCollection('v1.0');
  assert.deepEqual(h.pages,['frames']);
  assert.equal(h.elements.get('#collection-filter').value,'v1.0');
  assert.equal(h.elements.get('#frame-search').value,'');
  assert.equal(h.elements.get('#status-filter').value,'');
  assert.equal(new URL(h.requests[0],'http://fixture').searchParams.get('tag'),'v1.0');
  assert.equal(h.state.selected,'open-frame'); assert.equal(h.state.dirty,true);
});

test('rendered-output evidence has collapsed, local image links and preserves provenance labels',()=>{
  const h=harness();
  const html=h.renderModelImageAssets({assets:[
    {id:'local-image',role:'model_output_image',model_label:'checkpoint <old>',path:'objects/a.jpg',sha256:'sha',nas_path:'/archive/old/output_img/001.jpg'},
    {id:'remote-image',role:'model_output_image',model_label:'remote checkpoint',nas_path:'/archive/remote/001.jpg'},
    {id:'unrelated',role:'original_image',path:'objects/b.jpg',sha256:'sha'}
  ]});
  assert.match(html,/checkpoint &lt;old&gt;/);
  assert.match(html,/src="\/api\/assets\/local-image"/);
  assert.match(html,/loading="lazy"/);
  assert.match(html,/\/archive\/old\/output_img\/001.jpg/);
  assert.match(html,/not been materialized locally/);
  assert.ok(!html.includes('/api/assets/remote-image'));
  assert.ok(!html.includes('/api/assets/unrelated'));
  assert.ok(!/<details\b[^>]*\bopen\b/.test(html));
  assert.match(html,/not been transcribed/);
});

test('collection badges render multiple tags without trusting their markup',()=>{
  const h=harness(), html=h.collectionBadges({tags:['v1.0','<script>']});
  assert.match(html,/>v1.0<\/span>/);
  assert.match(html,/&lt;script&gt;/);
  assert.ok(!html.includes('<script>'));
  assert.equal(h.collectionBadges({}), '');
});
