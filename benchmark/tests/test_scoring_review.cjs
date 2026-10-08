const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {filterFrames, adjacentFrame} = require('../web/scoring-review.js');

const frames = [
  {frame_id:'b',recording_id:'record2',status:'judged',benchmark_index:0,tags:['v1.0']},
  {frame_id:'a',status:'judged',benchmark_index:80},
  {frame_id:'c',status:'unjudged',benchmark_index:null},
  {frame_id:'d',status:'stale',benchmark_index:90},
  {frame_id:'e',status:'ineligible',benchmark_index:null}
];
test('review distinguishes scored, unjudged and excluded frames without treating zero as missing', () => {
  assert.deepEqual(filterFrames(frames).map(r=>r.frame_id),['a','b']);
  assert.deepEqual(filterFrames(frames,'','excluded').map(r=>r.frame_id),['d','e']);
  assert.deepEqual(filterFrames(frames,'','all','lowest').map(r=>r.frame_id),['b','a','d','c','e']);
  assert.deepEqual(filterFrames(frames,'','all','highest').map(r=>r.frame_id),['d','a','b','c','e']);
  assert.deepEqual(filterFrames(frames,' RECORD2 ','all').map(r=>r.frame_id),['b']);
  assert.deepEqual(filterFrames(frames,'v1.0','judged').map(r=>r.frame_id),['b']);
  assert.equal(frames[0].frame_id,'b');
});
test('previous and next stay inside the filtered list and stop at its boundaries', () => {
  const rows=filterFrames(frames);
  assert.equal(adjacentFrame(rows,'a',-1),null);
  assert.equal(adjacentFrame(rows,'a',1).frame_id,'b');
  assert.equal(adjacentFrame(rows,'b',1),null);
  assert.equal(adjacentFrame(rows,'missing',1),null);
});

function harness() {
  const elements = new Map(), requests=[];
  const node = id => {
    if (!elements.has(id)) elements.set(id,{value:id==='score-review-filter'?'judged':id==='score-review-sort'?'frame':'',innerHTML:'',textContent:'',disabled:false,checked:false,handlers:{},
      addEventListener(type,handler){this.handlers[type]=handler;},classList:{toggle(){}},querySelectorAll(){return [];},focus(){},scrollIntoView(){}});
    return elements.get(id);
  };
  const deferred=url=>new Promise((resolve,reject)=>requests.push({url,resolve,reject}));
  const context=vm.createContext({document:{getElementById:node},window:{MousebenchScoringReviewView:{renderListItem:r=>r.frame_id,renderDetail:d=>d.frame.id}},
    URLSearchParams,history:{replaceState(){}},location:{hash:''},state:{page:'scoring-review'},api:deferred,
    empty:String,esc:String,showError:(el,error)=>{el.innerHTML=error.message;},switchPage(){},confirmDiscard:()=>false});
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../web/scoring-review.js'),'utf8'),context);
  return {elements,node,requests,controls:context.window.MousebenchScoringReview};
}
const tick=()=>new Promise(resolve=>setImmediate(resolve));
const response=(id,frame)=>({selected_run_id:id,selected_group_id:'g',runs:[{id}],groups:[{group_id:'g',run_ids:[id]}],frames:[{frame_id:frame,judgment_id:frame,status:'judged'}]});
test('late run responses cannot replace a newer selected run',async()=>{
  const h=harness();const one=h.controls.load(),two=h.controls.load();
  h.requests[1].resolve(response('new','frame-new'));await two;
  h.requests[0].resolve(response('old','frame-old'));await one;
  assert.equal(h.node('score-review-run').value,'new');
  assert.equal(h.requests.length,3);
  h.requests[2].resolve({frame:{id:'frame-new'}});await tick();
  assert.ok(h.node('score-review-detail').innerHTML.startsWith('frame-new'));
});
test('late frame evidence cannot appear under a different run or an empty filter',async()=>{
  const h=harness();const first=h.controls.load();h.requests[0].resolve(response('old','old-frame'));await first;
  const second=h.controls.load();h.requests[2].resolve(response('new','new-frame'));await second;
  h.requests[3].resolve({frame:{id:'new-frame'}});await tick();
  h.requests[1].resolve({frame:{id:'old-frame'}});await tick();
  assert.ok(h.node('score-review-detail').innerHTML.startsWith('new-frame'));
  h.node('score-review-search').value='absent';h.node('score-review-search').handlers.input();
  assert.match(h.node('score-review-detail').innerHTML,/No frames/);
  assert.equal(h.node('score-review-next').disabled,true);
});
test('changing filters during a run refresh cannot request an old-run frame',async()=>{
  const h=harness();const first=h.controls.load();h.requests[0].resolve(response('old','old-frame'));await first;
  const next=h.controls.load();
  assert.equal(h.node('score-review-group').disabled,true);
  const pending=h.requests.length;
  h.node('score-review-search').handlers.input();
  assert.equal(h.requests.length,pending);
  h.requests[2].resolve(response('new','new-frame'));await next;
  assert.equal(h.node('score-review-group').disabled,false);
});
