const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const comparison = require('../web/comparison-chart.js');
const {scoreLabels} = require('../web/rubric-view.js');

const data = {
  total_frames:435,
  groups:[{group_id:'old',comparison_number:1,created_at:'2026-10-01',track:'historical'},
    {group_id:'new',comparison_number:3,created_at:'2026-10-08',track:'controlled'}],
  models:[{model:'Old',run_id:'r1',group_id:'old',track:'historical'},
    {model:'New',run_id:'r2',group_id:'new',track:'controlled',reasoning_effort:'max'}]
};

test('overview renders newer group first with a chart and folded statistics for each group',()=>{
  const node={addEventListener(){}};
  const context=vm.createContext({document:{getElementById:()=>node},
    window:{MousebenchRubric:{scoreLabels},MousebenchComparison:comparison},
    esc:v=>String(v??''),arr:v=>Array.isArray(v)?v:[],fmt:v=>String(v??0),
    pct:v=>Number.isFinite(v)?`${v*100}%`:'—',json:JSON.stringify});
  const source=fs.readFileSync(path.join(__dirname,'../web/overview.js'),'utf8')
    .replace('window.MousebenchOverview={load};','window.MousebenchOverview={load,results};')
    .replace(/\n  load\(\);\n\}\)\(\);\s*$/,'\n})();');
  vm.runInContext(source,context);
  const html=context.window.MousebenchOverview.results({dataset:{frames:435},results:{scorecards:data}});
  assert.ok(html.indexOf('Comparison group 3')<html.indexOf('Comparison group 1'));
  assert.match(html,/data-overview-comparison="new"/);
  assert.match(html,/data-overview-comparison="old"/);
  assert.match(html,/<details class="workflow-details"><summary>Model statistics/);
  assert.match(html,/New.*max/);
});

function appHarness() {
  const mounts=[],reviews=[],pages=[],scrolls=[];
  const element={scrollIntoView:()=>scrolls.push(true)};
  const context=vm.createContext({document:{querySelector:()=>element,querySelectorAll:()=>[]},
    window:{MousebenchComparison:{mount:(node,options)=>mounts.push(options)},
      MousebenchScoringReview:{open:(...args)=>reviews.push(args)}},recordPage:p=>pages.push(p)});
  const source=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8').replace(/\ninit\(\);\s*$/,'');
  vm.runInContext(source+`
    switchPage=recordPage;
    setRunTrack=(track)=>{state.runTrack=track;comparisonView={groupId:'old'};};
    renderScorecards=()=>renderRunComparison(state.scorecards);
    globalThis.controls={state,renderRunComparison,openRunComparison};
  `,context);
  return {...context.controls,mounts,reviews,pages,scrolls};
}

test('Runs chart follows the track filter and retains view state across refreshes',()=>{
  const h=appHarness();
  h.renderRunComparison(data);
  assert.deepEqual(Array.from(h.mounts[0].data.models,m=>m.run_id),['r2']);
  assert.deepEqual(Array.from(h.mounts[0].data.groups,g=>g.group_id),['new']);
  h.mounts[0].onStateChange({groupId:'new',metric:'factual_f1',query:'sol'});
  h.renderRunComparison(data);
  assert.equal(h.mounts[1].state.metric,'factual_f1');
  h.mounts[1].onReview('r2','new');
  assert.deepEqual(h.reviews,[['r2','new']]);
});

test('overview exploration selects the requested group despite a conflicting current track',()=>{
  const h=appHarness();
  h.openRunComparison('old',data);
  assert.equal(h.state.runTrack,'historical');
  assert.deepEqual(h.pages,['runs']);
  assert.equal(h.mounts[0].state.groupId,'old');
  assert.deepEqual(Array.from(h.mounts[0].data.models,m=>m.run_id),['r1']);
  assert.equal(h.scrolls.length,1);
});

test('legacy overview exploration recovers the track from the model card',()=>{
  const h=appHarness();
  h.openRunComparison('old',{...data,groups:[]});
  assert.equal(h.state.runTrack,'historical');
  assert.equal(h.mounts[0].state.groupId,'old');
});

test('a late scorecard response or error cannot replace the newest comparison',async()=>{
  const pending=[],renders=[],errors=[];
  const context=vm.createContext({document:{querySelector:()=>({})},window:{},
    request:()=>new Promise((resolve,reject)=>pending.push({resolve,reject})),
    recordRender:value=>renders.push(value),recordError:error=>errors.push(error)});
  const source=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8').replace(/\ninit\(\);\s*$/,'');
  vm.runInContext(source+`
    api=request;renderScorecards=()=>recordRender(state.scorecards);
    showError=(node,error)=>recordError(error);
    globalThis.controls={state,loadScorecards};
  `,context);
  const old=context.controls.loadScorecards(), current=context.controls.loadScorecards();
  pending[1].resolve({id:'current'});await current;
  pending[0].resolve({id:'stale'});await old;
  assert.equal(context.controls.state.scorecards.id,'current');
  assert.equal(renders.length,1);
  const failed=context.controls.loadScorecards(), latest=context.controls.loadScorecards();
  pending[3].resolve({id:'latest'});await latest;
  pending[2].reject(new Error('late error'));await failed;
  assert.equal(context.controls.state.scorecards.id,'latest');
  assert.equal(errors.length,0);
});

test('chart is reachable before the long prediction list and is loaded before its consumers',()=>{
  const html=fs.readFileSync(path.join(__dirname,'../web/index.html'),'utf8');
  assert.ok(html.indexOf('id="runs-comparison"')<html.indexOf('id="runs-list"'));
  assert.ok(html.indexOf('src="/comparison-chart.js"')<html.indexOf('src="/app.js"'));
  assert.match(html,/href="\/comparison-chart.css"/);
});
