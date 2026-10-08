const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const {renderRubric,scoreLabels}=require('../web/rubric-view.js');
const fields=['Overall','Head','Limb','Torso','Others','Keywords'];
const anchors=prefix=>Object.fromEntries([0,.25,.5,.75,1].map(score=>[score,`${prefix} anchor ${score}`]));
function fixture(){return {id:'selected-graded-version',status:'draft_requires_calibration',schema_version:'mousebench.rubric.v1',task:'Describe the synchronized mouse frame.',fields,
  correctness_score_anchors:anchors('Correctness'),coverage_score_anchors:anchors('Coverage'),structure_score_anchors:anchors('Structure'),instruction_score_anchors:anchors('Instruction'),
  section_scope:Object.fromEntries(fields.map(f=>[f,`Scope ${f}`])),scoring:{version:'mousebench.graded-factual-f1.v2',primary:'graded factual F1',field_weights:{Overall:2,Head:1,Limb:1,Torso:1,Others:.5,Keywords:1},unknown_gt_weight:1,precision:'Weighted claim correctness / asserted weight',recall:'Weighted coverage / required weight',f1:'2 * P * R / (P + R)',index:{label:'Exploratory index',version:'mousebench.graded-quality.v2',formula:'100 * F1 * (0.8 + 0.1 * S + 0.1 * I)'},failure_policy:'Report missing judgments separately'},
  required_fact_coverage:'Only licensed evidence earns credit.',graded_examples:[{dimension:'correctness',evidence:'Both hind paws touch the floor.',candidate:'One hind paw is lifted.',score:.25,rationale:'Wrong contact for one paw.'}],
  analytic_criteria:Array.from({length:8},(_,i)=>({id:`criterion_${i}`,label:`Criterion ${i}`,group:'grounding',allow_not_assessable:i<3,definition:`Definition ${i}`})),rules:['No temporal inference from one image.'],claim_extraction:['Split compound claims.'],analytic_criteria_policy:{aggregation:'Counts only; no additive points'},calibration:{status:'not_started',release_gate:'Require held-out human evidence.'},provenance:{design:'Provisional design'}};}

test('selected rubric renders all four five-level scales, exact field weights and examples as readable tables',()=>{
  const r=fixture(),original=JSON.stringify(r),html=renderRubric(r,{id:'rubric-8',status:'draft'});
  for(const prefix of ['Correctness','Coverage','Structure','Instruction'])for(const score of [0,.25,.5,.75,1])assert.ok(html.includes(`${prefix} anchor ${score}`));
  assert.match(html,/<td>Overall<\/td><td>Scope Overall<\/td><td>2<\/td>/);
  assert.match(html,/<td>Others<\/td><td>Scope Others<\/td><td>0.5<\/td>/);
  assert.match(html,/Both hind paws touch the floor/);
  assert.match(html,/Wrong contact for one paw/);
  assert.match(html,/2 \* P \* R \/ \(P \+ R\)/);
  assert.match(html,/Saved version: rubric-8/);
  assert.match(html,/mousebench.graded-factual-f1.v2/);
  assert.match(html,/illustrations, not model results/);
  assert.equal(JSON.stringify(r),original);
});

test('full analytic definitions and operational rules remain readable in closed disclosures',()=>{
  const html=renderRubric(fixture());
  assert.match(html,/<details class="workflow-details rubric-advanced"><summary>Analytic checklist · 8 criteria/);
  for(let i=0;i<8;i++)assert.match(html,new RegExp(`Definition ${i}`));
  for(const text of ['No temporal inference','Split compound claims','Report missing judgments separately','Require held-out human evidence','Provisional design'])assert.ok(html.includes(text));
  assert.doesNotMatch(html,/<pre|<textarea|<details[^>]+open/);
});

test('legacy selected rubric keeps binary verdicts and does not invent graded anchors or weights',()=>{
  const html=renderRubric({id:'legacy-rubric',fields,claim_verdicts:{supported:'Source supports claim',contradicted:'Source contradicts claim',unverifiable:'Cannot decide'},scoring:{primary:'grounded factual F1',precision:'Supported / all'},structure_score_anchors:anchors('Old structure')},{status:'frozen'});
  assert.match(html,/Claim verdicts · legacy scoring/);
  assert.match(html,/Source supports claim/);
  assert.match(html,/grounded factual F1/);
  assert.match(html,/pill approved/);
  assert.doesNotMatch(html,/Factual quality · five-level|<th scope="col">Weight|Correctness anchor/);
});

test('untrusted rubric fields, anchors, provenance and metadata are escaped',()=>{
  const bad='<img src=x onerror="evil()">';const r=fixture();r.id=bad;r.task=bad;r.correctness_score_anchors['1']=bad;r.analytic_criteria[0].definition=bad;r.graded_examples[0].candidate=bad;r.provenance.design=bad;
  const html=renderRubric(r,{id:bad,status:bad});
  assert.doesNotMatch(html,/<img|<script|onerror="evil/);
  assert.ok(html.includes('&lt;img src=x onerror=&quot;evil()&quot;&gt;'));
});

test('metric labels distinguish graded scoring from legacy and unknown older versions',()=>{
  assert.deepEqual(scoreLabels({scoring_version:'mousebench.graded-factual-f1.v2'}),{graded:true,version:'mousebench.graded-factual-f1.v2',f1:'Graded factual F1',precision:'Graded correctness',recall:'Graded required coverage',versionLabel:'Graded scoring v2'});
  assert.equal(scoreLabels({}).f1,'Grounded factual F1');
  assert.equal(scoreLabels({}).recall,'Required recall');
  assert.equal(scoreLabels({scoring_version:'mousebench.factual.v1'}).graded,false);
});

test('workflow preview follows selected version and unsaved JSON while keeping invalid edits intact',()=>{
  const nodes=new Map();const node=id=>{if(!nodes.has(id))nodes.set(id,{value:'',innerHTML:'',textContent:'',disabled:false,addEventListener(){}});return nodes.get(id)};
  const rubric={id:'saved-1',status:'draft',content:fixture(),sha256:'hash'};
  const context=vm.createContext({document:{getElementById:node},window:{MousebenchRubric:{renderRubric}},json:v=>JSON.stringify(v),esc:v=>String(v),state:{}});
  const source=fs.readFileSync(path.join(__dirname,'../web/workflow.js'),'utf8').replace('window.MousebenchWorkflow={load,openTask}; initWorkflow();','window.MousebenchWorkflow={load,openTask}; globalThis.controls={workflow,setRubric,renderRubricPreview};');
  vm.runInContext(source,context);
  context.controls.workflow.data={rubrics:[rubric]};context.controls.setRubric(rubric);
  assert.match(node('rubric-readable').innerHTML,/selected-graded-version/);
  const edited=fixture();edited.id='unsaved-change';node('rubric-json').value=JSON.stringify(edited);context.controls.workflow.rubricDirty=true;context.controls.renderRubricPreview();
  assert.match(node('rubric-readable').innerHTML,/unsaved-change/);
  assert.match(node('rubric-readable').innerHTML,/Unsaved draft/);
  node('rubric-json').value='{ broken';context.controls.renderRubricPreview();
  assert.match(node('rubric-readable').innerHTML,/Readable preview unavailable/);
  assert.equal(node('rubric-json').value,'{ broken');
  assert.equal(context.controls.workflow.rubricDirty,true);
});

test('overview keeps legacy and graded labels separate and shows major-error rate only when available',()=>{
  const node={addEventListener(){}};
  const context=vm.createContext({document:{getElementById:()=>node},window:{MousebenchRubric:{scoreLabels},MousebenchComparison:require('../web/comparison-chart.js')},esc:v=>String(v??''),arr:v=>Array.isArray(v)?v:[],fmt:v=>String(v??0),pct:v=>Number.isFinite(v)?`${Math.round(v*100)}%`:'—',json:JSON.stringify});
  const source=fs.readFileSync(path.join(__dirname,'../web/overview.js'),'utf8').replace('window.MousebenchOverview={load};','window.MousebenchOverview={load,results};').replace(/\n  load\(\);\n\}\)\(\);\s*$/,'\n})();');
  vm.runInContext(source,context);
  const base={dataset:{frames:10},results:{scorecards:{models:[{model:'Legacy',group_id:'old',dimension_means:{factual_f1:.5},evaluable_count:1},{model:'Graded',group_id:'new',scoring_version:'mousebench.graded-factual-f1.v2',dimension_means:{factual_f1:.75,major_error_rate:0},evaluable_count:1}],groups:[]}}};
  const html=context.window.MousebenchOverview.results(base);
  assert.match(html,/<span>Grounded factual F1<\/span>/);
  assert.match(html,/<span>Graded factual F1<\/span>/);
  assert.match(html,/Graded correctness/);
  assert.match(html,/Graded required coverage/);
  assert.equal((html.match(/<dt>Major factual error rate<\/dt>/g)||[]).length,1);
  assert.match(html,/<dt>Major factual error rate<\/dt><dd>0%<\/dd>/);
  const noScores=context.window.MousebenchOverview.results({dataset:{frames:10},results:{scorecards:{models:[]}}});
  assert.match(noScores,/No eligible model scores yet/);
  assert.doesNotMatch(noScores,/<dt>Major factual error rate<\/dt>/);
});
