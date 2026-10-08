const test=require('node:test');
const assert=require('node:assert/strict');
const {renderDetail,renderSummary,renderListItem}=require('../web/scoring-review-view.js');

const sample=()=>({
  frame:{id:'240419_C11M4_00003972'},run:{id:'run-test',model:'Example model'},
  judgment:{schema_version:'mousebench.judgment.v2',id:'judgment-test',notes:'Both original images inspected. Limb contact is uncertain.'},
  prediction:{text:'{"Overall":"Walking","Limb":"Both paws raised"}',output_view:{json_valid:true,contract_valid:true,fields:[{name:'Overall',value:'Walking',status:'ok'},{name:'Limb',value:'Both paws raised',status:'ok'}],extra_fields:[],mechanical_checks:{}}},
  ground_truth:{revision:8,content:{sections:{Overall:'Walking low.',Limb:'One forepaw is raised.'},facts:[{id:'F01',field:'Limb',text:'One forepaw is raised.',required:true,accepted_alternatives:['A front paw is lifted.'],evidence_note:'Visible in the second simultaneous view.'},{id:'F02',field:'Torso',text:'The back is arched.',required:true}]}},
  input:{content:{system_prompt:'Both views show the same instant.',user_prompt:'Describe the mouse.'}},
  images:[{url:'/api/assets/image-one',label:'View 1'},{url:'/api/assets/image-two',label:'View 2'}],
  scores:{factual_precision:.75,required_fact_recall:.5,factual_f1:.6,structure_score:1,instruction_score:.75},benchmark_index:58.5,
  eligibility:{eligible:true,status:'scored',reasons:[]},
  claims:[{id:'C01',field:'Limb',text:'Both forepaws are raised.',correctness:.5,weight:1,evidence:'Only one forepaw is visibly lifted.',fact_ids:['F01'],error_type:'contradicted',severity:'major'},{id:'C02',field:'Overall',text:'The mouse is walking.',correctness:1,weight:2,evidence:'Supported by pose and speed.',fact_ids:[]}],
  fact_coverage:[{fact_id:'F01',field:'Limb',coverage:1,weight:1,evidence:'The supported paw is described.',claim_ids:['C01']},{fact_id:'F02',field:'Torso',coverage:0,weight:1,evidence:'No candidate claim describes the arch.',claim_ids:[]}],
  criteria:[{id:'uncertainty_handling',label:'Uncertainty handling',verdict:'unmet',evidence:'Contact is asserted too confidently.'},{id:'task_scope',label:'Task scope',verdict:'met',evidence:'Describes the mouse.'}],
  warnings:['Historical request correspondence remains unverified.'],bindings:{rubric_id:'rubric-test'}
});

test('summary shows six actual metrics without converting a missing value to zero',()=>{
  const html=renderSummary({factual_precision:0,required_fact_recall:null,factual_f1:0,structure_score:.75,instruction_score:1},0);
  assert.equal((html.match(/class="score-metric"/g)||[]).length,6);
  assert.ok(html.includes('<strong>0.0<small> / 100</small>'));
  assert.ok(html.includes('<strong>—<small></small>'));
  assert.ok(html.includes('Rubric grade: 0.75 / 1'));
  assert.ok(!html.includes('NaN'));
});

test('answer, original images, rationale and fact evidence are readable plain text',()=>{
  const html=renderDetail(sample());
  for(const text of ['Walking','Both paws raised','Only one forepaw is visibly lifted.','One forepaw is raised.','The back is arched.','Supported by pose and speed.','Contact is asserted too confidently.','Historical request correspondence remains unverified.']) assert.ok(html.includes(text),text);
  assert.ok(html.includes('0.5 / 1'));assert.ok(html.includes('Weight 2'));
  assert.ok(html.includes('src="/api/assets/image-one"'));
  assert.ok(html.includes('src="/api/assets/image-two"'));
  assert.ok(html.includes('simultaneous observations'));
  assert.ok(!html.includes('<textarea'));assert.ok(!html.includes('contenteditable'));
  assert.ok(html.includes('data-review-gt="240419_C11M4_00003972"'));
});

test('claims link to reference evidence and coverage links to actual candidate claim text',()=>{
  const html=renderDetail(sample());
  assert.ok(html.includes('id="score-claim-C01"'));
  assert.ok(html.includes('href="#score-claim-C01" data-score-claim="C01"'));
  assert.ok(html.includes('Ground-truth evidence'));
  assert.ok(html.includes('A front paw is lifted.'));
  assert.ok(html.includes('Visible in the second simultaneous view.'));
});

test('problem filter marks only full credit and keeps omitted facts visible',()=>{
  const html=renderDetail(sample());
  assert.ok(html.includes('score-point score-full-credit" id="score-claim-C02"'));
  assert.ok(html.includes('score-point score-partial-credit" id="score-claim-C01"'));
  assert.ok(html.includes('score-point score-partial-credit" id="score-fact-F02"'));
  assert.ok(html.includes('Omitted: no candidate claim covers this required fact.'));
  assert.ok(html.includes('score-criterion score-full-credit'));
  assert.ok(html.includes('not an average of this checklist'));
  assert.ok(html.includes('do not add deductions'));
});

test('saved stale score is conspicuously excluded, never presented as current aggregate evidence',()=>{
  const d=sample(); d.eligibility={eligible:false,status:'stale_ground_truth',reasons:[{code:'gt_mismatch',message:'Ground truth was revised after this judgment.'}]};
  const html=renderDetail(d);
  assert.ok(html.includes('score-status-note score-excluded'));
  assert.ok(html.includes('Excluded from current aggregate'));
  assert.ok(html.includes('not counted in the current model aggregate'));
  assert.ok(html.includes('Ground truth was revised after this judgment.'));
  assert.ok(html.includes('58.5'));
});

test('unjudged selection is inspectable without fabricated zero or eligibility claim',()=>{
  const d=sample();d.judgment=null;d.scores=null;d.benchmark_index=null;d.claims=[];d.fact_coverage=[];d.criteria=[];d.input=null;d.ground_truth=null;d.eligibility={eligible:false,status:'unjudged'};
  const html=renderDetail(d);
  assert.ok(html.includes('Not judged'));
  assert.ok(html.includes('Missing grades are not zero scores'));
  assert.ok(html.includes('Both paws raised'));
  assert.ok(!html.includes('Included in this score group'));
  assert.ok(!html.includes('>0.0<'));
  assert.ok(html.includes('No bound input is available'));
});

test('legacy raw judgments retain verdicts without inferring quarter grades',()=>{
  const d=sample();d.claims=[];d.fact_coverage=[];d.judgment={claims:[{text:'The mouse is crouched.',verdict:'supported',evidence:'Visible posture.'}],supported_fact_ids:['F01']};
  const html=renderDetail(d);
  assert.ok(html.includes('Graded claim details are unavailable'));
  assert.ok(html.includes('The mouse is crouched.'));
  assert.ok(html.includes('supported'));
  assert.ok(!html.includes('Correctness · 1 distinct claims'));
});

test('raw response, judgment and metadata are folded at the end',()=>{
  const html=renderDetail(sample());
  const rawAt=html.indexOf('<details class="score-raw">');
  assert.ok(rawAt>html.indexOf('Bound input prompt'));
  assert.ok(rawAt>html.indexOf('Quality checklist'));
  assert.ok(!html.includes('<details class="output-raw">'));
  assert.ok(html.endsWith('</details></section></div>'));
  assert.ok(html.includes('&quot;Both paws raised&quot;'));
});

test('malformed JSON surfaces parse error and preserves the unchanged response',()=>{
  const d=sample();d.prediction={text:'{broken',output_view:{json_valid:false,contract_valid:false,fields:[],parse_error:{message:'Expected property name',line:1,column:2},mechanical_checks:{}}};
  const html=renderDetail(d);
  assert.ok(html.includes('Expected property name (line 1, column 2)'));
  assert.ok(html.includes('{broken'));
  assert.ok(html.includes('Historical outputs need their original requirements verified'));
});

test('all source text, IDs and metadata are escaped and URI schemes are restricted',()=>{
  const attack='<img src=x onerror="alert(1)">';const d=sample();
  d.frame.id=attack;d.judgment.notes=attack;d.claims[0].text=attack;d.claims[0].id=attack;d.claims[0].evidence=attack;d.criteria[0].evidence=attack;d.warnings=[attack];d.ground_truth.content.facts[0].text=attack;d.images=[{url:'javascript:alert(1)',label:attack},{url:'//evil.example/image',label:attack}];d.prediction.text=attack;d.bindings={evil:attack};
  const html=renderDetail(d);
  assert.ok(!html.includes('<img'));assert.ok(html.includes('&lt;img'));
  assert.ok(!html.includes('src="javascript:'));assert.ok(!html.includes('src="//evil'));
});

test('list item preserves zero index, selection, safe IDs and absent scores',()=>{
  const html=renderListItem({frame_id:'<frame>',benchmark_index:0,eligibility:{eligible:true,status:'scored'}},true);
  assert.ok(html.includes('selected'));assert.ok(html.includes('aria-pressed="true"'));assert.ok(html.includes('0.0 / 100'));
  assert.ok(html.includes('data-score-frame="&lt;frame&gt;"'));assert.ok(!html.includes('<frame>'));
  const absent=renderListItem({frame_id:'F',benchmark_index:null,judged:false},false);
  assert.ok(absent.includes('Not judged'));assert.ok(absent.includes('<strong>—</strong>'));
});

test('normalized graded scorer details work when top-level convenience arrays are absent',()=>{
  const d=sample();d.scores.graded_details={claims:d.claims,fact_coverage:d.fact_coverage};delete d.claims;delete d.fact_coverage;
  const html=renderDetail(d);
  assert.ok(html.includes('Correctness · 2 distinct claims'));
  assert.ok(html.includes('Required-fact coverage · 2 facts'));
  assert.ok(html.includes('One forepaw is raised.'));
});

test('empty detail gracefully renders unavailable data and never crashes',()=>{
  const html=renderDetail({});
  assert.ok(html.includes('No saved model answer'));
  assert.ok(html.includes('Original input images are unavailable'));
  assert.ok(!html.includes('undefined'));
});

test('arithmetic uses saved weighted totals and an explicit policy without hardcoding weights',()=>{
  const d=sample();d.scores.denominators={weighted_correctness_credit:3,claim_weight:4,weighted_coverage_credit:2,required_fact_weight:4};d.index_details={formula:'Explicit policy from bound rubric',multiplier:.975};
  const html=renderDetail(d);
  assert.ok(html.includes('How this score is calculated'));
  assert.ok(html.includes('3 weighted credit / 4 total weight = 75.0 / 100'));
  assert.ok(html.includes('2 weighted credit / 4 total weight = 50.0 / 100'));
  assert.ok(html.includes('Explicit policy from bound rubric'));
  assert.ok(!html.includes('0.8 +'));
  assert.ok(html.indexOf('How this score is calculated')<html.indexOf('Judge’s rationale'));
});

test('parsed output cannot inject markup or terminate its raw foldout',()=>{
  const d=sample();const attack='</details><script>alert(1)</script><img src=x onerror=evil>';
  d.prediction.text=attack;d.prediction.output_view.fields[0].value=attack;d.images=[];
  const html=renderDetail(d);
  assert.ok(!html.includes('<script>'));assert.ok(!html.includes('<img'));
  assert.ok(html.includes('&lt;/details&gt;&lt;script&gt;'));
  assert.ok(html.endsWith('</details></section></div>'));
});

test('images are limited to generated local API URLs',()=>{
  const d=sample();d.images=[{url:'https://external.example/image'},{url:'/other/image'},{url:'/api/assets/valid'}];
  const html=renderDetail(d);
  assert.ok(!html.includes('external.example'));assert.ok(!html.includes('/other/image'));
  assert.ok(html.includes('src="/api/assets/valid"'));
});

test('unavailable or checksum-failed image has an explicit error and no broken image link',()=>{
  const d=sample();d.images=[{url:'/api/assets/missing',available:false,error:'Bound image checksum does not match.'}];
  const html=renderDetail(d);
  assert.ok(html.includes('View 1 · Image unavailable'));
  assert.ok(html.includes('Bound image checksum does not match.'));
  assert.ok(!html.includes('<img'));assert.ok(!html.includes('href="/api/assets/missing"'));
});

test('legacy backend claim and synthetic binary coverage arrays are not called graded claims',()=>{
  const d=sample();d.judgment={schema_version:'mousebench.judgment.v1',claims:[{id:'C1',text:'Legacy claim',verdict:'supported',evidence:'Original rationale'}],supported_fact_ids:['F01']};
  d.claims=d.judgment.claims;d.fact_coverage=[{fact_id:'F01',coverage:1,evidence:'Legacy binary coverage',claim_ids:[]}];
  const html=renderDetail(d);
  assert.ok(html.includes('Legacy claim'));assert.ok(html.includes('supported'));
  assert.ok(!html.includes('Correctness ·'));assert.ok(!html.includes('five-level'));
  assert.ok(html.includes('without inventing partial-credit grades'));
});

test('criterion definition is folded but evidence remains visible and not_assessable is explicit',()=>{
  const d=sample();d.criteria=[{id:'cross_view',label:'Cross-view consistency',definition:'The rule that defines this criterion.',verdict:'not_assessable',evidence:'A view is unavailable.'}];
  const html=renderDetail(d);
  assert.ok(html.includes('<details class="score-criterion-rule">'));
  assert.ok(html.includes('The rule that defines this criterion.'));
  assert.ok(html.includes('Not assessable'));assert.ok(html.includes('A view is unavailable.'));
});

test('fenced JSON presentation does not suppress original parse error or change raw answer',()=>{
  const d=sample();const raw='```json\n{"Overall":"Mouse is crouched"}\n```';
  d.prediction={text:raw,output_view:{json_valid:false,contract_valid:false,fields:[],parse_error:{message:'Unexpected code fence',line:1,column:1},mechanical_checks:{}},presentation_view:{json_valid:true,contract_valid:true,fields:[{name:'Overall',value:'Mouse is crouched',status:'ok'}],extra_fields:[],mechanical_checks:{}}};
  const html=renderDetail(d), end=html.indexOf('<section class="score-assessment"');
  assert.ok(html.includes('Unexpected code fence'));
  assert.ok(html.slice(0,end).includes('Mouse is crouched'));
  assert.ok(html.includes('Readable content inside code fence; original format check unchanged.'));
  assert.ok(html.includes('```json'));
});

test('arbitrary malformed JSON or plain prose remains visible on main answer pane',()=>{
  const d=sample();d.prediction={text:'The mouse is crouched. <script>bad</script>',output_view:{json_valid:false,contract_valid:false,fields:[],parse_error:{message:'Expected JSON'},mechanical_checks:{}}};
  const html=renderDetail(d), end=html.indexOf('<section class="score-assessment"');
  assert.ok(html.slice(0,end).includes('The mouse is crouched. &lt;script&gt;bad&lt;/script&gt;'));
  assert.ok(html.includes('score-unparsed-answer'));
  assert.ok(!html.includes('<script>bad</script>'));
});
