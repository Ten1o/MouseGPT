const test = require('node:test');
const assert = require('node:assert/strict');
const {renderRatingEvidence} = require('../web/rating-view.js');
const fields=['Overall','Head','Limb','Torso','Others','Keywords'];
const answer=(id,label='model-one')=>({id,model_label:label,text:'{"original":"unchanged"}',source_path:'/fixture/'+id,
  output_view:{json_valid:true,contract_valid:true,fields:fields.map(name=>({name,status:'ok',value:`${id}: ${name} content`})),extra_fields:[],mechanical_checks:{}}});
const rating=()=>({id:'rating-one',kind:'rubric',model_label:'model-one',reviewer:'expert',value:{Overall:true,Head:false,Limb:null,Torso:true,Others:false,Keywords:true},source_file:'fixture.xlsx',sheet:'source',row:2});
const refs=(status='candidate')=>({kind:'single',warnings:[],sides:[{side:'single',model_label:'model-one',status,reason:'Same frame and exact source label.',output_ids:['one']}]});
const index=()=>new Map([['one',answer('one')],['two',answer('two','model-two')]]);

test('single ratings have a closed output reference with section labels next to readable text',()=>{
  const html=renderRatingEvidence(rating(),refs(),index());
  assert.ok(html.includes('<details class="rating-output-references"><summary>Referenced model output'));
  assert.ok(html.includes('Recorded rating: True'));
  assert.ok(html.includes('Recorded rating: False'));
  assert.ok(html.includes('Recorded rating: Not rated'));
  assert.ok(html.includes('one: Head content'));
  assert.ok(html.includes('does not identify a particular incorrect sentence'));
  assert.ok(html.includes('Possible match · unverified'));
  assert.ok(html.includes('must not be treated as verified ratings of this copy'));
});

test('pairwise A/B order and preference remain explicit without fabricated factual labels',()=>{
  const row={...rating(),kind:'pairwise',model_a:'model-two',model_b:'model-one',value:'B is better'};
  const reference={sides:[{side:'A',model_label:'model-two',status:'candidate',output_ids:['two']},{side:'B',model_label:'model-one',status:'candidate',output_ids:['one']}]};
  const html=renderRatingEvidence(row,reference,index());
  assert.ok(html.indexOf('A — model-two')<html.indexOf('B — model-one'));
  assert.ok(html.indexOf('two: Overall content')<html.indexOf('one: Overall content'));
  assert.ok(html.includes('Other answer preferred'));
  assert.ok(html.includes('Expert preferred this answer'));
  assert.ok(!html.includes('Recorded rating:'));
  assert.ok(html.includes('does not label individual statements'));
  assert.equal((html.match(/<details class="rating-reference-side">/g)||[]).length,2);
});

test('both-good and both-bad source outcomes are retained without inventing a winner',()=>{
  for(const value of ['Both good','Both bad']) {
    const html=renderRatingEvidence({...rating(),kind:'pairwise',model_a:'A',model_b:'B',value},
      {sides:[{side:'A',status:'candidate',output_ids:['one']},{side:'B',status:'candidate',output_ids:['two']}]},index());
    assert.equal((html.match(new RegExp('Expert rated both '+value.split(' ')[1],'g'))||[]).length,2);
    assert.ok(!html.includes('Expert preferred this answer'));
  }
});

test('multiple versions stay folded individually and no answer is silently selected',()=>{
  const ref=refs('ambiguous');ref.sides[0].output_ids=['one','two'];
  const html=renderRatingEvidence(rating(),ref,index());
  assert.equal((html.match(/<details class="rating-output-alternative">/g)||[]).length,2);
  assert.ok(html.includes('one: Overall content'));assert.ok(html.includes('two: Overall content'));
  assert.ok(html.includes('Multiple versions · unverified'));
});

test('a missing comparison side retains its label and source preference',()=>{
  const row={...rating(),kind:'pairwise',model_a:'model-one',model_b:'missing-model',value:'Both bad'};
  const html=renderRatingEvidence(row,{sides:[{side:'A',model_label:'model-one',status:'candidate',output_ids:['one']},{side:'B',model_label:'missing-model',status:'unavailable',output_ids:[]}]},index());
  assert.ok(html.includes('B — missing-model'));
  assert.ok(html.includes('Output unavailable'));
  assert.ok(html.includes('No matching response is available for this side'));
  assert.ok(html.includes('Recorded preference: <strong>Both bad</strong>'));
});

test('a verified correspondence is distinct from scientific truth and conflicts remain visible',()=>{
  assert.ok(renderRatingEvidence(rating(),refs('verified'),index()).includes('verifies correspondence, not factual truth'));
  const ref=refs('conflict');ref.warnings=['Source model labels disagree.'];
  const html=renderRatingEvidence(rating(),ref,index());
  assert.ok(html.includes('Source model labels disagree.'));
  assert.ok(html.includes('Conflicting source links'));
  assert.ok(!html.includes('Human-verified output link'));
});

test('source labels, preferences, warnings and metadata are escaped',()=>{
  const attack='<img src=x onerror="alert(1)">';
  const row={...rating(),id:attack,model_label:attack,reviewer:attack};
  const ref=refs();ref.warnings=[attack];ref.sides[0].model_label=attack;ref.sides[0].reason=attack;
  const html=renderRatingEvidence(row,ref,index());
  assert.ok(!html.includes('<img'));assert.ok(html.includes('&lt;img'));
});

test('absent reference analysis remains unavailable instead of guessing model matches in the browser',()=>{
  const html=renderRatingEvidence(rating(),null,index());
  assert.ok(html.includes('Output unavailable'));assert.ok(!html.includes('one: Head content'));
  assert.ok(html.includes('Original row and provenance'));assert.ok(html.includes('Copy rating ID'));
});

test('invalid JSON retains its parse error and raw response inside the rating reference',()=>{
  const output=answer('one');output.text='{ broken';
  output.output_view={json_valid:false,contract_valid:false,fields:[],extra_fields:[],mechanical_checks:{},parse_error:{message:'Expected property name',line:1,column:3}};
  const html=renderRatingEvidence(rating(),refs(),new Map([['one',output]]));
  assert.ok(html.includes('Expected property name (line 1, column 3)'));
  assert.ok(html.includes('Raw response · unchanged'));
  assert.ok(html.includes('{ broken'));
});
