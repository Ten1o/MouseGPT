const test = require('node:test');
const assert = require('node:assert/strict');
const {renderModelOutput, modelOutputStatus} = require('../web/output-view.js');
const fields = ['Overall','Head','Limb','Torso','Others','Keywords'];
const record = () => ({text:'  {"Overall": "Exact raw text"}\n', source_path:'/fixture.json',
  output_view:{json_valid:true,contract_valid:true,parse_error:null,
    fields:fields.map(name => ({name,value:`Visible ${name} detail.`,status:'ok'})),extra_fields:[],mechanical_checks:{}}});

test('readable sections retain order and exact raw text in a closed final disclosure', () => {
  const value = record(), html = renderModelOutput(value);
  let position = -1;
  for (const field of fields) { const next=html.indexOf(`<h4>${field}</h4>`); assert.ok(next>position); position=next; }
  assert.ok(html.includes('<details class="output-raw"><summary>Raw response · unchanged</summary>'));
  assert.ok(html.includes('  {&quot;Overall&quot;: &quot;Exact raw text&quot;}\n'));
  assert.ok(html.indexOf('Source:') < html.indexOf('Raw response'));
  assert.deepEqual(modelOutputStatus(value),{label:'Six-field JSON valid',kind:'valid'});
});

test('malformed responses display the actual error location and do not fabricate sections', () => {
  const value=record();
  Object.assign(value.output_view,{json_valid:false,contract_valid:false,fields:[],parse_error:{message:'Expected a comma.',line:2,column:9}});
  const html=renderModelOutput(value);
  assert.ok(html.includes('Expected a comma. (line 2, column 9)'));
  assert.ok(!html.includes('<h4>Overall</h4>'));
  assert.ok(html.includes('Factual content is assessed separately'));
  assert.equal(modelOutputStatus(value).label,'JSON parse error');
});

test('duplicate, missing and wrong-type fields remain visible instead of being repaired', () => {
  const value=record(); value.output_view.contract_valid=false;
  value.output_view.fields[0]={name:'Overall',value:null,status:'duplicate'};
  value.output_view.fields[1]={name:'Head',value:null,status:'missing'};
  value.output_view.fields[2]={name:'Limb',value:['paw'],status:'wrong_type'};
  value.output_view.mechanical_checks={duplicate_fields:['Overall'],missing_fields:['Head'],type_errors:['Limb']};
  const html=renderModelOutput(value);
  assert.ok(html.includes('No occurrence was chosen'));
  assert.ok(html.includes('This field is missing'));
  assert.ok(html.includes('Expected text'));
  assert.ok(html.includes('<span class="output-value-text">paw</span>'));
  assert.equal(modelOutputStatus(value).label,'Format issues');
});

test('candidate content, extra keys, raw text, paths and errors cannot inject HTML', () => {
  const value=record(); const attack='<img src=x onerror="alert(1)">';
  value.text=attack;value.source_path=attack;value.sha256=attack;
  value.output_view.fields[0].value=attack;
  value.output_view.extra_fields=[{name:attack,value:attack}];
  const html=renderModelOutput(value);
  assert.ok(!html.includes('<img'));assert.ok(!html.includes('<script'));
  assert.ok(html.includes('&lt;img src=x onerror=&quot;alert(1)&quot;&gt;'));
});

test('non-object JSON is a schema error, not a syntax failure or a valid six-field response', () => {
  const value=record(); value.text='[]';
  Object.assign(value.output_view,{contract_valid:false,fields:[],mechanical_checks:{errors:['The JSON value must be an object.']}});
  const html=renderModelOutput(value);
  assert.ok(html.includes('JSON parsed; expected format not met'));
  assert.ok(html.includes('The JSON value must be an object.'));
  assert.ok(!html.includes('JSON could not be parsed'));
});

test('missing analysis remains unknown with raw response available', () => {
  const value=record();delete value.output_view;
  assert.equal(modelOutputStatus(value).label,'Format not checked');
  assert.ok(renderModelOutput(value).includes('Format analysis is unavailable'));
  assert.ok(renderModelOutput(value).includes('Raw response · unchanged'));
});

test('the historical two-field request is shown with its actual format requirement', () => {
  const value=record();
  value.output_view.expected_fields=['Description','Keywords'];
  value.output_view.fields=value.output_view.expected_fields.map(name=>({name,value:'Observed behavior',status:'ok'}));
  assert.equal(modelOutputStatus(value).label,'2-field JSON valid');
  const html=renderModelOutput(value);
  assert.match(html,/All 2 requested text fields are present/);
  assert.doesNotMatch(html,/All six text fields|<h4>Overall<\/h4>/);
});

test('nested JSON stays readable without hiding its field-type defect', () => {
  const value=record();value.output_view.contract_valid=false;
  value.output_view.fields[0]={name:'Overall',status:'wrong_type',value:{Activity:'Walking',Details:{visible:true,confidence:null},tags:['paw','tail']}};
  value.output_view.mechanical_checks={type_errors:['Overall']};
  const html=renderModelOutput(value);
  assert.ok(html.includes('Fields that must be text: Overall'));
  assert.ok(html.includes('<dt>Activity</dt><dd><span class="output-value-text">Walking</span>'));
  assert.ok(html.includes('<dt>visible</dt><dd><span class="output-value-text">true</span>'));
  assert.ok(html.includes('<dt>confidence</dt><dd><span class="output-value-text">null</span>'));
  assert.ok(html.includes('<ul class="output-value-list">'));
  assert.equal((html.match(/<pre>/g)||[]).length,1);
});
