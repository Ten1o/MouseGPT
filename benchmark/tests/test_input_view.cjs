const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../web/app.js'), 'utf8');

function harness(contract, recorded={status:'unavailable',error:'No original log'}) {
  const nodes = new Map();
  const node = selector => {
    if (!nodes.has(selector)) nodes.set(selector,{value:'',textContent:'',innerHTML:'',disabled:false,readOnly:false,addEventListener(){}});
    return nodes.get(selector);
  };
  const context = vm.createContext({document:{querySelector:node,querySelectorAll:()=>[]},window:{},structuredClone});
  vm.runInContext(source.replace(/\ninit\(\);\s*$/, '') + '\nglobalThis.controls={state,renderContract,recordedImages,renderFrame,renderEvidence};',context);
  context.controls.state.detail={frame:{id:'frame-1'},input_contract:contract,recorded_input:recorded};
  context.controls.renderContract();
  return {nodes,node,controls:context.controls,context};
}

function fixture(protocol) {
  return {protocol_version:protocol,revision:1,prompt:'Exact template with derived speeds.',system_prompt:'System text',
    image_detail:'high',input_hash:'frozen-hash',track:'image_velocity',reviewer:'Provenance import',
    input_images:[1,2].map(n=>({attachment_id:`input_image_${n}`,image_detail:'high'})),
    velocity:{units:'cm/s',value:{overall:1.23,right_forelimb:2.34,left_forelimb:3.45}}};
}

test('reconstructed input remains usable and is not presented as a recovered source log',()=>{
  const c={...fixture('mousebench.input.v3.reconstructed'),reconstruction:{manifest_id:'manifest-1',method_id:'method-1',limitations:['<untrusted>'],quality_flags:['Review <speed>']}};
  const h=harness(c);
  assert.match(h.node('#recorded-input-status').innerHTML,/Reconstructed input saved/);
  assert.match(h.node('#recorded-input-status').innerHTML,/Review &lt;speed&gt;/);
  assert.doesNotMatch(h.node('#recorded-input-status').innerHTML,/Original request imported|Source recheck|Recorded input unavailable/);
  assert.equal((h.node('#recorded-input-images').innerHTML.match(/<figure>/g)||[]).length,2);
  assert.match(h.node('#recorded-input-images').innerHTML,/Reconstructed input view 1/);
  assert.match(h.node('#recorded-input-speeds').innerHTML,/Derived speed estimates/);
  assert.match(h.node('#recorded-input-source').innerHTML,/Open reconstruction manifest/);
  assert.match(h.node('#recorded-input-source').innerHTML,/&lt;untrusted&gt;/);
  assert.doesNotMatch(h.node('#recorded-input-source').innerHTML,/Open original request log/);
  assert.equal(h.node('#save-contract').disabled,true);
  assert.equal(h.node('#contract-notes').readOnly,true);
});

test('logged input retains its original-log provenance and reimport controls',()=>{
  const c={...fixture('mousebench.input.v3.logged'),source_request:{record_id:'log-1',source_path:'/nas/original.json',sha256:'source-hash',generation_settings:{temperature:0}}};
  const h=harness(c,{status:'ready'});
  assert.match(h.node('#recorded-input-status').innerHTML,/Original request imported/);
  assert.match(h.node('#recorded-input-source').innerHTML,/Open original request log/);
  assert.match(h.node('#recorded-input-speeds').innerHTML,/Recorded speed estimates/);
  assert.equal(h.node('#save-contract').disabled,false);
  assert.equal(h.node('#contract-notes').readOnly,false);
});

test('missing input still blocks import and has no invented paired images',()=>{
  const h=harness(null);
  assert.match(h.node('#recorded-input-status').innerHTML,/Recorded input unavailable/);
  assert.equal(h.node('#recorded-input-images').innerHTML,'');
  assert.equal(h.node('#save-contract').disabled,true);
});

test('standardized logged input shows current prompt while preserving original-log image and speed provenance',()=>{
  const original={...fixture('mousebench.input.v3.logged'),source_request:{record_id:'log-1',source_path:'/nas/original.json',sha256:'source-hash',generation_settings:{temperature:0}}};
  const c={...fixture('mousebench.input.v3.adapted'),system_prompt:'Exact v1.0 six-field instruction',adaptation:{parent_input:{...original,output_fields:['Description','Keywords']},system_prompt_template:{frame_id:'v1.0-template'},changed_fields:['system_prompt']}};
  const h=harness(c,{status:'ready',system_prompt:'Historical two-field instruction'});
  assert.equal(h.node('#contract-system-prompt').value,'Exact v1.0 six-field instruction');
  assert.match(h.node('#recorded-input-status').innerHTML,/Standardized input saved/);
  assert.doesNotMatch(h.node('#recorded-input-status').innerHTML,/Original request imported|Recorded input unavailable/);
  assert.match(h.node('#contract-system-note').textContent,/Tail descriptions belong in Limb/);
  assert.match(h.node('#contract-system-note').textContent,/unchanged from the prior input version/);
  assert.match(h.node('#recorded-input-images').innerHTML,/Recorded input view 1/);
  assert.match(h.node('#recorded-input-speeds').innerHTML,/Recorded speed estimates/);
  assert.match(h.node('#recorded-input-source').innerHTML,/Prompt standardization provenance/);
  assert.match(h.node('#recorded-input-source').innerHTML,/Open original request log/);
  assert.equal(h.node('#save-contract').disabled,true);
  assert.equal(h.node('#save-contract').textContent,'Standardized input saved');
  assert.equal(h.node('#contract-notes').readOnly,true);
});

test('standardized reconstructed input retains derivation, limitations, flags and paired images',()=>{
  const original={...fixture('mousebench.input.v3.reconstructed'),reconstruction:{manifest_id:'manifest-1',limitations:['<untrusted>'],quality_flags:['Review <speed>']}};
  const c={...fixture('mousebench.input.v3.adapted'),adaptation:{parent_input:original,changed_fields:['system_prompt']}};
  const h=harness(c);
  assert.match(h.node('#recorded-input-status').innerHTML,/Standardized input saved/);
  assert.match(h.node('#recorded-input-status').innerHTML,/Review &lt;speed&gt;/);
  assert.doesNotMatch(h.node('#recorded-input-status').innerHTML,/Original request imported|Recorded input unavailable/);
  assert.match(h.node('#recorded-input-images').innerHTML,/Reconstructed input view 1/);
  assert.equal((h.node('#recorded-input-images').innerHTML.match(/<figure>/g)||[]).length,2);
  assert.match(h.node('#recorded-input-speeds').innerHTML,/Derived speed estimates/);
  assert.match(h.node('#recorded-input-source').innerHTML,/Open reconstruction manifest/);
  assert.match(h.node('#recorded-input-source').innerHTML,/&lt;untrusted&gt;/);
  assert.doesNotMatch(h.node('#recorded-input-source').innerHTML,/Open original request log/);
  assert.equal(h.node('#save-contract').disabled,true);
});

test('repeated prompt adaptations retain the original image reconstruction origin',()=>{
  const original={...fixture('mousebench.input.v3.reconstructed'),reconstruction:{manifest_id:'manifest-1'}};
  const prior={...fixture('mousebench.input.v3.adapted'),adaptation:{parent_input:original}};
  const h=harness({...fixture('mousebench.input.v3.adapted'),adaptation:{parent_input:prior}});
  assert.match(h.node('#recorded-input-images').innerHTML,/Reconstructed input view 1/);
  assert.match(h.node('#recorded-input-source').innerHTML,/Open reconstruction manifest/);
});


test('ground-truth evidence panel uses standardized prompt language and still shows reconstructed pair',()=>{
  const c={...fixture('mousebench.input.v3.adapted'),adaptation:{parent_input:{protocol_version:'mousebench.input.v3.reconstructed',reconstruction:{manifest_id:'manifest-1'}}}};
  const h=harness(c);
  vm.runInContext('renderEvidence=()=>{};renderEditor=()=>{};renderGate=()=>{};loadFrameReviewIssues=()=>{};',h.context);
  h.controls.renderFrame();
  const html=h.node('#frame-workspace').innerHTML;
  assert.match(html,/Reconstructed input images/);
  assert.match(html,/Reconstructed input view 1/);
  assert.match(html,/Source-reconstructed images and speed text are unchanged/);
  assert.match(html,/Standardized prompt &amp; speed information|Standardized prompt & speed information/);
  assert.match(html,/Current system instruction/);
  assert.match(html,/six output fields, with the tail in Limb/);
  assert.doesNotMatch(html,/Archived frame preview|Original system instruction/);
});

test('archived output panel explains old request requirements independently of new standardized input',()=>{
  const h=harness({...fixture('mousebench.input.v3.adapted'),adaptation:{parent_input:{protocol_version:'mousebench.input.v3.logged'}}});
  h.controls.state.evidenceTab='descriptions';
  h.controls.state.detail.descriptions=[];
  h.controls.renderEvidence();
  const html=h.node('#evidence-content').innerHTML;
  assert.match(html,/Historical output format/);
  assert.match(html,/archived answers retain their original request requirements/);
  assert.match(html,/New model runs use Overall, Head, Limb, Torso, Others and Keywords/);
});
