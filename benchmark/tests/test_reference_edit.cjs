const test = require('node:test');
const assert = require('node:assert/strict');
const {capture, summaryFromSections} = require('../web/reference-edit.js');

const fields = ['Overall', 'Head', 'Limb', 'Torso', 'Others', 'Keywords'];
const sections = () => Object.fromEntries(fields.map(field => [field, `${field} detail.`]));
const fact = (id, field) => ({id, text:`Fact ${id}`, field, evidence:'image', required:true,
  accepted_alternatives:[`Alternative ${id}`], evidence_note:`Note ${id}`, uncertain:false});
function fixture() {
  const value = {sections:sections(), facts:[fact('limb', 'Limb'), fact('head', 'Head')],
    uncertainties:['First uncertainty', '', '  '], reference_description:'', annotation:{source:'fixture'}};
  value.reference_description = summaryFromSections(value.sections);
  return value;
}
const formRow = value => ({id:value.id, text:value.text, field:value.field, evidence:value.evidence, required:value.required});

test('grouped form facts retain saved order, extra metadata and unrelated content', () => {
  const previous = fixture(), before = JSON.stringify(previous);
  const rows = [formRow(previous.facts[1]), {...formRow(previous.facts[0]), text:'Edited limb fact'}];
  const result = capture(previous, {facts:rows});
  assert.deepEqual(result.facts.map(value => value.id), ['limb', 'head']);
  assert.equal(result.facts[0].text, 'Edited limb fact');
  assert.deepEqual(result.facts[0].accepted_alternatives, ['Alternative limb']);
  assert.equal(result.facts[0].evidence_note, 'Note limb');
  assert.equal(result.facts[0].uncertain, false);
  assert.deepEqual(result.annotation, previous.annotation);
  assert.equal(JSON.stringify(previous), before);
  result.facts[0].accepted_alternatives.push('Local edit');
  result.annotation.source = 'Changed';
  assert.equal(JSON.stringify(previous), before);
});

test('a no-op capture preserves JSON key order and values despite grouped DOM order', () => {
  const previous = fixture();
  const reverseSections = Object.fromEntries(Object.entries(previous.sections).reverse());
  const result = capture(previous, {sections:reverseSections,
    facts:previous.facts.slice().reverse().map(formRow), uncertainties:previous.uncertainties,
    reference_description:previous.reference_description});
  assert.equal(JSON.stringify(result), JSON.stringify(previous));
  assert.notEqual(result, previous);
  assert.notEqual(result.sections, previous.sections);
  assert.notEqual(result.uncertainties, previous.uncertainties);
});

test('deleted facts are omitted and new facts append in form order', () => {
  const previous = fixture(), first = fact('new-one', 'Overall'), second = fact('new-two', 'Head');
  const result = capture(previous, {facts:[first, formRow(previous.facts[0]), second]});
  assert.deepEqual(result.facts.map(value => value.id), ['limb', 'new-one', 'new-two']);
  assert.deepEqual(result.facts[1], first);
  assert.notEqual(result.facts[1], first);
  assert.deepEqual(capture(previous, {facts:[]}).facts, []);
});

test('omitted fields and blank editing values are preserved without normalization', () => {
  const previous = fixture();
  assert.equal(JSON.stringify(capture(previous)), JSON.stringify(previous));
  const result = capture(previous, {uncertainties:['', ' ', 'Exact\ntext']});
  assert.deepEqual(result.uncertainties, ['', ' ', 'Exact\ntext']);
  assert.deepEqual(result.sections, previous.sections);
  assert.deepEqual(result.facts, previous.facts);
  assert.equal(result.reference_description, previous.reference_description);
  const partial = capture(previous, {sections:{Head:'Revised head'}});
  assert.equal(partial.sections.Limb, previous.sections.Limb);
});

test('generated and blank summaries follow section edits in the fixed six-section order', () => {
  const previous = fixture(), edits = {sections:{Head:'Revised head'}};
  const result = capture(previous, edits);
  assert.equal(result.reference_description, summaryFromSections(result.sections));
  assert.match(result.reference_description, /Head: Revised head\n\nLimb:/);
  for (const blank of ['', '  ', null]) {
    const value = {...previous, reference_description:blank};
    assert.equal(capture(value, edits).reference_description, result.reference_description);
  }
  assert.equal(capture(previous, {...edits, reference_description:previous.reference_description}).reference_description,
    result.reference_description);
  assert.equal(summaryFromSections({Keywords:'Last', Overall:'First'}),
    'Overall: First\n\nHead: \n\nLimb: \n\nTorso: \n\nOthers: \n\nKeywords: Last');
});

test('custom summaries remain exact and explicitly edited summaries override auto-sync', () => {
  const previous = fixture(), custom = `DRAFT\n\n${previous.reference_description}`;
  const customPrevious = {...previous, reference_description:custom};
  const edits = {sections:{Head:'Revised head'}};
  assert.equal(capture(customPrevious, edits).reference_description, custom);
  assert.equal(capture(customPrevious, {...edits, reference_description:custom}).reference_description, custom);
  assert.equal(capture(previous, {...edits, reference_description:'Custom correction'}).reference_description, 'Custom correction');
  assert.equal(capture(previous, {...edits, reference_description:''}).reference_description, '');
  assert.equal(capture(customPrevious, {reference_description:'Explicit replacement'}).reference_description, 'Explicit replacement');
});


test('opening and applying an empty draft does not create a summary', () => {
  const previous = {reference_description:'', facts:[], uncertainties:[], sections:Object.fromEntries(fields.map(field=>[field,'']))};
  assert.equal(JSON.stringify(capture(previous,{sections:previous.sections,facts:[],uncertainties:[]})),JSON.stringify(previous));
});
