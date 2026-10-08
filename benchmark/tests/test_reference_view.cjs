const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const {render, summaryFromSections} = require('../web/reference-view.js');
const fields = ['Overall', 'Head', 'Limb', 'Torso', 'Others', 'Keywords'];
const escape = value => String(value).replace(/[&<>"']/g, character => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[character]));
const occurrences = (html, text) => html.split(text).length - 1;
const visibleMarkup = html => html.replace(/<details\b[^>]*>[\s\S]*?<\/details>/g, '');
const section = (html, field) => html.match(new RegExp(`<section class="ref-section[^\"]*" data-ref-section="${field}">([\\s\\S]*?)<\\/section>`))?.[1];
const fixture = () => {
  const sections = Object.fromEntries(fields.map(field => [field, `Section ${field} exact text.\nAnother line for ${field}.`]));
  const assignments = ['Overall', 'Head', 'Head', 'Head', 'Head', 'Limb', 'Limb', 'Limb', 'Torso', 'Torso', 'Limb', 'Overall', 'Others'];
  return {sections, reference_description: summaryFromSections(sections), uncertainties: ['Some paws are obscured.', 'Gaze target remains uncertain.'],
    facts: assignments.map((field, index) => ({id: `F${index + 1}`, field, text: `Atomic claim number ${index + 1}.`, required: index % 2 === 0, evidence: index === 11 ? 'velocity' : 'image', evidence_note: `Evidence for claim ${index + 1}.`, accepted_alternatives: [`Alternative for claim ${index + 1}.`]}))};
};

test('all six section texts and 13 facts appear once in order, with every fact visible in its section', () => {
  const content = fixture(), html = render(content), visible = visibleMarkup(html);
  let previous = -1;
  for (const field of fields) {
    const start = html.indexOf(`data-ref-section="${field}"`);
    assert.ok(start > previous);
    previous = start;
    assert.equal(occurrences(html, escape(content.sections[field])), 1);
    assert.ok(section(html, field).includes(`<p class="ref-prose">${content.sections[field]}</p>`));
  }
  assert.equal(occurrences(html, 'class="ref-fact"'), 13);
  for (const fact of content.facts) {
    assert.equal(occurrences(html, fact.text), 1);
    assert.ok(visible.includes(fact.text));
    assert.ok(section(html, fact.field).includes(fact.text));
    for (const other of fields.filter(field => field !== fact.field)) assert.ok(!section(html, other).includes(fact.text));
  }
  assert.equal(occurrences(visible, '>Required</span>'), 7);
  assert.equal(occurrences(visible, '>Additional</span>'), 6);
  assert.equal(occurrences(visible, '>Image</span>'), 12);
  assert.equal(occurrences(visible, '>Speed</span>'), 1);
  assert.ok(!/<(?:textarea|input|select|form)\b|contenteditable|ondblclick|onclick/.test(html));
});

test('equivalent summaries are a compact note; custom or whitespace-different summaries remain visible in full', () => {
  const content = fixture();
  assert.equal(summaryFromSections(content.sections), fields.map(field => `${field}: ${content.sections[field]}`).join('\n\n'));
  assert.ok(render(content).includes('Reference summary matches the six sections.'));
  assert.ok(!render(content).includes('Additional reference summary'));
  for (const summary of ['Custom reference\nwith an extra observation.', `${content.reference_description}\n`]) {
    content.reference_description = summary;
    const html = render(content);
    assert.ok(!html.includes('Reference summary matches the six sections.'));
    assert.ok(visibleMarkup(html).includes(`<h4>Additional reference summary</h4>`));
    assert.ok(visibleMarkup(html).includes(`<p class="ref-prose">${escape(summary)}</p>`));
  }
});

test('uncertainties are a visible final list; evidence notes, scopes and alternatives are readable disclosures', () => {
  const content = fixture();
  content.facts[0].evidence_scope = {views: ['first', 'second'], limitation: 'Still images only.'};
  const html = render(content), visible = visibleMarkup(html);
  for (const uncertainty of content.uncertainties) assert.ok(visible.includes(uncertainty));
  assert.ok(html.indexOf('class="ref-uncertainties"') > html.indexOf('class="ref-summary"'));
  assert.ok(html.includes('<dt>Evidence note</dt>'));
  assert.ok(html.includes('<dt>Evidence scope</dt>'));
  assert.ok(html.includes('<dt>Accepted alternatives</dt>'));
  assert.ok(html.includes('<dt>views</dt>'));
  assert.ok(html.includes('Still images only.'));
  assert.ok(!visible.includes('Evidence for claim 1.'));
  assert.ok(!visible.includes('Alternative for claim 1.'));
  assert.ok(!html.includes('<pre>'), 'Common annotation fields should not become raw JSON');
  assert.ok(!/<details\b[^>]*\bopen\b/.test(html));
});

test('untrusted section text, facts, summaries, uncertainties and annotation metadata are escaped', () => {
  const attack = '<img src=x onerror="alert(1)">&\' <script>alert(2)</script>';
  const content = fixture();
  content.sections.Overall = attack;
  content.sections.Keywords = `tag one, ${attack}; plain\nkeywords`;
  content.reference_description = attack;
  content.uncertainties = [attack];
  content.facts = [{id: attack, field: attack, text: attack, required: false, evidence: attack, evidence_note: attack, evidence_scope: {[attack]: attack}, accepted_alternatives: [attack], [attack]: attack}];
  content[attack] = attack;
  const html = render(content);
  assert.ok(!html.includes('<img'));
  assert.ok(!html.includes('<script'));
  assert.ok(html.includes(escape(attack)));
  assert.ok(visibleMarkup(html).includes(`tag one, ${escape(attack)}; plain\nkeywords`));
  assert.ok(section(html, 'unassigned').includes(`Recorded field: ${escape(attack)}`));
  assert.ok(html.includes('>Unknown</span>'));
  for (const evidence of ['constructor', '__proto__', 'toString']) {
    assert.ok(render({facts: [{text: 'Unknown evidence value.', evidence}]}).includes('>Unknown</span>'));
  }
});

test('missing content has six helpful empty sections and unassigned facts are never dropped', () => {
  for (const content of [undefined, null, {}]) {
    const html = render(content);
    assert.equal(occurrences(html, 'class="ref-section"'), 6);
    assert.equal(occurrences(html, 'No section text recorded.'), 6);
    assert.ok(html.includes('No uncertainties recorded.'));
  }
  const html = render({facts: [{text: 'No field provided.', required: true}, {field: 'Unexpected', text: 'Unknown field retained.'}, 'Legacy plain fact.']});
  const group = section(html, 'unassigned');
  assert.ok(group.includes('No field provided.'));
  assert.ok(group.includes('Unknown field retained.'));
  assert.ok(group.includes('Legacy plain fact.'));
  assert.ok(group.includes('Recorded field: Unexpected'));
  assert.equal(occurrences(group, '>Unknown</span>'), 3);
  assert.ok(!visibleMarkup(html).includes('No fact text recorded.'));
});

test('unknown annotations and malformed common values remain inspectable without mutating input', () => {
  const content = fixture();
  content.facts[0].confidence = {source: 'reviewer', level: 0.7};
  content.sections.Extra = 'Unrecognized section text.';
  content.custom_annotation = 'Saved supplementary metadata.';
  content.facts.push({text: 'Unrecognized evidence is not inferred.', field: 'Keywords', evidence: 'sound', required: 'yes'});
  const before = structuredClone(content);
  const html = render(content);
  assert.deepEqual(content, before);
  assert.equal(render(content), html);
  for (const value of ['confidence', 'reviewer', '0.7', 'Unrecognized section text.', 'Saved supplementary metadata.', 'sound', 'yes']) assert.ok(html.includes(value));
  assert.ok(visibleMarkup(html).includes('Unrecognized evidence is not inferred.'));
  assert.ok(!visibleMarkup(html).includes('Saved supplementary metadata.'));
  assert.ok(render({facts: {legacy: 'Unexpected facts container.'}, sections: 'Unexpected sections container.'}).includes('Unexpected facts container.'));
});

test('browser global and CommonJS exports expose the same two pure functions', () => {
  const source = fs.readFileSync(require.resolve('../web/reference-view.js'), 'utf8');
  const context = {window: {}};
  vm.runInNewContext(source, context);
  assert.deepEqual(Object.keys(context.window.MousebenchReferenceView).sort(), ['render', 'summaryFromSections']);
  assert.equal(context.window.MousebenchReferenceView.render(fixture()), render(fixture()));
  assert.equal(context.window.MousebenchReferenceView.summaryFromSections(fixture().sections), summaryFromSections(fixture().sections));
});


test('each editable review region has a labeled non-submit edit button', () => {
  const content = fixture();
  content.reference_description = 'Custom summary';
  content.facts.push({id:'unassigned',text:'An unassigned fact'});
  const html = render(content);
  for (const region of [...fields,'summary','uncertainties','unassigned']) {
    assert.equal(occurrences(html, `data-edit-reference="${region}"`), 1);
    assert.match(html,new RegExp(`<button type="button"[^>]*data-edit-reference="${region}"[^>]*aria-label="Edit [^"]+">Edit</button>`));
  }
  assert.ok(!/<(?:textarea|input|select)\b|contenteditable/.test(html));
});
