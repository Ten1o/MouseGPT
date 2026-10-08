(function () {
  'use strict';

  const FIELDS = ['Overall', 'Head', 'Limb', 'Torso', 'Others', 'Keywords'];
  const EVIDENCE = new Map([['image', 'Image'], ['velocity', 'Speed'], ['speed', 'Speed'], ['unknown', 'Unknown']]);
  const FACT_FIELDS = new Set(['id', 'text', 'field', 'required', 'evidence', 'evidence_note', 'evidence_scope', 'accepted_alternatives']);
  const CONTENT_FIELDS = new Set(['sections', 'facts', 'reference_description', 'uncertainties']);
  const isRecord = value => value !== null && typeof value === 'object' && !Array.isArray(value);
  const own = (value, key) => Object.prototype.hasOwnProperty.call(value, key);
  const escape = value => String(value ?? '').replace(/[&<>"']/g, character => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[character]));
  const prose = value => `<p class="ref-prose">${escape(value)}</p>`;
  const heading = (label, region) => `<div class="ref-section-heading"><h4>${escape(label)}</h4><button type="button" class="text-button ref-section-edit" data-edit-reference="${escape(region)}" aria-label="Edit ${escape(label)}">Edit</button></div>`;
  const empty = message => `<p class="ref-empty">${escape(message)}</p>`;

  // Keep the saved text exactly as supplied: equality here must not hide a custom summary.
  function summaryFromSections(sections) {
    const source = isRecord(sections) ? sections : {};
    return FIELDS.map(field => `${field}: ${own(source, field) ? String(source[field] ?? '') : ''}`).join('\n\n');
  }

  function readable(value) {
    if (Array.isArray(value)) return value.length ? `<ul>${value.map(item => `<li>${readable(item)}</li>`).join('')}</ul>` : empty('None recorded.');
    if (isRecord(value)) {
      const entries = Object.entries(value);
      return entries.length ? `<dl>${entries.map(([key, item]) => `<div><dt>${escape(key)}</dt><dd>${readable(item)}</dd></div>`).join('')}</dl>` : empty('No details recorded.');
    }
    return prose(value === null ? 'null' : value === undefined ? 'Not recorded' : value === '' ? '(empty text)' : value);
  }

  function rawAnnotation(value, label = 'Additional annotation fields') {
    return `<details class="ref-annotation"><summary>${escape(label)}</summary><pre>${escape(JSON.stringify(value, null, 2))}</pre></details>`;
  }

  function renderFact(value) {
    const fact = isRecord(value) ? value : {};
    const text = typeof value === 'string' ? value : fact.text;
    const required = fact.required === true;
    const evidence = EVIDENCE.get(fact.evidence) || 'Unknown';
    const field = typeof fact.field === 'string' ? fact.field : '';
    const unassigned = !FIELDS.includes(field);
    const annotations = [];
    for (const [key, label] of [['evidence_note', 'Evidence note'], ['evidence_scope', 'Evidence scope'], ['accepted_alternatives', 'Accepted alternatives'], ['id', 'Fact ID']]) {
      if (own(fact, key)) annotations.push(`<div><dt>${label}</dt><dd>${readable(fact[key])}</dd></div>`);
    }
    if (own(fact, 'evidence') && !EVIDENCE.has(fact.evidence)) {
      annotations.push(`<div><dt>Recorded evidence value</dt><dd>${readable(fact.evidence)}</dd></div>`);
    }
    if (own(fact, 'required') && typeof fact.required !== 'boolean') {
      annotations.push(`<div><dt>Recorded requirement value</dt><dd>${readable(fact.required)}</dd></div>`);
    }
    if (unassigned && own(fact, 'field') && typeof fact.field !== 'string') {
      annotations.push(`<div><dt>Recorded field</dt><dd>${readable(fact.field)}</dd></div>`);
    }
    const extras = Object.fromEntries(Object.entries(fact).filter(([key]) => !FACT_FIELDS.has(key)));
    const extraMarkup = Object.keys(extras).length ? rawAnnotation(extras) : '';
    const details = annotations.length ? `<details class="ref-annotation"><summary>Evidence &amp; annotation details</summary><dl>${annotations.join('')}</dl></details>` : '';
    const malformed = !isRecord(value) && typeof value !== 'string' ? rawAnnotation(value, 'Original fact annotation') : '';
    return `<li class="ref-fact"><div class="ref-fact-labels"><span class="ref-fact-status ${required ? 'required' : 'additional'}">${required ? 'Required' : 'Additional'}</span><span class="ref-evidence-label">${evidence}</span>${unassigned && field ? `<span class="ref-fact-field">Recorded field: ${escape(field)}</span>` : ''}</div>${typeof text === 'string' && text.length ? prose(text) : text != null && typeof text !== 'string' ? readable(text) : empty('No fact text recorded.')}${details}${extraMarkup}${malformed}</li>`;
  }

  function render(content) {
    const source = isRecord(content) ? content : {};
    const sections = isRecord(source.sections) ? source.sections : {};
    const facts = Array.isArray(source.facts) ? source.facts : [];
    const groups = new Map(FIELDS.map(field => [field, []]));
    const unassigned = [];
    for (const fact of facts) {
      const group = isRecord(fact) ? groups.get(fact.field) : null;
      (group || unassigned).push(fact);
    }
    const factList = list => `<ul class="ref-facts">${list.map(renderFact).join('')}</ul>`;
    let html = FIELDS.map(field => {
      const value = own(sections, field) ? sections[field] : null;
      const text = typeof value === 'string' && value.length ? prose(value) : value != null && typeof value !== 'string' ? readable(value) : empty('No section text recorded.');
      const matches = groups.get(field);
      return `<section class="ref-section" data-ref-section="${field}">${heading(field, field)}${text}${matches.length ? factList(matches) : empty('No facts recorded for this section.')}</section>`;
    }).join('');
    if (unassigned.length) html += `<section class="ref-section ref-unassigned" data-ref-section="unassigned">${heading('Unassigned facts', 'unassigned')}${factList(unassigned)}</section>`;

    const summary = source.reference_description;
    if (typeof summary === 'string' && summary.length) {
      html += summary === summaryFromSections(sections)
        ? '<div class="ref-summary"><p class="ref-summary-note">Reference summary matches the six sections.</p></div>'
        : `<section class="ref-summary">${heading('Additional reference summary', 'summary')}${prose(summary)}</section>`;
    } else if (summary != null && typeof summary !== 'string') {
      html += `<section class="ref-summary">${heading('Additional reference summary', 'summary')}${readable(summary)}</section>`;
    }

    const extras = Object.fromEntries(Object.entries(source).filter(([key]) => !CONTENT_FIELDS.has(key)));
    const extraSections = Object.fromEntries(Object.entries(sections).filter(([key]) => !FIELDS.includes(key)));
    if (Object.keys(extraSections).length) extras.sections = extraSections;
    for (const [key, valid] of [['sections', isRecord(source.sections)], ['facts', Array.isArray(source.facts)]]) {
      if (own(source, key) && !valid) extras[key] = source[key];
    }
    if (Object.keys(extras).length) html += rawAnnotation(extras);

    const uncertainties = Array.isArray(source.uncertainties) ? source.uncertainties : own(source, 'uncertainties') && source.uncertainties != null ? [source.uncertainties] : [];
    html += `<section class="ref-uncertainties">${heading('Uncertainties & alternatives', 'uncertainties')}${uncertainties.length ? `<ul>${uncertainties.map(value => `<li>${readable(value)}</li>`).join('')}</ul>` : empty('No uncertainties recorded.')}</section>`;
    return `<div class="ref-review">${html}</div>`;
  }

  const api = {render, summaryFromSections};
  if (typeof window !== 'undefined') window.MousebenchReferenceView = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
}());
