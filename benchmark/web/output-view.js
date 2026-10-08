'use strict';

// The server parses the unchanged response once, including duplicate-key checks.
// Rendering never repairs the response or treats valid JSON as factual accuracy.
function modelOutputStatus(record) {
  const view = record.output_view;
  if (!view) return {label: 'Format not checked', kind: 'pending'};
  if (view.contract_valid) return {label: Array.isArray(view.expected_fields) && view.expected_fields.length !== 6 ? `${view.expected_fields.length}-field JSON valid` : 'Six-field JSON valid', kind: 'valid'};
  return {label: view.json_valid ? 'Format issues' : 'JSON parse error', kind: 'invalid'};
}

function expertRatingValue(value) {
  if (value === true || value === 1 || value === '1') return {label:'True',kind:'true'};
  if (value === false || value === 0 || value === '0') return {label:'False',kind:'false'};
  if (value === null || value === undefined || value === '') return {label:'Not rated',kind:'missing'};
  return {label:String(value),kind:'unknown'};
}

function renderModelOutput(record, options = {}) {
  const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const list = value => Array.isArray(value) ? value : [];
  const readableValue = (value, depth = 0) => {
    if (depth > 10) return '<span class="output-notice">Further nested content is available in the raw response.</span>';
    if (Array.isArray(value)) return value.length ? `<ul class="output-value-list">${value.map(item => `<li>${readableValue(item, depth + 1)}</li>`).join('')}</ul>` : '<span class="output-notice">Empty array</span>';
    if (value !== null && typeof value === 'object') {
      const entries = Object.entries(value);
      return entries.length ? `<dl class="output-value-object">${entries.map(([key, item]) => `<div><dt>${escape(key)}</dt><dd>${readableValue(item, depth + 1)}</dd></div>`).join('')}</dl>` : '<span class="output-notice">Empty object</span>';
    }
    return `<span class="output-value-text">${value === null ? 'null' : value === '' ? '(empty text)' : escape(value)}</span>`;
  };
  const view = record.output_view;
  const raw = typeof record.text === 'string' ? record.text : '';
  const source = record.source_path || record.nas_path || record.path || '';
  let body;
  if (!view) {
    body = '<p class="output-notice">Format analysis is unavailable. Refresh after restarting the workbench; the original response is preserved below.</p>';
  } else {
    const checks = view.mechanical_checks || {};
    const problems = [];
    if (!view.json_valid) {
      const error = view.parse_error || {};
      const location = Number.isInteger(error.line) && Number.isInteger(error.column) ? ` (line ${error.line}, column ${error.column})` : '';
      problems.push(escape((error.message || 'The response is not valid JSON.') + location));
    } else if (!view.contract_valid) {
      for (const [key, label] of [['missing_fields','Missing fields'],['duplicate_fields','Repeated fields'],['empty_fields','Empty fields'],['type_errors','Fields that must be text'],['unexpected_fields','Unexpected fields']]) {
        if (list(checks[key]).length) problems.push(`${label}: ${escape(checks[key].join(', '))}.`);
      }
      problems.push(...list(checks.errors).map(escape));
    }
    const title = !view.json_valid ? 'JSON could not be parsed' : 'JSON parsed; expected format not met';
    body = problems.length ? `<div class="output-format-warning" role="note"><strong>${title}</strong><ul>${problems.map(p => `<li>${p}</li>`).join('')}</ul><p>The original response has not been repaired.</p></div>` : '';
    if (view.contract_valid) body += `<p class="output-notice">${Array.isArray(view.expected_fields) && view.expected_fields.length !== 6 ? `All ${view.expected_fields.length} requested text fields are present.` : 'All six text fields are present.'} Format validity does not establish factual correctness.</p>`;
    if (view.json_valid && !view.contract_valid) body += '<p class="output-notice">Readable fields are shown below. Missing, repeated or incorrectly typed fields remain flagged.</p>';
    if (!view.contract_valid) body += '<p class="output-rubric-note">When the frozen prompt requires this JSON format, these defects inform instruction-following assessment. Factual content is assessed separately. Historical outputs need their original requirements verified.</p>';
    body += list(view.fields).map(field => {
      const messages = {missing:'This field is missing.', empty:'This field contains no text.', wrong_type:'Expected text, but received a different JSON type.', duplicate:'This field occurs more than once. No occurrence was chosen; inspect the original response below.'};
      const status = field.status || 'missing';
      const text = status === 'ok' ? `<p class="output-section-text">${escape(field.value)}</p>`
        : `<p class="output-field-error">${escape(messages[status] || 'This field cannot be displayed reliably.')}</p>${status === 'wrong_type' ? readableValue(field.value) : ''}`;
      const sectionRating = options.sectionRatings ? expertRatingValue(options.sectionRatings[field.name]) : null;
      const ratingBadge = sectionRating ? `<span class="rating-flag ${sectionRating.kind}">Recorded rating: ${escape(sectionRating.label)}</span>` : '';
      return `<section class="output-section${status === 'ok' ? '' : ' has-error'}"><div class="output-section-heading"><h4>${escape(field.name)}</h4>${ratingBadge}</div>${text}</section>`;
    }).join('');
    if (list(view.extra_fields).length) body += `<section class="output-section has-error"><h4>Unexpected fields</h4>${view.extra_fields.map(field => `<p class="output-extra-name">${escape(field.name)}</p>${readableValue(field.value)}`).join('')}</section>`;
  }
  if (source) body += `<p class="source-path">Source: ${escape(source)}</p>`;
  if (record.sha256) body += `<p class="source-path">SHA-256 ${escape(record.sha256)}</p>`;
  body += `<details class="output-raw"><summary>Raw response · unchanged</summary>${raw ? `<pre>${escape(raw)}</pre>` : '<p class="output-notice">No response text was materialized.</p>'}</details>`;
  return `<div class="model-output-reader">${body}</div>`;
}

if (typeof module !== 'undefined' && module.exports) module.exports = {renderModelOutput, modelOutputStatus, expertRatingValue};
