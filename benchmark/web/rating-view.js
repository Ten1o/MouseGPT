'use strict';

// These references help a human compare archived evidence. They never create
// verified links, relabel an answer or infer sentence-level expert judgments.
function renderRatingEvidence(rating, reference, outputIndex) {
  const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const list = value => Array.isArray(value) ? value : [];
  const pairwise = rating.kind === 'pairwise';
  const fields = ['Overall','Head','Limb','Torso','Others','Keywords'];
  const helpers = typeof module !== 'undefined' && module.exports ? require('./output-view.js') : {renderModelOutput, modelOutputStatus, expertRatingValue};
  const valueBadge = value => { const flag = helpers.expertRatingValue(value); return `<span class="rating-flag ${flag.kind}">${escape(flag.label)}</span>`; };
  const preference = side => {
    if (rating.value === 'A is better' || rating.value === 'B is better') return rating.value[0] === side ? 'Expert preferred this answer' : 'Other answer preferred';
    if (rating.value === 'Both good') return 'Expert rated both good';
    if (rating.value === 'Both bad') return 'Expert rated both bad';
    return 'Preference not interpreted';
  };
  const heading = pairwise ? `Comparison · A: ${rating.model_a || 'Unknown'} / B: ${rating.model_b || 'Unknown'} · ${rating.value || 'Not recorded'}` : `Single model · ${rating.model_label || 'Unknown model'}`;
  const ratingTable = pairwise
    ? `<p class="rating-verdict">Recorded preference: <strong>${escape(rating.value || 'Not recorded')}</strong></p><p class="field-note">A: ${escape(rating.model_a || 'Unknown')}<br>B: ${escape(rating.model_b || 'Unknown')}</p><p class="rating-scope-note">This compares the whole answers; it does not label individual statements as true or false.</p>`
    : `<table class="rating-table"><thead><tr><th>Section</th><th>Expert label</th></tr></thead><tbody>${fields.map(name => `<tr><td>${name}</td><td>${valueBadge((rating.value || {})[name])}</td></tr>`).join('')}</tbody></table><p class="rating-scope-note">These are the expert’s section-level labels. “False” does not identify a particular incorrect sentence; “Not rated” is a missing label.</p>`;
  const fallbackSides = pairwise ? [{side:'A',model_label:rating.model_a},{side:'B',model_label:rating.model_b}] : [{side:'single',model_label:rating.model_label}];
  const sides = list(reference?.sides).length ? reference.sides : fallbackSides;
  const statusNames = {verified:'Human-verified output link',candidate:'Possible match · unverified',ambiguous:'Multiple versions · unverified',unavailable:'Output unavailable',conflict:'Conflicting source links'};
  const outputs = sides.map(side => {
    const status = Object.hasOwn(statusNames,side.status) ? side.status : 'unavailable';
    const records = list(side.output_ids).map(id => outputIndex.get(id)).filter(Boolean);
    const sideTitle = pairwise ? `${side.side} — ${side.model_label || 'Unknown model'}` : side.model_label || 'Unknown model';
    const warning = status === 'verified'
      ? 'A reviewer linked this exact saved answer to the source rating. This verifies correspondence, not factual truth.'
      : 'Shown for context. The exact answer rated in the worksheet has not been established; these labels must not be treated as verified ratings of this copy.';
    const renderAnswer = output => {
      const format = helpers.modelOutputStatus(output);
      const label = output.model_label || output.model || output.model_directory || 'Saved response';
      return `<p class="rating-output-label">Saved output: ${escape(label)} <span class="output-format-badge ${escape(format.kind)}">${escape(format.label)}</span></p>${helpers.renderModelOutput(output,pairwise ? {} : {sectionRatings:rating.value || {}})}`;
    };
    const content = records.length === 1 ? renderAnswer(records[0]) : records.length
      ? records.map((output,index) => `<details class="rating-output-alternative"><summary>Saved copy ${index + 1} · ${escape(output.model_label || output.model || 'Response')}</summary>${renderAnswer(output)}</details>`).join('')
      : '<p class="empty compact">No matching response is available for this side. The source rating is retained.</p>';
    const header = `<div class="rating-side-heading"><h4>${escape(sideTitle)}</h4>${pairwise ? `<span class="rating-preference">${escape(preference(side.side))}</span>` : ''}</div><span class="rating-link-status ${status}">${statusNames[status]}</span>`;
    const body = `<p class="rating-link-note">${escape(side.reason || 'Reference lookup is unavailable.')}</p>${records.length ? `<p class="rating-scope-note">${warning}</p>` : ''}${content}`;
    return pairwise ? `<details class="rating-reference-side"><summary>${header}</summary>${body}</details>` : `<section class="rating-reference-side">${header}${body}</section>`;
  }).join('');
  const warnings = list(reference?.warnings).length ? `<div class="output-format-warning"><strong>Source correspondence needs review</strong><ul>${reference.warnings.map(item => `<li>${escape(item)}</li>`).join('')}</ul></div>` : '';
  return `<details class="candidate rating-evidence" data-rating-id="${escape(rating.id)}"><summary>${escape(heading)}</summary><p class="field-note">${escape(rating.reviewer || 'Reviewer not recorded')} · ${escape(rating.source_file)} · ${escape(rating.sheet)} row ${escape(rating.row)}</p>${ratingTable}<details class="rating-output-references"><summary>${pairwise ? 'Referenced outputs · A and B' : 'Referenced model output'}</summary>${warnings}<div class="rating-reference-body${pairwise ? ' rating-comparison-grid' : ''}">${outputs}</div></details><p class="field-note">${escape(rating.matching_observation_copies || 1)} matching source copies</p><p class="source-path">Rating ID: ${escape(rating.id || 'Not recorded')}</p>${rating.id ? `<button type="button" class="copy-button" data-rating-copy="${escape(rating.id)}">Copy rating ID</button>` : ''}<details class="rating-row-provenance"><summary>Original row and provenance</summary><pre>${escape(JSON.stringify(rating,null,2))}</pre></details></details>`;
}

if (typeof module !== 'undefined' && module.exports) module.exports = {renderRatingEvidence};
