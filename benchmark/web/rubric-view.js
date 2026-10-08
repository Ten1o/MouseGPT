'use strict';

// Read-only, source-driven presentation. Saving/freezing remains in workflow.js.
(function(root) {
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const list = value => Array.isArray(value) ? value : [];
  const object = value => value && typeof value === 'object' && !Array.isArray(value) ? value : {};
  const label = key => String(key).replaceAll('_',' ').replace(/^./,c=>c.toUpperCase());
  const present = value => value !== undefined && value !== null;
  function readable(value) {
    if (Array.isArray(value)) return `<ul>${value.map(v=>`<li>${readable(v)}</li>`).join('')}</ul>`;
    if (value && typeof value === 'object') return `<dl class="rubric-details-list">${Object.entries(value).map(([key,v])=>`<div><dt>${esc(label(key))}</dt><dd>${readable(v)}</dd></div>`).join('')}</dl>`;
    return esc(value === true ? 'Yes' : value === false ? 'No' : value ?? 'Not specified');
  }
  function table(headers,rows,caption='') {
    return `<div class="table-wrap"><table class="rubric-table">${caption?`<caption>${esc(caption)}</caption>`:''}<thead><tr>${headers.map(v=>`<th scope="col">${esc(v)}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map(cell=>`<td>${cell}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  }
  function anchors(title, values) {
    const rows=Object.entries(object(values)).sort(([a],[b])=>Number(b)-Number(a));
    return rows.length ? `<section class="rubric-anchor"><h4>${esc(title)}</h4>${table(['Score','Evidence-based meaning'],rows.map(([score,text])=>[esc(score),readable(text)]))}</section>` : '';
  }
  function scoreLabels(record={}) {
    const version=record.scoring_version || record.scoring?.version || '';
    const graded=version==='mousebench.graded-factual-f1.v2';
    return {graded,version,f1:graded?'Graded factual F1':'Grounded factual F1',precision:graded?'Graded correctness':'Fact precision',recall:graded?'Graded required coverage':'Required recall',versionLabel:graded?'Graded scoring v2':version?`Legacy scoring · ${version}`:'Legacy factual scoring'};
  }
  function renderRubric(content,meta={}) {
    const r=object(content), scoring=object(r.scoring), fields=list(r.fields), weights=object(scoring.field_weights);
    const statuses=[meta.unsaved?'Unsaved draft':meta.status||'Proposed draft',r.status].filter(Boolean);
    let html=`<div class="rubric-reader"><div class="rubric-reader-heading"><h4>${esc(r.id||'Rubric draft')}</h4><span class="pill ${meta.status==='frozen'&&!meta.unsaved?'approved':'draft'}">${esc(statuses.join(' · '))}</span></div><p class="rubric-version">${esc(meta.id?`Saved version: ${meta.id} · `:'')}${esc(scoring.version||'Legacy factual scoring')} · ${esc(r.schema_version||'Schema not specified')}</p><p class="rubric-provisional">Provisional assessment standard. Freezing a rubric preserves its rules; it does not establish judge reliability. The examples below are illustrations, not model results.</p>${r.task?`<p class="rubric-task">${esc(r.task)}</p>`:''}`;
    if (r.correctness_score_anchors || r.coverage_score_anchors) {
      html+=`<h4 class="rubric-section-title">Factual quality · five-level scoring</h4><div class="rubric-anchor-grid">${anchors('Correctness of each distinct claim',r.correctness_score_anchors)}${anchors('Coverage of each required fact',r.coverage_score_anchors)}</div>`;
    } else if(r.claim_verdicts) {
      html+=`<h4 class="rubric-section-title">Claim verdicts · legacy scoring</h4>${table(['Verdict','Meaning'],Object.entries(object(r.claim_verdicts)).map(([key,text])=>[esc(label(key)),readable(text)]))}`;
    }
    html+=`<h4 class="rubric-section-title">Sections and factual weights</h4>${table(['Section','Scope',...(Object.keys(weights).length?['Weight']:[])],fields.map(field=>[esc(field),readable(r.section_scope?.[field]||'Not specified'),...(Object.keys(weights).length?[esc(present(weights[field])?weights[field]:'Not specified')]:[])]))}`;
    if(present(scoring.unknown_gt_weight))html+=`<p class="field-note">Fallback weight for a required GT fact without a recognized section: ${esc(scoring.unknown_gt_weight)}.</p>`;
    html+=`<h4 class="rubric-section-title">How the score is calculated</h4><p class="rubric-primary">Primary metric: <strong>${esc(scoring.primary||'Not specified')}</strong></p>${table(['Component','Formula / rule'],['precision','recall','f1'].filter(key=>present(scoring[key])).map(key=>[esc({precision:'Correctness / precision',recall:'Required-fact coverage / recall',f1:'Factual F1'}[key]),readable(scoring[key])]))}`;
    if(r.required_fact_coverage)html+=`<p class="rubric-policy">${readable(r.required_fact_coverage)}</p>`;
    if(list(r.graded_examples).length) html+=`<h4 class="rubric-section-title">Worked grading examples</h4>${table(['Dimension','Available evidence','Candidate wording','Score','Why'],r.graded_examples.map(item=>[esc(item.dimension),readable(item.evidence),readable(item.candidate),esc(item.score),readable(item.rationale)]),'Illustrative judgments; no model evaluation was run')}`;
    html+=`<h4 class="rubric-section-title">Organization and instruction following</h4>${r.structure_instruction_separation?`<p class="rubric-policy">${esc(r.structure_instruction_separation)}</p>`:''}<div class="rubric-anchor-grid">${anchors('Structure',r.structure_score_anchors)}${anchors('Instruction following',r.instruction_score_anchors)}</div>`;
    if(list(r.analytic_criteria).length) html+=`<details class="workflow-details rubric-advanced"><summary>Analytic checklist · ${r.analytic_criteria.length} criteria</summary><p class="field-note">These diagnostics explain quality; they do not automatically add points to the primary metric.</p>${table(['Criterion','Group','Decision rule','N/A allowed'],r.analytic_criteria.map(c=>[`${esc(c.label||c.id)}<small class="rubric-criterion-id">${esc(c.id)}</small>`,esc(label(c.group||'')),readable(c.definition),esc(c.allow_not_assessable?'Yes, with explanation':'No')]))}${readable(r.analytic_criteria_policy||{})}</details>`;
    html+=`<details class="workflow-details rubric-advanced"><summary>Claim extraction and grading rules</summary>${r.claim_extraction?`<h4>Extracting claims</h4>${readable(r.claim_extraction)}`:''}${r.rules?`<h4>Grading rules</h4>${readable(r.rules)}`:''}${r.claim_verdicts&&r.correctness_score_anchors?`<h4>Claim diagnostics</h4>${readable(r.claim_verdicts)}`:''}</details>`;
    const policies=Object.fromEntries(Object.entries(scoring).filter(([key])=>!['primary','version','components_separate','field_weights','unknown_gt_weight','precision','recall','f1','index'].includes(key)));
    html+=`<details class="workflow-details rubric-advanced"><summary>Aggregation, comparisons and exploratory index</summary>${readable(policies)}${scoring.index?`<h4>${esc(scoring.index.label||'Exploratory index')}</h4>${readable(scoring.index)}`:''}</details>`;
    if(r.calibration)html+=`<details class="workflow-details rubric-advanced"><summary>Human calibration and scientific release</summary>${readable(r.calibration)}</details>`;
    if(r.provenance)html+=`<details class="workflow-details rubric-advanced"><summary>Design provenance and research sources</summary>${readable(r.provenance)}</details>`;
    const known=new Set(['id','status','schema_version','task','fields','correctness_score_anchors','coverage_score_anchors','structure_score_anchors','instruction_score_anchors','claim_verdicts','required_fact_coverage','section_scope','scoring','graded_examples','structure_instruction_separation','analytic_criteria','analytic_criteria_policy','claim_extraction','rules','calibration','provenance']);
    const additional=Object.fromEntries(Object.entries(r).filter(([key])=>!known.has(key)));
    if(Object.keys(additional).length)html+=`<details class="workflow-details rubric-advanced"><summary>Additional rubric definitions</summary>${readable(additional)}</details>`;
    return html+'</div>';
  }
  const api={renderRubric,scoreLabels};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
  if(root)root.MousebenchRubric=api;
})(typeof window!=='undefined'?window:globalThis);
