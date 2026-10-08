'use strict';

// Read-only rendering of persisted judgments. No inference, score repair, or edits.
(function(root) {
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const list = value => Array.isArray(value) ? value : [];
  const obj = value => value && typeof value === 'object' && !Array.isArray(value) ? value : {};
  const numeric = value => typeof value === 'number' && Number.isFinite(value);
  const content = value => obj(obj(value).content || value);
  const label = value => String(value ?? '').replaceAll('_',' ').replace(/^./,c=>c.toUpperCase());
  const anchor = (kind,id) => `score-${kind}-${encodeURIComponent(String(id ?? ''))}`;
  const note = text => `<p class="score-note">${esc(text)}</p>`;
  const grade = value => numeric(value) ? `${esc(value)} / 1` : 'Not scored';
  const creditClass = value => value === 1 ? 'score-full-credit' : 'score-partial-credit';
  const percentage = value => numeric(value) ? (value * 100).toFixed(1) : '—';
  const raw = value => esc(JSON.stringify(value ?? null,null,2));
  function readable(value) {
    if (Array.isArray(value)) return value.length ? `<ul>${value.map(v=>`<li>${readable(v)}</li>`).join('')}</ul>` : 'None recorded';
    if (value && typeof value === 'object') return `<dl class="score-readable-list">${Object.entries(value).map(([key,v])=>`<div><dt>${esc(label(key))}</dt><dd>${readable(v)}</dd></div>`).join('')}</dl>`;
    return esc(value === null || value === undefined ? 'Not recorded' : value === '' ? '(empty text)' : value);
  }
  function renderSummary(scores,index) {
    const s=obj(scores);
    const metrics=[['Combined index',numeric(index)?index.toFixed(1):'—',null],
      ['Factual F1',percentage(s.factual_f1),null],
      ['Correctness',percentage(s.factual_precision),null],
      ['Required-fact coverage',percentage(s.required_fact_recall),null],
      ['Structure',percentage(s.structure_score),s.structure_score],
      ['Instruction following',percentage(s.instruction_score),s.instruction_score]];
    return `<div class="score-metrics">${metrics.map(([title,value,quarter])=>`<div class="score-metric"><span>${title}</span><strong>${value}<small>${value==='—'?'':' / 100'}</small></strong>${numeric(quarter)?`<small>Rubric grade: ${grade(quarter)}</small>`:''}</div>`).join('')}</div>`;
  }
  function renderArithmetic(scores,index,indexDetails) {
    const s=obj(scores), d=obj(s.denominators), policy=obj(indexDetails);
    const policySummary=Object.fromEntries(['formula','weights','provisional'].filter(key=>Object.prototype.hasOwnProperty.call(policy,key)).map(key=>[key,policy[key]]));
    if (!Object.keys(s).length) return '';
    const fraction=(credit,weight,result)=>numeric(credit)&&numeric(weight)?(weight===0?`No weighted items. Saved result: ${percentage(result)} / 100.`:`${esc(credit)} weighted credit / ${esc(weight)} total weight = ${percentage(result)} / 100`):'The saved judgment does not expose weighted totals.';
    return `<details class="score-arithmetic"><summary>How this score is calculated</summary><dl class="score-readable-list"><div><dt>Correctness (P)</dt><dd>${fraction(d.weighted_correctness_credit,d.claim_weight,s.factual_precision)}</dd></div><div><dt>Required-fact coverage (R)</dt><dd>${fraction(d.weighted_coverage_credit,d.required_fact_weight,s.required_fact_recall)}</dd></div><div><dt>Factual F1</dt><dd>2 × P × R / (P + R). Saved result: ${percentage(s.factual_f1)} / 100.</dd></div><div><dt>Structure / instruction following</dt><dd>${grade(s.structure_score)} / ${grade(s.instruction_score)} respectively. Each is a separate rubric grade, not an average of checklist verdicts.</dd></div><div><dt>Combined index</dt><dd>${numeric(index)?`${index.toFixed(1)} / 100`:'Unavailable'}${Object.keys(policySummary).length?readable(policySummary):'<p>The applied index policy is not included in this response; no weighting is inferred here.</p>'}</dd></div></dl>${note('Weighted totals are reproduced from recorded claim grades and their bound weights. Error labels and severity do not impose another deduction.')}</details>`;
  }
  function renderListItem(row,selected=false) {
    const r=obj(row), id=r.frame_id || r.frame?.id || r.id || '', e=obj(r.eligibility);
    const status=e.status==='unjudged'||r.status==='unjudged'||r.judged===false?'Not judged':e.status==='stale'?'Stale · excluded':e.eligible===false?'Ineligible · excluded':e.status || r.status || (numeric(r.benchmark_index)?'Scored':'Not judged');
    const value=numeric(r.benchmark_index)?`${r.benchmark_index.toFixed(1)} / 100`:'—';
    return `<button type="button" class="score-frame-item${selected?' selected':''}" data-score-frame="${esc(id)}" aria-pressed="${selected?'true':'false'}"><span class="score-frame-id">${esc(id)}</span><span class="score-frame-summary"><span>${esc(label(status))}</span><strong>${value}</strong></span></button>`;
  }
  function renderEligibility(detail) {
    const e=obj(detail.eligibility), judged=!!detail.judgment;
    if (!judged) return '<div class="score-status-note score-pending" role="status"><strong>Not judged</strong><p>No individual assessment is available for this selection. Missing grades are not zero scores.</p></div>';
    const excluded=e.eligible===false;
    const title=excluded?'Excluded from current aggregate':e.eligible===true?'Included in this score group':'Aggregate eligibility not established';
    const reasons=list(e.reasons).map(reason=>`<li>${esc(typeof reason==='string'?reason:reason.message || label(reason.code))}</li>`).join('');
    return `<div class="score-status-note${excluded?' score-excluded':''}" role="note"><strong>${title}</strong>${excluded?'<p>Saved scores remain visible for review. This judgment is not counted in the current model aggregate.</p>':''}${e.status?`<p>${esc(label(e.status))}</p>`:''}${reasons?`<ul>${reasons}</ul>`:''}</div>`;
  }
  function factText(fact) {
    if (typeof fact==='string') return esc(fact);
    const f=obj(fact);
    return `<span class="score-fact-id">${esc(f.id || '')}${f.field?` · ${esc(f.field)}`:''}</span> ${esc(f.text || 'Reference text unavailable')}`;
  }
  function factAnnotations(fact) {
    const f=obj(fact);
    const details=`${list(f.accepted_alternatives).length?`<div class="score-evidence-text"><strong>Accepted alternatives:</strong> ${readable(f.accepted_alternatives)}</div>`:''}${f.evidence_note?`<p class="score-evidence-text"><strong>Reference evidence:</strong> ${esc(f.evidence_note)}</p>`:''}`;
    return details?`<details class="score-fact-details"><summary>Reference details</summary>${details}</details>`:'';
  }
  function renderClaim(claim) {
    const c=obj(claim), weight=numeric(c.weight)?` · Weight ${c.weight}`:'', facts=list(c.facts);
    return `<article class="score-point ${creditClass(c.correctness)}" id="${esc(anchor('claim',c.id))}"><div class="score-point-heading"><span>${esc(c.id)}${c.field?` · ${esc(c.field)}`:''}${esc(weight)}</span><strong class="score-point-grade">${grade(c.correctness)}</strong></div><p class="score-point-text">${esc(c.text || 'Claim text unavailable')}</p>${c.evidence?`<p class="score-evidence-text"><strong>Why:</strong> ${esc(c.evidence)}</p>`:''}${list(c.additional_evidence).length?`<div class="score-evidence-text">${readable(c.additional_evidence)}</div>`:''}${facts.length?`<div class="score-linked-facts"><strong>Ground-truth evidence</strong><ul>${facts.map(f=>`<li>${factText(f)}${factAnnotations(f)}</li>`).join('')}</ul></div>`:list(c.fact_ids).length?note(`Linked reference facts: ${c.fact_ids.join(', ')}. Their text is unavailable in this response.`):''}${list(c.aliases).length?note(`Equivalent claim IDs merged: ${c.aliases.join(', ')}`):''}${c.error_type && c.error_type!=='none'?`<p class="score-diagnostic">${esc(label(c.error_type))}${c.severity && c.severity!=='none'?` · ${esc(c.severity)} error`:''}</p>`:''}</article>`;
  }
  function renderCoverage(coverage,claimLookup) {
    const c=obj(coverage), fact=c.fact || {id:c.fact_id,text:'Reference text unavailable',field:c.field};
    const links=list(c.claim_ids);
    return `<article class="score-point ${creditClass(c.coverage)}" id="${esc(anchor('fact',c.fact_id))}"><div class="score-point-heading"><span>${esc(c.fact_id)}${c.field?` · ${esc(c.field)}`:''}${numeric(c.weight)?` · Weight ${esc(c.weight)}`:''}</span><strong class="score-point-grade">${grade(c.coverage)}</strong></div><p class="score-point-text">${esc(obj(fact).text || (typeof fact==='string'?fact:'Reference text unavailable'))}</p>${factAnnotations(fact)}${c.evidence?`<p class="score-evidence-text"><strong>Why:</strong> ${esc(c.evidence)}</p>`:''}${links.length?`<div class="score-linked-facts"><strong>Candidate claims</strong><ul>${links.map(id=>`<li><a href="#${esc(anchor('claim',id))}" data-score-claim="${esc(id)}">${esc(id)}</a>${claimLookup.has(id)?` — ${esc(claimLookup.get(id).text)}`:''}</li>`).join('')}</ul></div>`:note(c.coverage===0?'Omitted: no candidate claim covers this required fact.':'No linked candidate claim recorded.')}</article>`;
  }
  function renderCriteria(criteria) {
    return `<div class="score-criteria">${criteria.map(value=>{const c=obj(value);return `<article class="score-criterion${c.verdict==='met'?' score-full-credit':''}"><div class="score-point-heading"><strong>${esc(c.label || label(c.id))}</strong><span>${esc({met:'Met',unmet:'Unmet',not_assessable:'Not assessable'}[c.verdict] || label(c.verdict) || 'Not assessed')}</span></div>${c.description || c.definition?`<details class="score-criterion-rule"><summary>Criterion rule</summary>${note(c.description || c.definition)}</details>`:''}${c.evidence?`<p class="score-evidence-text">${esc(c.evidence)}</p>`:note('No rationale recorded.')}</article>`;}).join('')}</div>`;
  }
  function renderReference(reference,frameId) {
    const gt=content(reference), facts=list(gt.facts), sections=obj(gt.sections);
    const available=Object.keys(gt).length>0;
    return `<details class="score-reference"><summary>Bound ground truth · reference description &amp; facts</summary>${available?Object.entries(sections).map(([name,text])=>`<section class="output-section"><h4>${esc(name)}</h4><p class="output-section-text">${esc(text)}</p></section>`).join(''):note('The bound ground-truth revision is unavailable.')}${gt.reference_description&&!Object.keys(sections).length?`<p>${esc(gt.reference_description)}</p>`:''}${facts.length?`<h4>Reference facts</h4><ul class="score-reference-facts">${facts.map(f=>`<li>${factText(f)} ${obj(f).required===true?'<small>Required</small>':'<small>Additional</small>'}${factAnnotations(f)}</li>`).join('')}</ul>`:''}${gt.uncertainties?`<h4>Uncertainties</h4>${readable(gt.uncertainties)}`:''}${frameId?`<button type="button" class="text-button" data-review-gt="${esc(frameId)}">Open current ground-truth review</button>`:''}</details>`;
  }
  function imageUrl(value) {
    // The server exposes bound images through local API paths only.
    if (typeof value!=='string') return '';
    const url=value.trim();
    return /^\/api\//.test(url) ? url : '';
  }
  function renderImages(images) {
    const entries=list(images).map(i=>({...obj(i),safeUrl:imageUrl(obj(i).url)}));
    if (!entries.length) return note('Original input images are unavailable for this selection.');
    const rendered=entries.map((i,index)=>{
      const name=i.label || `View ${index+1}`;
      if (i.available===false || !i.safeUrl) return `<figure class="score-image-unavailable"><div class="score-status-note score-excluded"><strong>${esc(name)} · Image unavailable</strong><p>${esc(i.error || 'No valid local image URL is available.')}</p></div></figure>`;
      return `<figure><a href="${esc(i.safeUrl)}" target="_blank" rel="noopener noreferrer"><img src="${esc(i.safeUrl)}" alt="${esc(i.label || `Original input image ${index+1}`)}"></a><figcaption>${esc(name)}</figcaption></figure>`;
    }).join('');
    return `<div class="score-images">${rendered}</div>${entries.some(i=>i.available!==false&&i.safeUrl)?note('The views are simultaneous observations of the same frame. Click an image to inspect it at original size.'):''}`;
  }
  function renderOutput(prediction) {
    if (!prediction) return note('No saved model answer is available for this frame.');
    const helpers=typeof module!=='undefined'&&module.exports?require('./output-view.js'):{renderModelOutput};
    // The exact response remains unchanged. A strictly extracted code-fence
    // presentation never replaces its original format diagnostics or grades.
    const withoutRaw=view=>helpers.renderModelOutput({text:prediction.text,output_view:view})
      .replace(/<details class="output-raw">[\s\S]*?<\/details>/,'');
    let html=withoutRaw(prediction.output_view);
    if (!prediction.output_view?.json_valid) {
      if (prediction.presentation_view?.json_valid) {
        html='<details class="score-format-details"><summary>Code-fenced response · strict JSON check failed</summary>'+html+'</details><div class="score-presentation-answer"><p class="score-presentation-note">Readable content inside code fence; original format check unchanged.</p>'+withoutRaw(prediction.presentation_view)+'</div>';
      } else {
        html+=`<section class="score-unparsed-response"><h4>Model answer · unchanged text</h4><p class="score-unparsed-answer">${esc(prediction.text || 'No response text was saved.')}</p></section>`;
      }
    }
    return html;
  }
  function renderDetail(detail) {
    const d=obj(detail), j=obj(d.judgment), s=obj(d.scores), frame=obj(d.frame), gt=content(d.ground_truth);
    const frameId=frame.id || frame.frame_id || (typeof d.frame==='string'?d.frame:'') || j.frame_id || '';
    const facts=new Map(list(gt.facts).map(f=>[obj(f).id,f]));
    const claims=(Array.isArray(d.claims)?d.claims:list(s.graded_details?.claims)).map(c=>({...c,facts:Array.isArray(c.facts)?c.facts:list(c.fact_ids).map(id=>facts.get(id)).filter(Boolean)}));
    const coverage=(Array.isArray(d.fact_coverage)?d.fact_coverage:list(s.graded_details?.fact_coverage)).map(c=>({...c,fact:c.fact || facts.get(c.fact_id)}));
    const criteria=Array.isArray(d.criteria)?d.criteria:Object.entries(obj(j.criterion_assessments)).map(([id,value])=>({id,...obj(value)}));
    const claimLookup=new Map(claims.map(c=>[c.id,c]));
    const isGraded=j.schema_version==='mousebench.judgment.v2' || s.scoring_version==='mousebench.graded-factual-f1.v2' || !!s.graded_details;
    const warnings=list(d.warnings).map(w=>typeof w==='string'?w:w.message || label(w.code)).filter(Boolean);
    let assessment=`${renderEligibility(d)}${renderSummary(d.scores,d.benchmark_index)}${renderArithmetic(d.scores,d.benchmark_index,d.index_details)}`;
    if (warnings.length) assessment+=`<details class="score-status-note score-context"><summary>Provenance &amp; calibration context</summary><ul>${warnings.map(w=>`<li>${esc(w)}</li>`).join('')}</ul></details>`;
    if (j.notes) assessment+=`<details class="score-review-section score-rationale-details"><summary>Judge’s rationale</summary><p class="score-rationale">${esc(j.notes)}</p></details>`;
    if (isGraded && claims.length) assessment+=`<section class="score-review-section"><h3>Correctness · ${claims.length} distinct claims</h3>${note('Each claim receives 0, 0.25, 0.5, 0.75 or 1. Section weights determine its contribution to correctness.')}${claims.map(renderClaim).join('')}</section>`;
    if (isGraded && coverage.length) assessment+=`<section class="score-review-section"><h3>Required-fact coverage · ${coverage.length} facts</h3>${note('Coverage asks how much of each required fact the answer conveys. Missing facts remain visible even when only problems are shown.')}${coverage.map(c=>renderCoverage(c,claimLookup)).join('')}</section>`;
    if (d.judgment&&(!isGraded||(!claims.length&&!coverage.length))) assessment+=`<section class="score-review-section"><h3>Saved assessment</h3>${note('Graded claim details are unavailable. The recorded assessment is shown below without inventing partial-credit grades.')}${readable({claims:j.claims || [],fact_coverage:j.fact_coverage || [],supported_fact_ids:j.supported_fact_ids || []})}</section>`;
    if (criteria.length) assessment+=`<section class="score-review-section"><h3>Quality checklist</h3>${note('Structure and instruction following are separate rubric grades, not an average of this checklist. These diagnostics explain the assessment and do not add deductions.')}${renderCriteria(criteria)}</section>`;
    assessment+=renderReference(d.ground_truth,frameId);
    assessment+=`<details class="score-prompt"><summary>Bound input prompt &amp; context</summary>${d.input?readable(content(d.input)):note('No bound input is available.')}</details>`;
    assessment+=`<details class="score-raw"><summary>Raw answer, judgment &amp; version metadata</summary><h4>Raw model answer · unchanged</h4><pre>${esc(d.prediction?.text ?? 'No saved response.')}</pre><h4>Saved judgment</h4><pre>${raw(d.judgment)}</pre><h4>Bindings and eligibility</h4><pre>${raw({bindings:d.bindings,eligibility:d.eligibility,run:d.run,group:d.group,index_calculation:d.index_details,ground_truth_version:obj(d.ground_truth).revision,input_version:obj(d.input).id})}</pre></details>`;
    return `<div class="score-detail-grid"><section class="score-evidence" aria-label="Frame evidence and model answer"><h3>${esc(frameId || 'Frame evidence')}</h3>${renderImages(d.images)}<h3>Model answer</h3>${renderOutput(d.prediction)}</section><section class="score-assessment" aria-label="Individual scoring and rationale">${assessment}</section></div>`;
  }
  const api={renderDetail,renderSummary,renderListItem};
  if (typeof module!=='undefined'&&module.exports) module.exports=api;
  if (root) root.MousebenchScoringReviewView=api;
})(typeof window!=='undefined'?window:globalThis);
