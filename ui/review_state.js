/* Review drafts belong to a canonical source directory and original frame ID. */
state.reviewSessions ||= {};
if (state.reviews && Object.keys(state.reviews).length) {
  state.legacyReviews = state.reviews;
}
delete state.reviews;
delete state.description;
let reviewContext = null;
let loadedReview = null;
let reviewGeneration = 0;
let framePreviewGeneration = 0;

function reviewSession(context) {
  if (!context || !/^[a-f0-9]{64}$/.test(context.session_id)) throw Error('Load a description source first.');
  return state.reviewSessions[context.session_id] ||= {
    description_directory: context.description_directory, reviews: {}
  };
}

function activeReview() {
  return loadedReview && loadedReview.inputDirectory === val('descriptions').trim() ? loadedReview : null;
}

function persistReview() {
  const loaded = activeReview();
  if (!loaded || !$('#review-facts')) return;
  const records = reviewSession(loaded).reviews;
  const record = {...(records[loaded.frame_id] || {}), ...loaded.reference,
    original: loaded.description, updatedAt: new Date().toISOString()};
  document.querySelectorAll('[data-review]').forEach(input => record[input.dataset.review] = input.value);
  records[loaded.frame_id] = record;
  save();
}

function invalidateReview() {
  persistReview();
  ++reviewGeneration;
  ++framePreviewGeneration;
  reviewContext = null;
  loadedReview = null;
  delete state.description;
  delete state.fields.reviewedDescriptions;
  if ($('#review-facts')) renderFacts();
}

async function prepareReviewContext() {
  const inputDirectory = val('descriptions').trim();
  if (!inputDirectory) throw Error('Select the source description directory first.');
  const context = await api('review-source', {descriptions: inputDirectory});
  if (inputDirectory !== val('descriptions').trim()) throw Error('Description source changed. Load the frame again.');
  reviewContext = {...context, inputDirectory};
  const session = reviewSession(context);
  if (session.reviewedDescriptions) state.fields.reviewedDescriptions = session.reviewedDescriptions;
  else delete state.fields.reviewedDescriptions;
  return reviewContext;
}

function acceptReviewFrame(data, context) {
  if (context.inputDirectory !== val('descriptions').trim() || data.session_id !== context.session_id ||
      data.description_directory !== context.description_directory) throw Error('Review source changed. Load the frame again.');
  if (!data.description || typeof data.description !== 'object' || Array.isArray(data.description)) {
    throw Error('Visual review requires a structured description JSON.');
  }
  const reference = Object.fromEntries(['session_id','description_directory','frame_id','source_file','source_sha256'].map(key => [key,data[key]]));
  const session = reviewSession(context);
  const previous = session.reviews[data.frame_id];
  if (previous && previous.source_sha256 !== data.source_sha256) {
    (session.previousDrafts ||= []).push(previous);
    delete session.reviews[data.frame_id];
  }
  loadedReview = {...reference, reference, inputDirectory: context.inputDirectory, description: data.description};
  state.description = data.description;
  state.frame = Number(data.frame_id);
  save();
}

async function loadReviewFrame(frame) {
  persistReview();
  const generation = ++reviewGeneration;
  const context = await prepareReviewContext();
  const data = await job('review', {frame}, true);
  if (generation !== reviewGeneration) throw Error('Another frame was selected. Load the frame again.');
  acceptReviewFrame(data, context);
  renderFacts();
  return data;
}

async function loadFrame() {
  const frame = Number($('#frame')?.value || 0);
  if (!Number.isInteger(frame) || frame < 0) throw Error('Enter a nonnegative frame ID.');
  const generation = ++framePreviewGeneration, page = location.hash;
  const snapshot = Object.fromEntries(['descriptions','crop3','crop5','previews'].map(key => [key,val(key)]));
  const current = () => generation === framePreviewGeneration && page === location.hash &&
    Number(state.frame) === frame && Object.entries(snapshot).every(([key,value]) => val(key) === value);
  persistReview();
  loadedReview = null;
  state.frame = frame;
  delete state.description;
  save(); renderFacts();
  $('#preview-image').hidden = true;
  $('#preview-empty').hidden = false;
  try {
    if (page === '#stage2') {
      const data = await loadReviewFrame(frame);
      if (!current()) return;
      const velocity = data.velocity || {};
      $('#velocity-metrics').innerHTML = [['Body',velocity.base],['Head',velocity.head],
        ['Forepaw L / R',velocity.forepawL == null ? null : Number(velocity.forepawL).toFixed(2)+' / '+Number(velocity.forepawR).toFixed(2)]]
        .map(([label,value]) => `<div class="metric"><b>${value == null ? '—' : typeof value === 'number' ? value.toFixed(2) : esc(value)}</b><span>${label} cm/s</span></div>`).join('');
    }
    await ensureEnvironment('preview-image');
    if (!current()) return;
    const image = await api('preview', {frame, ...(page === '#stage2' ?
      {crop3:snapshot.crop3,crop5:snapshot.crop5,previews:snapshot.previews} : {previews:snapshot.previews})});
    if (!current()) return;
    $('#preview-image').src = image.image;
    $('#preview-image').hidden = false;
    $('#preview-empty').hidden = true;
    $('.preview-label').textContent = 'Frame ' + String(frame).padStart(8,'0');
  } catch (error) {
    if (current()) throw error;
  }
}

async function importReviewDescription(description) {
  if (!description || !['Overall','Head','Limb','Torso','Others','Keywords'].every(key => typeof description[key] === 'string')) {
    throw Error('Expected six structured description fields.');
  }
  const frame = Number($('#frame').value);
  if (!Number.isInteger(frame) || frame < 0) throw Error('Enter a nonnegative frame ID.');
  const data = await loadReviewFrame(frame);
  const keys = Object.keys(data.description);
  if (Object.keys(description).length !== keys.length || !keys.every(key => description[key] === data.description[key])) {
    throw Error('Imported JSON does not match this source frame. Select its matching description directory.');
  }
}

async function saveCurrentReview() {
  persistReview();
  const context = await prepareReviewContext();
  const session = reviewSession(context);
  if (!Object.keys(session.reviews).length) throw Error('Load and review at least one frame in this session.');
  if (!val('out2').trim()) throw Error('Select a Stage 2 output directory.');
  const result = await api('reviews', {output: val('out2') + '/reviews',
    descriptions: context.description_directory, session_id: context.session_id, reviews: session.reviews});
  session.reviewedDescriptions = result.reviewed_descriptions;
  if (context.inputDirectory === val('descriptions').trim()) state.fields.reviewedDescriptions = result.reviewed_descriptions;
  save();
  toast('Saved ' + result.frames + ' frame(s) as a new review snapshot.');
  renderFacts();
}

function exportCurrentReview() {
  persistReview();
  const loaded = activeReview();
  if (!loaded) throw Error('Load a frame from the selected source before exporting reviews.');
  download('mousegpt-reviews.json', {schemaVersion: 3, session_id: loaded.session_id,
    description_directory: loaded.description_directory, reviews: reviewSession(loaded).reviews});
}

function renderFacts() {
  if (!$('#review-facts')) return;
  const loaded = activeReview();
  if (!loaded) {
    $('#review-facts').innerHTML = '<div class="empty">Select a description directory and load a frame. Reviews are kept separately for each source session.</div>';
    if ($('#review-count')) $('#review-count').textContent = '';
    return;
  }
  const records = reviewSession(loaded).reviews, record = records[loaded.frame_id] || {};
  $('#review-facts').innerHTML = Object.entries(loaded.description).map(([category,text]) => {
    const facts = String(text).split(/(?<=[.!?。！？])\s*|[,，;；]\s*/).filter(Boolean);
    const score = state.reviewMode === 'score';
    const items = score ? ['Accuracy','Completeness','Clarity'] : facts;
    const choices = score ? ['Unrated','1','2','3','4','5'] : ['Unreviewed','Correct','Incorrect','Partly correct','Insufficient evidence'];
    return `<section class="review-item"><h3>${esc(category)} · ${category === 'Keywords' ? 'tags' : facts.length + ' fact drafts'}</h3><div class="muted">${esc(text)}</div>${items.map((item,index) => {
      const key = category + '-' + (score ? 'score-' : '') + index;
      return `<div class="fact"><span>${esc(item)}</span><select data-review="${key}" aria-label="${esc(category+' '+item)}">${choices.map(choice => `<option ${record[key]===choice?'selected':''}>${choice}</option>`).join('')}</select></div>`;
    }).join('')}<textarea data-review="${category}-revised" aria-label="${category} revised description" placeholder="Revised description (blank = original)">${esc(record[category+'-revised']||'')}</textarea><textarea data-review="${category}-notes" aria-label="${category} corrections" placeholder="Correction / notes">${esc(record[category+'-notes']||'')}</textarea></section>`;
  }).join('') + `<label class="field"><span>Frame status</span><select data-review="status">${['Unreviewed','Passed','Needs correction'].map(choice => `<option ${record.status===choice?'selected':''}>${choice}</option>`).join('')}</select></label><small class="muted">Fact splitting is a punctuation-based draft. Saved snapshots include all reviewed frames, regardless of status.</small>`;
  if ($('#review-count')) $('#review-count').textContent = Object.keys(records).length + ' reviewed frames in this session';
}
