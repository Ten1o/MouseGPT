'use strict';

(() => {
  const fields = ['Overall', 'Head', 'Limb', 'Torso', 'Others', 'Keywords'];
  const owns = (value, key) => Object.prototype.hasOwnProperty.call(value, key);
  const clone = value => structuredClone(value);

  function summaryFromSections(sections) {
    return fields.map(field => `${field}: ${sections?.[field] ?? ''}`).join('\n\n');
  }

  function capture(previous, edits = {}) {
    const content = clone(previous);
    if (owns(edits, 'sections')) {
      content.sections = Object.assign(content.sections || {}, clone(edits.sections));
    }
    if (owns(edits, 'facts')) {
      const originals = Array.isArray(previous.facts) ? previous.facts : [];
      const rows = new Map(edits.facts.map(fact => [fact.id, fact]));
      const originalIds = new Set(originals.map(fact => fact.id));
      // The form groups rows by section; the saved fact order remains unchanged.
      content.facts = originals.filter(fact => rows.has(fact.id))
        .map(fact => Object.assign(clone(fact), clone(rows.get(fact.id))));
      content.facts.push(...edits.facts.filter(fact => !originalIds.has(fact.id)).map(clone));
    }
    if (owns(edits, 'uncertainties')) content.uncertainties = clone(edits.uncertainties);
    if (owns(edits, 'reference_description')) content.reference_description = edits.reference_description;

    const oldSummary = previous.reference_description;
    const generated = oldSummary == null || (typeof oldSummary === 'string' && !oldSummary.trim())
      || oldSummary === summaryFromSections(previous.sections);
    const summaryUnedited = !owns(edits, 'reference_description') || edits.reference_description === oldSummary;
    const sectionsChanged = summaryFromSections(content.sections) !== summaryFromSections(previous.sections);
    if (owns(edits, 'sections') && generated && summaryUnedited && (sectionsChanged || oldSummary === summaryFromSections(previous.sections))) {
      content.reference_description = summaryFromSections(content.sections);
    }
    return content;
  }

  const api = {capture, summaryFromSections};
  if (typeof window !== 'undefined') window.MousebenchReferenceEdit = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})();
