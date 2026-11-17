import argparse
import json
import re
from collections import Counter
from pathlib import Path
LIST_FIELDS = ['core_themes', 'narrative_style', 'emotional_tone', 'story_keywords', 'audience_preference_facets', 'semantic_terms_for_embedding', 'named_entities']
STRUCTURED_FIELDS = ['core_themes', 'narrative_style', 'emotional_tone', 'named_entities']
EXPANDABLE_FIELDS = ['story_keywords', 'audience_preference_facets', 'semantic_terms_for_embedding']
MIN_KEEP = {'story_keywords': 3, 'audience_preference_facets': 3, 'semantic_terms_for_embedding': 6}

def normalize_text(value):
    value = '' if value is None else str(value)
    value = value.strip().lower()
    value = re.sub('[\\u2018\\u2019]', "'", value)
    value = re.sub('[\\u201c\\u201d]', '"', value)
    value = re.sub('[^a-z0-9]+', ' ', value)
    value = re.sub('\\s+', ' ', value).strip()
    return value

def as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value] if value.strip() else []
    return []

def dedupe_within(values):
    seen = set()
    out = []
    removed = 0
    for value in as_list(values):
        norm = normalize_text(value)
        if not norm:
            removed += 1
            continue
        if norm in seen:
            removed += 1
            continue
        seen.add(norm)
        out.append(value)
    return (out, removed)

def remove_exact_overlaps(values, protected_norms, min_keep):
    out = []
    removed = []
    for value in values:
        norm = normalize_text(value)
        if norm in protected_norms and len(values) - len(removed) > min_keep:
            removed.append(value)
        else:
            out.append(value)
    return (out, removed)

def clean_profile(row):
    llm = row.get('llm_profile') or {}
    counters = Counter()
    notes = []
    for field in LIST_FIELDS:
        if field in llm:
            cleaned, removed = dedupe_within(llm.get(field))
            llm[field] = cleaned
            if removed:
                counters[f'{field}:within_removed'] += removed
    structured_norms = set()
    for field in STRUCTURED_FIELDS:
        structured_norms.update((normalize_text(v) for v in as_list(llm.get(field))))
    structured_norms.discard('')
    story_before = list(as_list(llm.get('story_keywords')))
    story_after, removed_story = remove_exact_overlaps(story_before, structured_norms, MIN_KEEP['story_keywords'])
    if removed_story:
        llm['story_keywords'] = story_after
        counters['story_keywords:cross_field_removed'] += len(removed_story)
        notes.append({'field': 'story_keywords', 'removed': removed_story})
    protected_for_facets = set(structured_norms)
    protected_for_facets.update((normalize_text(v) for v in as_list(llm.get('story_keywords'))))
    protected_for_facets.discard('')
    facets_before = list(as_list(llm.get('audience_preference_facets')))
    facets_after, removed_facets = remove_exact_overlaps(facets_before, protected_for_facets, MIN_KEEP['audience_preference_facets'])
    if removed_facets:
        llm['audience_preference_facets'] = facets_after
        counters['audience_preference_facets:cross_field_removed'] += len(removed_facets)
        notes.append({'field': 'audience_preference_facets', 'removed': removed_facets})
    protected_for_terms = set(structured_norms)
    for field in ('story_keywords', 'audience_preference_facets'):
        protected_for_terms.update((normalize_text(v) for v in as_list(llm.get(field))))
    protected_for_terms.discard('')
    terms_before = list(as_list(llm.get('semantic_terms_for_embedding')))
    terms_after, removed_terms = remove_exact_overlaps(terms_before, protected_for_terms, MIN_KEEP['semantic_terms_for_embedding'])
    if removed_terms:
        llm['semantic_terms_for_embedding'] = terms_after
        counters['semantic_terms_for_embedding:cross_field_removed'] += len(removed_terms)
        notes.append({'field': 'semantic_terms_for_embedding', 'removed': removed_terms})
    row['llm_profile'] = llm
    if notes:
        row['dedupe_applied'] = True
        row['dedupe_notes'] = notes
    else:
        row['dedupe_applied'] = False
    return (row, counters)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    counters = Counter()
    length_before = Counter()
    length_after = Counter()
    rows = 0
    changed_rows = 0
    empty_terms = 0
    with input_path.open('r', encoding='utf-8') as fin, output_path.open('w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            before = {field: len(as_list((row.get('llm_profile') or {}).get(field))) for field in LIST_FIELDS}
            for field, value in before.items():
                length_before[f'{field}:sum'] += value
            cleaned, row_counts = clean_profile(row)
            counters.update(row_counts)
            if cleaned.get('dedupe_applied') or row_counts:
                changed_rows += 1
            after = {field: len(as_list((cleaned.get('llm_profile') or {}).get(field))) for field in LIST_FIELDS}
            for field, value in after.items():
                length_after[f'{field}:sum'] += value
            if after['semantic_terms_for_embedding'] == 0:
                empty_terms += 1
            fout.write(json.dumps(cleaned, ensure_ascii=False, sort_keys=True) + '\n')
    report = {'input': str(input_path), 'output': str(output_path), 'rows': rows, 'changed_rows': changed_rows, 'empty_semantic_terms_for_embedding': empty_terms, 'dedupe_counters': dict(sorted(counters.items())), 'field_lengths_before': dict(sorted(length_before.items())), 'field_lengths_after': dict(sorted(length_after.items())), 'notes': {'within_field': 'Exact normalized duplicates are removed in all list fields.', 'cross_field': 'Exact overlaps are removed only from larger expandable fields, with per-field minimum keep counts.', 'min_keep': MIN_KEEP}}
    with report_path.open('w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write('\n')
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
if __name__ == '__main__':
    main()
