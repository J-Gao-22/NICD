import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path
SEMANTIC_BACKFILL_FIELDS = ['tags', 'story_keywords', 'core_themes', 'narrative_style', 'audience_preference_facets', 'surface_genres']

def is_empty(value):
    return value is None or value == '' or value == [] or (value == {})

def normalize_text(value):
    value = '' if value is None else str(value)
    value = value.lower().strip()
    value = re.sub('\\(\\d{4}\\)$', '', value).strip()
    value = value.replace('&', 'and')
    value = value.translate(str.maketrans('', '', string.punctuation))
    value = re.sub('\\s+', ' ', value).strip()
    return value

def ordered_dedupe(values):
    if not isinstance(values, list):
        return values
    out = []
    seen = set()
    for value in values:
        if is_empty(value):
            continue
        key = normalize_text(value)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out

def remove_exact_title(values, title_norm):
    if not isinstance(values, list):
        return (values, 0)
    out = []
    removed = 0
    for value in values:
        if normalize_text(value) == title_norm:
            removed += 1
        else:
            out.append(value)
    return (out, removed)

def remove_named_entity_overlaps(values, blocked_norms):
    if not isinstance(values, list):
        return (values, 0)
    out = []
    removed = 0
    for value in values:
        if normalize_text(value) in blocked_norms:
            removed += 1
        else:
            out.append(value)
    return (out, removed)

def compact_dict(obj):
    return {k: v for k, v in obj.items() if not is_empty(v)}

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
    stats = Counter()
    empty_field_counts = Counter()
    list_dedupe_counts = Counter()
    semantic_backfill_counts = Counter()
    ids = set()
    duplicate_ids = []
    with input_path.open(encoding='utf-8') as src, output_path.open('w', encoding='utf-8') as out:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            stats['input_rows'] += 1
            obj = json.loads(line)
            item_id = int(obj['item_id'])
            if item_id in ids:
                duplicate_ids.append(item_id)
            ids.add(item_id)
            metadata = dict(obj.get('metadata') or {})
            llm_profile = dict(obj.get('llm_profile') or {})
            title = metadata.get('title') or obj.get('title') or ''
            title_norm = normalize_text(title)
            blocked_entities = {title_norm}
            for field in ('director',):
                if metadata.get(field):
                    blocked_entities.add(normalize_text(metadata[field]))
            for actor in metadata.get('actors') or []:
                blocked_entities.add(normalize_text(actor))
            blocked_entities.discard('')
            if 'named_entities' in llm_profile:
                llm_profile['named_entities'], removed = remove_named_entity_overlaps(llm_profile['named_entities'], blocked_entities)
                stats['named_entities_removed_title_director_writer_actor_overlap'] += removed
            for field in ('tags', 'story_keywords'):
                if field in llm_profile:
                    llm_profile[field], removed = remove_exact_title(llm_profile[field], title_norm)
                    stats[f'{field}_removed_exact_title'] += removed
            for field in ('tags', 'semantic_keywords', 'core_themes'):
                if field in llm_profile and isinstance(llm_profile[field], list):
                    before = len(llm_profile[field])
                    llm_profile[field] = ordered_dedupe(llm_profile[field])
                    list_dedupe_counts[field] += before - len(llm_profile[field])
            existing = llm_profile.get('semantic_keywords')
            if not isinstance(existing, list):
                existing = []
            backfilled = list(existing)
            for field in SEMANTIC_BACKFILL_FIELDS:
                values = llm_profile.get(field)
                if isinstance(values, list):
                    backfilled.extend(values)
            backfilled = ordered_dedupe(backfilled)
            added = len(backfilled) - len(ordered_dedupe(existing))
            llm_profile['semantic_keywords'] = backfilled
            semantic_backfill_counts['rows_touched'] += int(added > 0)
            semantic_backfill_counts['terms_added'] += max(0, added)
            for field in ('tags', 'semantic_keywords', 'core_themes'):
                if field in llm_profile and isinstance(llm_profile[field], list):
                    before = len(llm_profile[field])
                    llm_profile[field] = ordered_dedupe(llm_profile[field])
                    list_dedupe_counts[field] += before - len(llm_profile[field])
            for namespace, values in (('metadata', metadata), ('llm_profile', llm_profile)):
                for field, value in list(values.items()):
                    if is_empty(value):
                        empty_field_counts[f'{namespace}.{field}'] += 1
            metadata = compact_dict(metadata)
            llm_profile = compact_dict(llm_profile)
            record = {'item_id': item_id, 'title': title, 'metadata': metadata, 'llm_profile': llm_profile}
            out.write(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n')
            stats['output_rows'] += 1
    report = {'status': 'ok' if not duplicate_ids else 'warning', 'input_path': str(input_path), 'output_path': str(output_path), 'total_rows': stats['input_rows'], 'output_rows': stats['output_rows'], 'unique_item_ids': len(ids), 'duplicate_item_ids': duplicate_ids, 'removal_counts': {'named_entities_removed_title_director_actor_overlap': stats['named_entities_removed_title_director_writer_actor_overlap'], 'tags_removed_exact_title': stats['tags_removed_exact_title'], 'story_keywords_removed_exact_title': stats['story_keywords_removed_exact_title']}, 'dedupe_removed_counts': dict(sorted(list_dedupe_counts.items())), 'semantic_keywords_backfill': dict(sorted(semantic_backfill_counts.items())), 'empty_fields_omitted_counts': dict(sorted(empty_field_counts.items()))}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
