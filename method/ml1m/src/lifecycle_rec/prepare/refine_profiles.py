import argparse
import html
import json
import re
import string
from collections import Counter
from pathlib import Path
MISLEADING_SEMANTIC_KEYWORDS = {'mission impossible'}

def normalize_text(value):
    value = '' if value is None else str(value)
    value = html.unescape(value)
    value = value.lower().strip()
    value = re.sub('\\(\\d{4}\\)$', '', value).strip()
    value = value.replace('&', 'and')
    value = value.translate(str.maketrans('', '', string.punctuation))
    value = re.sub('\\s+', ' ', value).strip()
    return value

def title_case_entity(value):
    words = []
    for word in str(value).split():
        if word.isupper() or any((ch.isdigit() for ch in word)):
            words.append(word)
        elif len(word) <= 3 and word.lower() in {'ira', 'fbi', 'cia', 'lapd', 'nypd'}:
            words.append(word.upper())
        else:
            words.append(word[:1].upper() + word[1:])
    return ' '.join(words)

def exact_case_from_text(phrase, text):
    phrase = str(phrase).strip()
    if not phrase:
        return phrase
    pattern = re.compile(re.escape(phrase), flags=re.IGNORECASE)
    match = pattern.search(text or '')
    if match:
        return match.group(0)
    return title_case_entity(phrase)

def ordered_dedupe(values):
    out = []
    seen = set()
    for value in values if isinstance(values, list) else []:
        if value is None:
            continue
        text = html.unescape(str(value)).strip()
        if not text:
            continue
        key = normalize_text(text)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out

def clean_semantic_keywords(llm_profile, metadata):
    genres = []
    for field in ('ml1m_genres', 'tmdb_genres'):
        genres.extend(metadata.get(field) or [])
    genres.extend(llm_profile.get('surface_genres') or [])
    genre_norms = {normalize_text(x) for x in genres if normalize_text(x)}
    duplicate_sources = []
    for field in ('tags', 'core_themes', 'audience_preference_facets'):
        duplicate_sources.extend(llm_profile.get(field) or [])
    duplicate_norms = {normalize_text(x) for x in duplicate_sources if normalize_text(x)}
    out = []
    counts = Counter()
    seen = set()
    for value in llm_profile.get('semantic_keywords') or []:
        text = html.unescape(str(value)).strip()
        key = normalize_text(text)
        if not key:
            continue
        if key in genre_norms:
            counts['removed_genre'] += 1
            continue
        if key in duplicate_norms:
            counts['removed_duplicate_with_tags_themes_facets'] += 1
            continue
        if key in MISLEADING_SEMANTIC_KEYWORDS:
            counts['removed_misleading_keyword'] += 1
            continue
        if key in seen:
            counts['removed_internal_duplicate'] += 1
            continue
        seen.add(key)
        out.append(text)
    llm_profile['semantic_keywords'] = out
    return counts

def remove_misleading_terms(llm_profile):
    counts = Counter()
    for field in ('semantic_keywords', 'tags', 'story_keywords'):
        if field not in llm_profile:
            continue
        out = []
        for value in llm_profile.get(field) or []:
            if normalize_text(value) in MISLEADING_SEMANTIC_KEYWORDS:
                counts[f'{field}.removed_misleading_keyword'] += 1
            else:
                out.append(value)
        llm_profile[field] = out
    return counts

def split_ampersand_entity(entity, evidence_text):
    key = normalize_text(entity)
    parts = key.split()
    if len(parts) != 2:
        return None
    a, b = parts
    pattern = re.compile(f'\\b({re.escape(a)})\\b\\s*(?:&|and)\\s*\\b({re.escape(b)})\\b', flags=re.IGNORECASE)
    match = pattern.search(evidence_text or '')
    if not match:
        return None
    return [match.group(1), match.group(2)]

def clean_named_entities(llm_profile, metadata, top_title):
    overview = metadata.get('overview', '') or ''
    evidence_text = ' '.join((str(x) for x in [top_title, metadata.get('title', ''), overview, metadata.get('director', ''), ' '.join(metadata.get('actors') or [])]))
    blocked = {normalize_text(top_title), normalize_text(metadata.get('title')), normalize_text(metadata.get('director'))}
    blocked.update((normalize_text(actor) for actor in metadata.get('actors') or []))
    blocked.discard('')
    out = []
    seen = set()
    counts = Counter()
    for entity in llm_profile.get('named_entities') or []:
        split_entities = split_ampersand_entity(entity, evidence_text)
        candidates = split_entities if split_entities else [entity]
        if split_entities:
            counts['split_ampersand_pair'] += 1
        for candidate in candidates:
            candidate = html.unescape(str(candidate)).strip()
            key = normalize_text(candidate)
            if not key:
                continue
            if key in blocked:
                counts['removed_title_director_actor_overlap'] += 1
                continue
            normalized = exact_case_from_text(candidate, evidence_text)
            norm_key = normalize_text(normalized)
            if norm_key in seen:
                counts['removed_internal_duplicate'] += 1
                continue
            seen.add(norm_key)
            out.append(normalized)
    if out:
        llm_profile['named_entities'] = out
    elif 'named_entities' in llm_profile:
        del llm_profile['named_entities']
    return counts

def compact_dict(obj):
    return {k: v for k, v in obj.items() if v not in (None, '', [], {})}

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
    totals = Counter()
    examples = {}
    with input_path.open(encoding='utf-8') as src, output_path.open('w', encoding='utf-8') as out:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            totals['rows'] += 1
            metadata = dict(row.get('metadata') or {})
            llm_profile = dict(row.get('llm_profile') or {})
            if 'overview' in metadata:
                before = metadata['overview']
                metadata['overview'] = html.unescape(str(metadata['overview']))
                if metadata['overview'] != before:
                    totals['overview_html_unescaped'] += 1
                    examples.setdefault('overview_html_unescaped', row['item_id'])
            before_keywords = list(llm_profile.get('semantic_keywords') or [])
            keyword_counts = clean_semantic_keywords(llm_profile, metadata)
            for key, value in keyword_counts.items():
                totals[f'semantic_keywords.{key}'] += value
                if value:
                    examples.setdefault(f'semantic_keywords.{key}', row['item_id'])
            misleading_counts = remove_misleading_terms(llm_profile)
            for key, value in misleading_counts.items():
                totals[key] += value
                if value:
                    examples.setdefault(key, row['item_id'])
            for field in ('tags', 'core_themes', 'audience_preference_facets'):
                if field in llm_profile:
                    before_len = len(llm_profile.get(field) or [])
                    llm_profile[field] = ordered_dedupe(llm_profile.get(field) or [])
                    totals[f'{field}.dedupe_removed'] += before_len - len(llm_profile[field])
            entity_counts = clean_named_entities(llm_profile, metadata, row.get('title', ''))
            for key, value in entity_counts.items():
                totals[f'named_entities.{key}'] += value
                if value:
                    examples.setdefault(f'named_entities.{key}', row['item_id'])
            metadata = compact_dict(metadata)
            llm_profile = compact_dict(llm_profile)
            row['metadata'] = metadata
            row['llm_profile'] = llm_profile
            out.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    report = {'input': args.input, 'output': args.output, 'rows': totals['rows'], 'counts': dict(sorted(totals.items())), 'misleading_semantic_keyword_blacklist': sorted(MISLEADING_SEMANTIC_KEYWORDS)}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
