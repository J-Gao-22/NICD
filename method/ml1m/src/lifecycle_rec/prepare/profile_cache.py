from __future__ import annotations
import argparse
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
GENERIC_TAGS = {'adventure', 'action', 'animation', 'comedy', 'drama', 'fantasy', 'romance', 'thriller', 'crime', 'mystery', 'horror', "children's", 'family-friendly', 'family fun', 'childhood wonder', 'fantasy escapade', 'adventure-seeking', 'relatable characters', 'humorous situations', 'emotional journey', 'character development', 'intriguing mystery', 'exciting action sequences', 'action-packed', 'tense drama', 'dark humor', 'urban life', 'moral choices'}
GENERIC_SUBSTRINGS = {'relatable character', 'humorous situation', 'emotional journey', 'character development', 'intriguing mystery', 'exciting action sequence', 'family fun', 'childhood wonder', 'fantasy escapade', 'adventure-seeking'}
TAG_EXCLUDE_SUBSTRINGS = {'sultry', 'divorc', 'rhinoceros', 'rhinoceroses', 'evil monkeys', 'local bait shop', "child's bedroom", 'bedroom', 'magical journey'}
NON_ENTITY_SUBSTRINGS = {'daughter', 'son', 'father', 'mother', 'brother', 'sister', 'lobbyist', 'detective', 'president', 'thief', 'lawyer', 'manager', 'assistant', 'shop', 'restaurant', 'bedroom', 'room', 'creatures', 'rhinoceroses', 'monkeys', 'divorc', 'adult', 'humans', 'cyborgs'}
CANONICAL_TOKEN = {'powerful': 'power', 'powers': 'power', 'dynamics': 'power', 'struggle': 'conflict', 'struggles': 'conflict', 'mission': 'plot', 'revenge': 'revenge', 'retaliation': 'revenge', 'rivalry': 'rivalry', 'rivalries': 'rivalry', 'friendship': 'friendship', 'friends': 'friendship', 'romantic': 'romance', 'romance': 'romance', 'mysterious': 'mystery', 'mystery': 'mystery', 'criminal': 'crime', 'crime': 'crime'}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile-jsonl', type=str, required=True)
    parser.add_argument('--keyword-tag-jsonl', type=str, required=True)
    parser.add_argument('--output-jsonl', type=str, required=True)
    parser.add_argument('--quality-report-json', type=str, required=True)
    parser.add_argument('--repair-candidates-jsonl', type=str, required=True)
    parser.add_argument('--min-tags', type=int, default=3)
    parser.add_argument('--max-tags', type=int, default=7)
    parser.add_argument('--max-semantic-keywords', type=int, default=12)
    parser.add_argument('--similarity-threshold', type=float, default=0.72)
    return parser.parse_args()

def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open('r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)

def normalize_phrase(text: str) -> str:
    text = unicodedata.normalize('NFKC', str(text or '')).strip().casefold()
    text = text.replace('_', ' ')
    text = re.sub("[^\\w' -]+", ' ', text, flags=re.UNICODE)
    text = re.sub('\\s+', ' ', text).strip(' -')
    return text

def tokens(text: str) -> List[str]:
    out = []
    for token in re.findall("[a-z0-9']+", normalize_phrase(text)):
        token = CANONICAL_TOKEN.get(token, token)
        if len(token) > 3 and token.endswith('s'):
            token = token[:-1]
        out.append(token)
    return out

def token_set(text: str) -> set[str]:
    return set(tokens(text))

def phrase_similarity(a: str, b: str) -> float:
    ta = token_set(a)
    tb = token_set(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    containment = inter / max(1, min(len(ta), len(tb)))
    jaccard = inter / max(1, union)
    return max(jaccard, 0.85 * containment)

def is_generic(phrase: str) -> bool:
    phrase = normalize_phrase(phrase)
    if not phrase:
        return True
    if phrase in GENERIC_TAGS:
        return True
    if any((part in phrase for part in GENERIC_SUBSTRINGS)):
        return True
    if len(phrase.split()) == 1 and phrase in {'warm', 'tense', 'funny', 'uplifting', 'dramatic', 'playful'}:
        return True
    return False

def is_bad_tag_candidate(phrase: str) -> bool:
    phrase = normalize_phrase(phrase)
    if is_generic(phrase):
        return True
    return any((part in phrase for part in TAG_EXCLUDE_SUBSTRINGS))

def specificity_score(phrase: str) -> float:
    phrase = normalize_phrase(phrase)
    toks = tokens(phrase)
    score = len(set(toks)) + 0.15 * len(phrase)
    if is_generic(phrase):
        score -= 10.0
    if '-' in phrase:
        score += 0.5
    if any((term in phrase for term in ['reconciliation', 'rivalry', 'survival', 'heist', 'investigation', 'conspiracy', 'romantic comedy', 'family animation'])):
        score += 1.0
    return score

def add_unique(out: List[str], phrase: str, limit: int | None=None) -> None:
    phrase = normalize_phrase(phrase)
    if not phrase or phrase in out:
        return
    out.append(phrase)
    if limit is not None and len(out) > limit:
        del out[limit:]

def compress_phrases(phrases: List[str], *, limit: int, threshold: float) -> List[str]:
    ranked: List[str] = []
    for phrase in phrases:
        phrase = normalize_phrase(phrase)
        if not phrase:
            continue
        duplicate_idx = None
        for idx, existing in enumerate(ranked):
            if phrase_similarity(phrase, existing) >= threshold:
                duplicate_idx = idx
                break
        if duplicate_idx is None:
            ranked.append(phrase)
        elif specificity_score(phrase) > specificity_score(ranked[duplicate_idx]):
            ranked[duplicate_idx] = phrase
    ranked.sort(key=specificity_score, reverse=True)
    return ranked[:limit]

def is_named_entity(value: str) -> bool:
    raw = str(value or '').strip()
    phrase = normalize_phrase(raw)
    if not phrase:
        return False
    if any((term in phrase for term in NON_ENTITY_SUBSTRINGS)):
        return False
    parts = raw.split()
    if len(parts) > 4:
        return False
    has_proper_casing = any((part[:1].isupper() or part.isupper() or any((ch.isdigit() for ch in part)) for part in parts))
    known_event_or_org = any((term in phrase for term in ['dirty war', 'lapd', 'fbi', 'cia', 'james bond', 'the skulls']))
    return has_proper_casing or known_event_or_org

def collect_profile_values(profile: Dict[str, Any], keyword_row: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    semantic_pool: List[str] = []
    generic_pool: List[str] = []
    priority_fields = [('semantic_tags', profile.get('semantic_tags', [])), ('tags', keyword_row.get('tags', [])), ('core_themes', profile.get('core_themes', [])), ('narrative_style', profile.get('narrative_style', [])), ('story_keywords', profile.get('story_keywords', [])), ('audience_preference_facets', profile.get('audience_preference_facets', [])), ('semantic_keywords', keyword_row.get('semantic_keywords', []))]
    for _, values in priority_fields:
        if isinstance(values, str):
            values = [values]
        for value in values if isinstance(values, list) else []:
            phrase = normalize_phrase(str(value))
            if not phrase:
                continue
            if is_generic(phrase):
                add_unique(generic_pool, phrase)
            else:
                add_unique(semantic_pool, phrase)
    return (semantic_pool, generic_pool)

def clean_named_entities(profile: Dict[str, Any], keyword_row: Dict[str, Any]) -> List[str]:
    entities: List[str] = []
    for source in [profile.get('named_entities', []), keyword_row.get('named_entities', [])]:
        if isinstance(source, str):
            source = [source]
        for value in source if isinstance(source, list) else []:
            if is_named_entity(str(value)):
                add_unique(entities, str(value), limit=6)
    return entities

def quality_flags(profile: Dict[str, Any], tags: List[str], semantic_keywords: List[str], named_entities: List[str], generic_pool: List[str], args: argparse.Namespace) -> Dict[str, Any]:
    flags = {'tag_quality': 'high', 'has_generic_tags': False, 'has_duplicate_tags': False, 'needs_repair': False, 'repair_reasons': []}
    if not str(profile.get('recommendation_profile', '')).strip():
        flags['repair_reasons'].append('missing_recommendation_profile')
    if not profile.get('core_themes'):
        flags['repair_reasons'].append('empty_core_themes')
    if not profile.get('story_keywords'):
        flags['repair_reasons'].append('empty_story_keywords')
    if len(tags) < int(args.min_tags):
        flags['repair_reasons'].append('too_few_tags')
    if not semantic_keywords:
        flags['repair_reasons'].append('empty_semantic_keywords')
    if generic_pool and len(tags) < int(args.min_tags) + 2:
        flags['has_generic_tags'] = True
    for i, left in enumerate(tags):
        for right in tags[i + 1:]:
            if phrase_similarity(left, right) >= float(args.similarity_threshold):
                flags['has_duplicate_tags'] = True
    if flags['repair_reasons']:
        flags['tag_quality'] = 'low'
    elif flags['has_generic_tags'] or flags['has_duplicate_tags'] or len(tags) < 5:
        flags['tag_quality'] = 'medium'
    flags['needs_repair'] = flags['tag_quality'] == 'low'
    return flags

def main() -> None:
    args = parse_args()
    root = Path.cwd()
    profiles = list(iter_jsonl(root / args.profile_jsonl))
    keyword_rows = {int(row['item_id']): row for row in iter_jsonl(root / args.keyword_tag_jsonl)}
    output_path = root / args.output_jsonl
    quality_path = root / args.quality_report_json
    repair_path = root / args.repair_candidates_jsonl
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = Counter()
    quality_counter = Counter()
    repair_reasons = Counter()
    written = 0
    repair_written = 0
    with output_path.open('w', encoding='utf-8') as out, repair_path.open('w', encoding='utf-8') as repair_out:
        for profile in profiles:
            item_id = int(profile['item_id'])
            keyword_row = keyword_rows.get(item_id, {})
            semantic_pool, generic_pool = collect_profile_values(profile, keyword_row)
            semantic_keywords = compress_phrases(semantic_pool + generic_pool, limit=int(args.max_semantic_keywords), threshold=float(args.similarity_threshold))
            tag_candidates = [p for p in semantic_pool if not is_bad_tag_candidate(p)]
            tags = compress_phrases(tag_candidates, limit=int(args.max_tags), threshold=float(args.similarity_threshold))
            named_entities = clean_named_entities(profile, keyword_row)
            flags = quality_flags(profile, tags, semantic_keywords, named_entities, generic_pool, args)
            quality_counter[flags['tag_quality']] += 1
            report['needs_repair'] += int(flags['needs_repair'])
            report['has_generic_tags'] += int(flags['has_generic_tags'])
            report['has_duplicate_tags'] += int(flags['has_duplicate_tags'])
            repair_reasons.update(flags['repair_reasons'])
            row = {'item_id': item_id, 'title': profile.get('title', ''), 'surface_genres': profile.get('surface_genres', []), 'core_themes': profile.get('core_themes', []), 'narrative_style': profile.get('narrative_style', []), 'emotional_tone': profile.get('emotional_tone', []), 'story_keywords': profile.get('story_keywords', []), 'audience_preference_facets': profile.get('audience_preference_facets', []), 'semantic_keywords': semantic_keywords, 'tags': tags, 'named_entities': named_entities, 'recommendation_profile': profile.get('recommendation_profile', ''), 'evidence_used': profile.get('evidence_used', []), 'profile_confidence': profile.get('profile_confidence', ''), 'quality': flags}
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
            written += 1
            if flags['needs_repair']:
                repair_out.write(json.dumps({'item_id': item_id, 'title': profile.get('title', ''), 'repair_reasons': flags['repair_reasons'], 'current_profile': profile, 'current_cache': row}, ensure_ascii=False) + '\n')
                repair_written += 1
    summary = {'written': written, 'repair_candidates': repair_written, 'quality_counts': dict(quality_counter), 'flag_counts': dict(report), 'repair_reasons': dict(repair_reasons), 'inputs': {'profile_jsonl': str(root / args.profile_jsonl), 'keyword_tag_jsonl': str(root / args.keyword_tag_jsonl)}, 'outputs': {'output_jsonl': str(output_path), 'quality_report_json': str(quality_path), 'repair_candidates_jsonl': str(repair_path)}}
    quality_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    main()
