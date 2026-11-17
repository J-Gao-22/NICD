from __future__ import annotations
import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List
PROFILE_LIST_FIELDS = ['core_themes', 'narrative_style', 'story_keywords', 'audience_preference_facets', 'semantic_tags']
TAG_FIELDS = ['semantic_tags', 'core_themes', 'narrative_style', 'audience_preference_facets']
STOPWORDS = {'a', 'an', 'and', 'are', 'as', 'at', 'by', 'for', 'from', 'in', 'into', 'is', 'of', 'on', 'or', 'the', 'to', 'with', 'film', 'movie', 'story', 'about', 'featuring', 'where', 'between', 'their', 'they', 'them', 'must', 'complete', 'save', 'everyone', 'with', 'that', 'this', 'classic', 'popular', 'entertaining', 'high-rated', 'high', 'rated'}
BANNED_KEYWORD_SUBSTRINGS = {'classic', 'popular', 'entertaining', 'high-rated', 'high rated', 'family fun', 'childhood wonder', 'fantasy escapade', 'adventure-seeking', 'relatable characters', 'humorous situations', 'family-friendly', 'child-friendly', 'sultry', 'divorc', 'magical journey'}
NON_ENTITY_SUBSTRINGS = {'daughter', 'son', 'father', 'mother', 'lobbyist', 'detective', 'president', 'thief', 'lawyer', 'brother', 'sister'}
GENERIC_TAGS = {'adventure', 'fantasy', 'comedy', 'romance', 'animation', "children's", 'drama', 'tense', 'funny', 'warm', 'growth', 'trapped', 'family-friendly', 'family fun', 'childhood wonder', 'fantasy escapade', 'adventure-seeking', 'child-friendly', 'family entertainment', 'relatable characters', 'humorous situations'}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile-jsonl', type=str, required=True)
    parser.add_argument('--output-jsonl', type=str, required=True)
    parser.add_argument('--max-keywords', type=int, default=12)
    parser.add_argument('--max-tags', type=int, default=8)
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

def add_unique(out: List[str], phrase: str, seen: set[str], limit: int, excluded: set[str] | None=None) -> None:
    phrase = normalize_phrase(phrase)
    excluded = excluded or set()
    if not phrase or phrase in seen or len(phrase) < 3:
        return
    if phrase in excluded or any((item and item in phrase for item in excluded)):
        return
    if any((banned in phrase for banned in BANNED_KEYWORD_SUBSTRINGS)):
        return
    seen.add(phrase)
    out.append(phrase)
    if len(out) > limit:
        del out[limit:]

def phrase_tokens(text: str) -> set[str]:
    return set(re.findall("[a-z0-9']+", normalize_phrase(text)))

def supported_by_row(row: Dict[str, Any], phrase: str) -> bool:
    phrase = normalize_phrase(phrase)
    genres = {normalize_phrase(str(item)) for item in row.get('surface_genres', [])} if isinstance(row.get('surface_genres'), list) else set()
    title = normalize_phrase(str(row.get('title', '')))
    profile = normalize_phrase(str(row.get('recommendation_profile', '')))
    text = normalize_phrase(' '.join([title, profile, ' '.join(sorted(genres))]))
    if any((term in phrase for term in ['magical', 'magic', 'enchanted'])):
        if 'fantasy' not in genres and 'adventure' not in genres:
            return False
    if any((term in phrase for term in ['late-life', 'aging', 'elderly', 'retirement'])):
        if not any((term in text for term in ['old', 'older', 'aging', 'elderly', 'retirement', 'late-life', 'grumpy'])):
            return False
    if 'romantic comedy' in phrase:
        if 'romance' not in genres and (not any((term in profile for term in ['romance', 'romantic', 'relationship', 'relationships', 'lover', 'lovers']))):
            return False
    if 'domestic conflict' in phrase:
        if not any((term in text for term in ['neighbor', 'next-door', 'wedding', 'marriage', 'household', 'domestic', 'restaurant'])):
            return False
    return True

def add_high_quality_tag(out: List[str], phrase: str, seen: set[str], limit: int) -> None:
    phrase = normalize_phrase(phrase)
    if not phrase or phrase in GENERIC_TAGS or len(phrase.split()) < 2:
        return
    tokens = phrase_tokens(phrase)
    if not tokens:
        return
    for existing in list(out):
        existing_tokens = phrase_tokens(existing)
        if tokens <= existing_tokens:
            return
        if existing_tokens < tokens:
            out.remove(existing)
            seen.discard(existing)
            break
    add_unique(out, phrase, seen, limit)

def build_keywords(row: Dict[str, Any], max_keywords: int) -> List[str]:
    keywords: List[str] = []
    seen: set[str] = set()
    named_entity_set = {normalize_phrase(str(item)) for item in row.get('named_entities', []) if str(item or '').strip()} if isinstance(row.get('named_entities'), list) else set()
    if isinstance(row.get('semantic_tags'), list):
        for value in row['semantic_tags']:
            if not supported_by_row(row, str(value)):
                continue
            add_unique(keywords, str(value), seen, max_keywords, excluded=named_entity_set)
    for value in derive_tags(row):
        add_unique(keywords, value, seen, max_keywords, excluded=named_entity_set)
    for field in PROFILE_LIST_FIELDS:
        values = row.get(field, [])
        if isinstance(values, str):
            values = [values]
        for value in values if isinstance(values, list) else []:
            if field in {'core_themes', 'audience_preference_facets'} and len(normalize_phrase(str(value)).split()) < 2:
                continue
            if not supported_by_row(row, str(value)):
                continue
            add_unique(keywords, str(value), seen, max_keywords, excluded=named_entity_set)
    return keywords[:max_keywords]

def build_tags(row: Dict[str, Any], max_tags: int) -> List[str]:
    tags: List[str] = []
    seen: set[str] = set()
    for value in derive_tags(row):
        add_high_quality_tag(tags, value, seen, max_tags)
    if isinstance(row.get('semantic_tags'), list):
        for value in row['semantic_tags']:
            if not supported_by_row(row, str(value)):
                continue
            add_high_quality_tag(tags, str(value), seen, max_tags)
    for field in TAG_FIELDS:
        values = row.get(field, [])
        if isinstance(values, str):
            values = [values]
        for value in values if isinstance(values, list) else []:
            if not supported_by_row(row, str(value)):
                continue
            add_high_quality_tag(tags, str(value), seen, max_tags)
    return tags[:max_tags]

def derive_tags(row: Dict[str, Any]) -> List[str]:
    genres = {normalize_phrase(str(item)) for item in row.get('surface_genres', [])} if isinstance(row.get('surface_genres'), list) else set()
    story = ' '.join((str(item) for item in row.get('story_keywords', []) if item))
    themes = ' '.join((str(item) for item in row.get('core_themes', []) if item))
    text = normalize_phrase(' '.join([story, themes, str(row.get('recommendation_profile', ''))]))
    tags: List[str] = []
    if 'animation' in genres and any((g in genres for g in {"children's", 'children', 'family'})):
        tags.append('family animation')
    if 'animation' in genres and any((term in text for term in ['toy', 'toys', 'child', 'bedroom'])):
        tags.append('childhood fantasy')
    if 'adventure' in genres and 'fantasy' in genres and any((g in genres for g in {"children's", 'children', 'family'})):
        tags.append('family fantasy adventure')
    if 'fantasy' in genres and any((term in text for term in ['danger', 'terrifying', 'creature', 'chase', 'rhinoceros', 'monkey'])):
        tags.append('fantasy survival')
    if any((term in text for term in ['rhinoceros', 'rhinoceroses', 'monkey', 'monkeys', 'creature', 'creatures'])):
        tags.append('creature chase')
    if 'comedy' in genres and 'romance' in genres:
        tags.append('romantic comedy')
    return tags

def build_named_entities(row: Dict[str, Any], limit: int=24) -> List[str]:
    values = row.get('named_entities', [])
    if isinstance(values, str):
        values = [values]
    entities: List[str] = []
    seen: set[str] = set()
    for value in values if isinstance(values, list) else []:
        phrase = normalize_phrase(str(value))
        if any((term in phrase for term in NON_ENTITY_SUBSTRINGS)):
            continue
        add_unique(entities, str(value), seen, limit)
    return entities[:limit]

def main() -> None:
    args = parse_args()
    root = Path.cwd()
    profile_path = root / args.profile_jsonl
    output_path = root / args.output_jsonl
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open('w', encoding='utf-8') as out:
        for row in iter_jsonl(profile_path):
            semantic_keywords = build_keywords(row, int(args.max_keywords))
            result = {'item_id': int(row.get('item_id')), 'title': str(row.get('title', '')), 'semantic_keywords': semantic_keywords, 'keywords': semantic_keywords, 'named_entities': build_named_entities(row), 'tags': build_tags(row, int(args.max_tags)), 'recommendation_profile': str(row.get('recommendation_profile', '')), 'source_fields': PROFILE_LIST_FIELDS + ['named_entities', 'recommendation_profile']}
            out.write(json.dumps(result, ensure_ascii=False) + '\n')
            written += 1
    print(json.dumps({'output_jsonl': str(output_path), 'written': written}, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    main()
