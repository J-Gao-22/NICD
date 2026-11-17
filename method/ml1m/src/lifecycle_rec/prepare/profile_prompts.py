from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List
SCHEMA_KEYS = ['surface_genres', 'core_themes', 'narrative_style', 'emotional_tone', 'story_keywords', 'audience_preference_facets', 'semantic_tags', 'named_entities', 'recommendation_profile', 'evidence_used', 'profile_confidence']
EVIDENCE_PRIORITY = ['overview', 'genres', 'title', 'year', 'director', 'actors', 'writer', 'production_countries', 'original_language']

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--movie-side-csv', type=str, default='movie_side_merged_v2.csv')
    parser.add_argument('--item-universe-json', type=str, default='sampled_v4_fullrank/fullrank_item_universe.json')
    parser.add_argument('--prompt-template', type=str, required=True)
    parser.add_argument('--output-jsonl', type=str, required=True)
    parser.add_argument('--take', type=int, default=0)
    parser.add_argument('--max-overview-chars', type=int, default=1600)
    return parser.parse_args()

def split_bar(value: str, limit: int=20) -> List[str]:
    return [part.strip() for part in str(value or '').split('|') if part.strip()][:limit]

def cut(text: str, limit: int) -> str:
    text = str(text or '').strip()
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + ' ...'

def load_universe(path: Path) -> List[int]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    items = payload.get('items', payload) if isinstance(payload, dict) else payload
    out: List[int] = []
    for row in items:
        item_id = row.get('item_id', row.get('movieId', row.get('id'))) if isinstance(row, dict) else row
        if item_id is not None:
            out.append(int(item_id))
    return out

def load_movie_side(path: Path) -> Dict[int, Dict[str, str]]:
    rows: Dict[int, Dict[str, str]] = {}
    with path.open('r', encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            item_id = row.get('movieId') or row.get('item_id') or row.get('id')
            if item_id:
                rows[int(item_id)] = {str(k): str(v or '') for k, v in row.items()}
    return rows

def build_payload(item_id: int, row: Dict[str, str], max_overview_chars: int) -> Dict[str, Any]:
    return {'item_id': item_id, 'title': row.get('title', f'item_{item_id}'), 'year': row.get('year', ''), 'overview': cut(row.get('overview', ''), max_overview_chars), 'genres': split_bar(row.get('genres', ''), limit=12), 'director': row.get('director', ''), 'actors': split_bar(row.get('actors', ''), limit=8), 'writer': row.get('writer', ''), 'production_countries': split_bar(row.get('production_countries', ''), limit=8), 'original_language': row.get('original_language', '')}

def make_prompt(template: str, payload: Dict[str, Any]) -> str:
    evidence = {key: payload.get(key, '') for key in EVIDENCE_PRIORITY if payload.get(key)}
    return template.rstrip() + '\n\nMovie record, ordered by evidence priority:\n' + json.dumps(evidence, ensure_ascii=False, indent=2)

def main() -> None:
    args = parse_args()
    root = Path.cwd()
    template = (root / args.prompt_template).read_text(encoding='utf-8')
    item_ids = load_universe(root / args.item_universe_json)
    movie_side = load_movie_side(root / args.movie_side_csv)
    output = root / args.output_jsonl
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    missing = 0
    with output.open('w', encoding='utf-8') as out:
        for item_id in item_ids:
            row = movie_side.get(item_id, {})
            if not row:
                missing += 1
            payload = build_payload(item_id, row, int(args.max_overview_chars))
            out.write(json.dumps({'item_id': item_id, 'title': payload.get('title') or f'item_{item_id}', 'evidence_priority': EVIDENCE_PRIORITY, 'schema_keys': SCHEMA_KEYS, 'payload': payload, 'prompt': make_prompt(template, payload)}, ensure_ascii=False) + '\n')
            written += 1
            if int(args.take) > 0 and written >= int(args.take):
                break
    print(json.dumps({'output_jsonl': str(output), 'written': written, 'missing_movie_side': missing}, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    main()
