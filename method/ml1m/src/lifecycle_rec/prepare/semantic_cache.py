import argparse
import json
from pathlib import Path
METADATA_FIELDS = ['title', 'year', 'ml1m_genres', 'tmdb_genres', 'overview', 'director', 'writer', 'actors', 'production_countries', 'original_language']
LLM_PROFILE_FIELDS = ['surface_genres', 'core_themes', 'narrative_style', 'emotional_tone', 'story_keywords', 'audience_preference_facets', 'semantic_keywords', 'tags', 'named_entities', 'recommendation_profile']

def title_without_trailing_year(title, year):
    text = '' if title is None else str(title).strip()
    year = '' if year is None else str(year).strip()
    if year and text.endswith(f'({year})'):
        return text[:-len(f'({year})')].strip()
    return text

def is_empty(value):
    return value is None or value == '' or value == [] or (value == {})

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
    total = 0
    ids = set()
    duplicate_ids = []
    empty_counts = {'metadata': {field: 0 for field in METADATA_FIELDS}, 'llm_profile': {field: 0 for field in LLM_PROFILE_FIELDS}}
    with input_path.open(encoding='utf-8') as src, output_path.open('w', encoding='utf-8') as out:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            total += 1
            obj = json.loads(line)
            item_id = int(obj['item_id'])
            if item_id in ids:
                duplicate_ids.append(item_id)
            ids.add(item_id)
            metadata = obj.get('metadata') or {}
            llm_profile = obj.get('llm_profile') or {}
            metadata_semantic = {field: metadata.get(field) for field in METADATA_FIELDS}
            if is_empty(metadata_semantic.get('title')):
                metadata_semantic['title'] = title_without_trailing_year(metadata.get('ml1m_title') or obj.get('title'), metadata_semantic.get('year'))
            llm_semantic = {field: llm_profile.get(field) for field in LLM_PROFILE_FIELDS}
            for field, value in metadata_semantic.items():
                empty_counts['metadata'][field] += int(is_empty(value))
            for field, value in llm_semantic.items():
                empty_counts['llm_profile'][field] += int(is_empty(value))
            record = {'item_id': item_id, 'title': metadata_semantic.get('title') or obj.get('title'), 'metadata': metadata_semantic, 'llm_profile': llm_semantic}
            out.write(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n')
    report = {'status': 'ok' if not duplicate_ids else 'warning', 'input_path': str(input_path), 'output_path': str(output_path), 'total': total, 'unique_item_ids': len(ids), 'duplicate_item_ids': duplicate_ids, 'metadata_fields': METADATA_FIELDS, 'llm_profile_fields': LLM_PROFILE_FIELDS, 'empty_counts': empty_counts}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
