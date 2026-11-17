import argparse
import json
import re
from pathlib import Path
GENRE_WORDS = {'action', 'adventure', 'animation', 'children', "children's", 'comedy', 'crime', 'documentary', 'drama', 'fantasy', 'film-noir', 'horror', 'musical', 'mystery', 'romance', 'sci-fi', 'science fiction', 'thriller', 'war', 'western', 'family', 'history', 'music', 'tv movie'}
PROFILE_LIST_FIELDS = ['core_themes', 'narrative_style', 'emotional_tone', 'story_keywords', 'audience_preference_facets', 'semantic_terms_for_embedding']

def read_jsonl(path: Path):
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return []

def clean_scalar(value):
    if value is None:
        return ''
    if isinstance(value, list):
        return '; '.join(as_list(value))
    text = str(value).strip()
    text = re.sub('\\s+', ' ', text)
    if text.lower() == 'no overview found.':
        return ''
    return text

def norm(value):
    value = clean_scalar(value).lower()
    value = re.sub('[\\u2018\\u2019]', "'", value)
    value = re.sub('[\\u201c\\u201d]', '"', value)
    value = re.sub('[^a-z0-9]+', ' ', value)
    return re.sub('\\s+', ' ', value).strip()

def strip_year(title):
    return re.sub('\\s*\\(\\d{4}\\)\\s*$', '', clean_scalar(title)).strip()

def join_list(value):
    return ', '.join(as_list(value))

def dedupe_preserve(values):
    seen = set()
    out = []
    for value in as_list(values):
        key = norm(value)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out

def combine_genres(metadata):
    genres = []
    for field in ('ml1m_genres', 'tmdb_genres'):
        genres.extend(as_list(metadata.get(field)))
    return dedupe_preserve(genres)

def truncate_words(text, max_words=35):
    text = clean_scalar(text)
    if not text:
        return ''
    words = text.split()
    if len(words) <= max_words:
        return text
    return ' '.join(words[:max_words]).rstrip(' ,.;:') + '...'

def country_language_context(metadata):
    countries = as_list(metadata.get('production_countries'))
    language = clean_scalar(metadata.get('original_language'))
    parts = []
    if countries:
        parts.append('produced in ' + ', '.join(countries[:3]))
    if language:
        parts.append('original language ' + language)
    if not parts:
        return ''
    text = '; '.join(parts)
    return text[:1].upper() + text[1:] + '.'

def cleanup_embedding_terms(terms, metadata):
    title_norms = {norm(metadata.get('title')), norm(strip_year(metadata.get('title')))}
    title_norms.discard('')
    director_norms = {norm(x) for x in as_list(metadata.get('director'))}
    actor_norms = {norm(x) for x in as_list(metadata.get('actors'))}
    blocked = title_norms | director_norms | actor_norms
    cleaned = []
    seen = set()
    for term in as_list(terms):
        key = norm(term)
        if not key or key in seen:
            continue
        if key in GENRE_WORDS:
            continue
        if key in blocked:
            continue
        cleaned.append(term)
        seen.add(key)
    return cleaned

def add_line(lines, label, value, period=True):
    if isinstance(value, list):
        value = ', '.join(value)
    value = clean_scalar(value)
    if not value:
        return
    suffix = '.' if period and (not value.endswith(('.', '!', '?'))) else ''
    lines.append(f'{label}: {value}{suffix}')

def build_full_text(row):
    metadata = row.get('metadata') or {}
    llm = row.get('llm_profile') or {}
    title = clean_scalar(metadata.get('title') or row.get('title'))
    year = clean_scalar(metadata.get('year'))
    lines = []
    heading = f'Movie: {title}'
    if year:
        heading += f' ({year})'
    lines.append(heading + '.')
    genres = combine_genres(metadata)
    add_line(lines, 'Genres', genres)
    add_line(lines, 'Overview', metadata.get('overview'))
    people = []
    director = join_list(metadata.get('director'))
    writer = join_list(metadata.get('writer'))
    actors = join_list(metadata.get('actors'))
    if director:
        people.append(f'Director: {director}')
    if writer:
        people.append(f'Writer: {writer}')
    if actors:
        people.append(f'Actors: {actors}')
    if people:
        lines.append('Credits: ' + '; '.join(people) + '.')
    context = country_language_context(metadata)
    if context:
        lines.append('Context: ' + context)
    add_line(lines, 'Recommendation profile', llm.get('recommendation_profile'))
    for label, field in [('Core themes', 'core_themes'), ('Narrative style', 'narrative_style'), ('Emotional tone', 'emotional_tone'), ('Audience preference facets', 'audience_preference_facets'), ('Semantic interest terms', 'semantic_terms_for_embedding'), ('Story cues', 'story_keywords'), ('Named entities', 'named_entities')]:
        add_line(lines, label, llm.get(field))
    return '\n'.join(lines)

def build_semantic_compact_text(row):
    metadata = row.get('metadata') or {}
    llm = row.get('llm_profile') or {}
    title = clean_scalar(metadata.get('title') or row.get('title'))
    year = clean_scalar(metadata.get('year'))
    lines = []
    heading = f'Movie: {title}'
    if year:
        heading += f' ({year})'
    lines.append(heading + '.')
    add_line(lines, 'Recommendation profile', llm.get('recommendation_profile'))
    add_line(lines, 'Core themes', llm.get('core_themes'))
    add_line(lines, 'Narrative style', llm.get('narrative_style'))
    add_line(lines, 'Emotional tone', llm.get('emotional_tone'))
    add_line(lines, 'Audience preference facets', llm.get('audience_preference_facets'))
    add_line(lines, 'Semantic interest terms', cleanup_embedding_terms(llm.get('semantic_terms_for_embedding'), metadata))
    add_line(lines, 'Story cues', llm.get('story_keywords'))
    return '\n'.join(lines)

def build_balanced_semantic_text(row):
    metadata = row.get('metadata') or {}
    llm = row.get('llm_profile') or {}
    title = clean_scalar(metadata.get('title') or row.get('title'))
    year = clean_scalar(metadata.get('year'))
    lines = []
    movie_line = f'Movie: {title}'
    if year:
        movie_line += f' ({year})'
    lines.append(movie_line + '.')
    genres = combine_genres(metadata)
    if genres:
        lines.append('Genres: ' + ', '.join(genres) + '.')
    director = join_list(metadata.get('director'))
    actors = as_list(metadata.get('actors'))[:4]
    creative_parts = []
    if director:
        creative_parts.append(f'directed by {director}')
    if actors:
        creative_parts.append('starring ' + ', '.join(actors))
    if creative_parts:
        lines.append('Creative signals: ' + '; '.join(creative_parts) + '.')
    context = country_language_context(metadata)
    if context:
        lines.append('Context: ' + context)
    add_line(lines, 'Preference profile', llm.get('recommendation_profile'))
    add_line(lines, 'Core themes', llm.get('core_themes'))
    add_line(lines, 'Narrative style', llm.get('narrative_style'))
    add_line(lines, 'Emotional tone', llm.get('emotional_tone'))
    add_line(lines, 'Audience preference facets', llm.get('audience_preference_facets'))
    add_line(lines, 'Semantic interest terms', cleanup_embedding_terms(llm.get('semantic_terms_for_embedding'), metadata))
    add_line(lines, 'Story cues', llm.get('story_keywords'))
    short_overview = truncate_words(metadata.get('overview'), 35)
    if not short_overview:
        short_overview = clean_scalar(llm.get('recommendation_profile'))
    add_line(lines, 'Short plot context', short_overview)
    return '\n'.join(lines)

def write_text_rows(rows, output_path, text_type, builder):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    empty = 0
    lengths = []
    with output_path.open('w', encoding='utf-8') as f:
        for row in rows:
            text = builder(row)
            if not text.strip():
                empty += 1
            lengths.append(len(text))
            out = {'item_id': int(row['item_id']), 'title': clean_scalar((row.get('metadata') or {}).get('title') or row.get('title')), 'text_type': text_type, 'text': text}
            f.write(json.dumps(out, ensure_ascii=False, sort_keys=True) + '\n')
            count += 1
    return {'path': str(output_path), 'rows': count, 'empty_text': empty, 'text_length_chars': {'min': min(lengths) if lengths else 0, 'max': max(lengths) if lengths else 0, 'avg': sum(lengths) / len(lengths) if lengths else 0}}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--report', default='')
    args = parser.parse_args()
    input_path = Path(args.input)
    outdir = Path(args.outdir)
    if not input_path.exists():
        raise FileNotFoundError(f'Input cache not found: {input_path}')
    rows = list(read_jsonl(input_path))
    rows.sort(key=lambda r: int(r['item_id']))
    ids = [int(r['item_id']) for r in rows]
    outputs = {'full': write_text_rows(rows, outdir / 'item_text_full.jsonl', 'full', build_full_text), 'semantic_compact': write_text_rows(rows, outdir / 'item_text_semantic_compact.jsonl', 'semantic_compact', build_semantic_compact_text), 'balanced_semantic': write_text_rows(rows, outdir / 'item_text_balanced_semantic.jsonl', 'balanced_semantic', build_balanced_semantic_text)}
    report = {'input': str(input_path), 'outdir': str(outdir), 'rows': len(rows), 'unique_ids': len(set(ids)), 'duplicate_ids': len(ids) - len(set(ids)), 'outputs': outputs}
    report_path = Path(args.report) if args.report else outdir / 'build_balanced_semantic_text.report.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open('w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write('\n')
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
if __name__ == '__main__':
    main()
