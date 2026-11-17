import argparse
import csv
import json
from collections import Counter
from pathlib import Path

def split_pipe(value):
    if value is None:
        return []
    value = str(value).strip()
    if not value:
        return []
    return [part.strip() for part in value.split('|') if part.strip()]

def clean_scalar(value):
    if value is None:
        return ''
    return str(value).strip()

def load_metadata(path):
    rows = {}
    ids = []
    with open(path, newline='', encoding='utf-8-sig', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            movie_id = int(float(row['MovieID']))
            ids.append(movie_id)
            metadata = {'movie_id': movie_id, 'ml1m_title': clean_scalar(row.get('Title')), 'ml1m_genres': split_pipe(row.get('Genres')), 'year': clean_scalar(row.get('Year')), 'imdb_id': clean_scalar(row.get('imdb_id')), 'tmdb_id': clean_scalar(row.get('id')), 'production_countries': split_pipe(row.get('production_countries')), 'tmdb_genres': split_pipe(row.get('genres')), 'original_language': clean_scalar(row.get('original_language')), 'original_title': clean_scalar(row.get('original_title')), 'title': clean_scalar(row.get('title')), 'overview': clean_scalar(row.get('overview')), 'release_date': clean_scalar(row.get('release_date')), 'vote_average': clean_scalar(row.get('vote_average')), 'director': clean_scalar(row.get('director')), 'writer': clean_scalar(row.get('writer')), 'actors': split_pipe(row.get('actors'))}
            metadata['raw'] = {k: clean_scalar(v) for k, v in row.items()}
            rows[movie_id] = metadata
    dup_ids = sorted([item_id for item_id, n in Counter(ids).items() if n > 1])
    return (rows, dup_ids)

def load_profiles(path):
    rows = {}
    ids = []
    with open(path, encoding='utf-8', errors='replace') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            item_id = int(obj['item_id'])
            ids.append(item_id)
            obj['item_id'] = item_id
            rows[item_id] = obj
    dup_ids = sorted([item_id for item_id, n in Counter(ids).items() if n > 1])
    return (rows, dup_ids)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--llm-profile', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    metadata, metadata_dups = load_metadata(args.metadata)
    profiles, profile_dups = load_profiles(args.llm_profile)
    metadata_ids = set(metadata)
    profile_ids = set(profiles)
    missing_profile = sorted(metadata_ids - profile_ids)
    missing_metadata = sorted(profile_ids - metadata_ids)
    if metadata_dups or profile_dups or missing_profile or missing_metadata:
        report = {'status': 'failed', 'metadata_count': len(metadata), 'llm_profile_count': len(profiles), 'metadata_duplicate_ids': metadata_dups, 'llm_profile_duplicate_ids': profile_dups, 'metadata_ids_missing_llm_profile': missing_profile, 'llm_profile_ids_missing_metadata': missing_metadata}
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        raise SystemExit('ID alignment failed; see report.')
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    confidence_counts = Counter()
    tag_quality_counts = Counter()
    fallback_count = 0
    repair_count = 0
    with open(out_path, 'w', encoding='utf-8') as out:
        for item_id in sorted(metadata_ids):
            llm_profile = profiles[item_id]
            quality = llm_profile.get('quality') or {}
            confidence_counts[llm_profile.get('profile_confidence', '')] += 1
            tag_quality_counts[quality.get('tag_quality', '')] += 1
            fallback_count += int(bool(llm_profile.get('fallback_applied')))
            repair_count += int(bool(llm_profile.get('repair_applied')))
            record = {'item_id': item_id, 'title': metadata[item_id].get('ml1m_title') or llm_profile.get('title'), 'metadata': metadata[item_id], 'llm_profile': llm_profile, 'alignment': {'metadata_id_field': 'MovieID', 'llm_profile_id_field': 'item_id', 'id_match': metadata[item_id]['movie_id'] == llm_profile['item_id'], 'title_match': metadata[item_id].get('ml1m_title') == llm_profile.get('title')}}
            out.write(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n')
    report = {'status': 'ok', 'metadata_path': args.metadata, 'llm_profile_path': args.llm_profile, 'output_path': args.output, 'metadata_count': len(metadata), 'llm_profile_count': len(profiles), 'combined_count': len(metadata_ids), 'id_alignment': {'metadata_duplicate_ids': metadata_dups, 'llm_profile_duplicate_ids': profile_dups, 'metadata_ids_missing_llm_profile': missing_profile, 'llm_profile_ids_missing_metadata': missing_metadata}, 'llm_profile_confidence_counts': dict(sorted(confidence_counts.items())), 'llm_tag_quality_counts': dict(sorted(tag_quality_counts.items())), 'repair_applied_count': repair_count, 'fallback_applied_count': fallback_count}
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
