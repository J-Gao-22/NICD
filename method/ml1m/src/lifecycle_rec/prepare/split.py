from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
SPLIT_SLOTS = {'train': tuple(range(14, 24)), 'valid': tuple(range(24, 29)), 'test': tuple(range(29, 34))}
SPLIT_FLAGS = {'train': -1, 'valid': 0, 'test': 1}
EXPECTED = {'train': {'total': 33891, 'positive': 18191, 'negative': 15700}, 'valid': {'total': 10401, 'positive': 5495, 'negative': 4906}, 'test': {'total': 7331, 'positive': 3998, 'negative': 3333}}
EXPECTED_ACTIVE_USERS = 838
EXPECTED_ACTIVE_ITEMS = 3255

def read_movies(path: Path) -> dict[int, dict[str, Any]]:
    movies: dict[int, dict[str, Any]] = {}
    with path.open('r', encoding='latin-1') as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.rstrip('\n').split('::')
            if len(parts) != 3:
                raise ValueError(f'Malformed movies.dat line {line_number}')
            item_id = int(parts[0])
            title = re.sub('\\s+\\((\\d+)\\)$', '', parts[1])
            genres = parts[2].split('|')
            movies[item_id] = {'title': title, 'genres': genres, 'keywords': ', '.join(genres)}
    return movies

def read_ratings(path: Path, valid_item_ids: set[int]) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    with path.open('r', encoding='latin-1') as handle:
        for source_index, line in enumerate(handle):
            parts = line.rstrip('\n').split('::')
            if len(parts) != 4:
                raise ValueError(f'Malformed ratings.dat line {source_index + 1}')
            user_id, item_id, rating, timestamp = map(int, parts)
            if item_id not in valid_item_ids:
                continue
            rows.append({'source_index': source_index, 'user_id': user_id, 'item_id': item_id, 'rating': rating, 'timestamp': timestamp})
    if not rows:
        raise ValueError('No ratings were loaded')
    return rows

def first_seen_mapping(values: Iterable[int]) -> dict[int, int]:
    mapping: dict[int, int] = {0: 0}
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping)
    return mapping

def month_slot(timestamp: int, min_year: int, min_month: int) -> int:
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return (dt.year - min_year) * 12 + dt.month - min_month

def split_for_slot(slot: int) -> str | None:
    for split, slots in SPLIT_SLOTS.items():
        if slot in slots:
            return split
    return None

def add_strict_histories(rows: list[dict[str, Any]]) -> None:
    by_user: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_user[int(row['user_id'])].append(row)
    for user_rows in by_user.values():
        user_rows.sort(key=lambda row: (int(row['timestamp']), int(row['source_index'])))
        history_original: list[int] = []
        history_thinkrec: list[int] = []
        history_labels: list[int] = []
        history_timestamps: list[int] = []
        history_splits: list[str] = []
        cursor = 0
        while cursor < len(user_rows):
            timestamp = int(user_rows[cursor]['timestamp'])
            end = cursor + 1
            while end < len(user_rows) and int(user_rows[end]['timestamp']) == timestamp:
                end += 1
            for row in user_rows[cursor:end]:
                row['history_item_ids'] = [0, *history_original]
                row['thinkrec_his'] = [0, *history_thinkrec]
                row['his_label'] = [0, *history_labels]
                row['history_timestamps'] = [0, *history_timestamps]
                row['history_splits'] = ['padding', *history_splits]
            for row in user_rows[cursor:end]:
                history_original.append(int(row['item_id']))
                history_thinkrec.append(int(row['thinkrec_iid']))
                history_labels.append(int(row['label']))
                history_timestamps.append(int(row['timestamp']))
                history_splits.append(str(row['split']))
            cursor = end

def json_dump(path: Path, value: Any) -> None:
    with path.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write('\n')

def jsonl_dump(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
            count += 1
    return count

def public_interaction(row: dict[str, Any]) -> dict[str, Any]:
    return {'source_index': int(row['source_index']), 'user_id': int(row['user_id']), 'item_id': int(row['item_id']), 'thinkrec_uid': int(row['thinkrec_uid']), 'thinkrec_iid': int(row['thinkrec_iid']), 'rating': int(row['rating']), 'label': int(row['label']), 'timestamp': int(row['timestamp']), 'month_slot': int(row['month_slot']), 'split': str(row['split']), 'flag': int(row['flag']), 'not_cold': int(row['not_cold']), 'history_item_ids': [int(x) for x in row['history_item_ids']], 'thinkrec_his': [int(x) for x in row['thinkrec_his']], 'his_label': [int(x) for x in row['his_label']], 'history_timestamps': [int(x) for x in row['history_timestamps']], 'history_splits': list(row['history_splits'])}

def pickle_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame({'uid': np.asarray([row['thinkrec_uid'] for row in rows], dtype=np.int64), 'iid': np.asarray([row['thinkrec_iid'] for row in rows], dtype=np.int64), 'label': np.asarray([row['label'] for row in rows], dtype=np.int64), 'timestamp': np.asarray([row['timestamp'] for row in rows], dtype=np.int64), 'his': [np.asarray(row['thinkrec_his'], dtype=np.int64) for row in rows], 'flag': np.asarray([row['flag'] for row in rows], dtype=np.int64), 'his_label': [np.asarray(row['his_label'], dtype=np.int64) for row in rows], 'not_cold': np.asarray([row['not_cold'] for row in rows], dtype=np.int64)}, columns=['uid', 'iid', 'label', 'timestamp', 'his', 'flag', 'his_label', 'not_cold'])

def positive_row(row: dict[str, Any], positive_index: int, split_index: int) -> dict[str, Any]:
    return {'user_id': int(row['user_id']), 'item_id': int(row['item_id']), 'thinkrec_uid': int(row['thinkrec_uid']), 'thinkrec_iid': int(row['thinkrec_iid']), 'rating': int(row['rating']), 'label': 1, 'timestamp': int(row['timestamp']), 'positive_index': positive_index, 'split_index': split_index, 'split': str(row['split']), 'not_cold': int(row['not_cold'])}

def negative_row(row: dict[str, Any], negative_index: int) -> dict[str, Any]:
    return {'user_id': int(row['user_id']), 'item_id': int(row['item_id']), 'thinkrec_uid': int(row['thinkrec_uid']), 'thinkrec_iid': int(row['thinkrec_iid']), 'rating': int(row['rating']), 'label': 0, 'timestamp': int(row['timestamp']), 'negative_index': negative_index, 'split': str(row['split']), 'not_cold': int(row['not_cold'])}

def assert_expected_counts(rows_by_split: dict[str, list[dict[str, Any]]]) -> None:
    for split, expected in EXPECTED.items():
        rows = rows_by_split[split]
        actual = {'total': len(rows), 'positive': sum((int(row['label']) for row in rows))}
        actual['negative'] = actual['total'] - actual['positive']
        if actual != expected:
            raise AssertionError(f'{split} counts differ: expected={expected}, actual={actual}')
    active_users = {int(row['user_id']) for rows in rows_by_split.values() for row in rows}
    active_items = {int(row['item_id']) for rows in rows_by_split.values() for row in rows}
    if len(active_users) != EXPECTED_ACTIVE_USERS or len(active_items) != EXPECTED_ACTIVE_ITEMS:
        raise AssertionError(f'Active cardinality differs: users={len(active_users)} (expected {EXPECTED_ACTIVE_USERS}), items={len(active_items)} (expected {EXPECTED_ACTIVE_ITEMS})')

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def write_checksums(outdir: Path) -> None:
    files = sorted((path for path in outdir.iterdir() if path.is_file() and path.name != 'SHA256SUMS'))
    with (outdir / 'SHA256SUMS').open('w', encoding='utf-8', newline='\n') as handle:
        for path in files:
            handle.write(f'{sha256_file(path)}  {path.name}\n')

def build_user_compatibility_rows(active_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_user: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in active_rows:
        by_user[int(row['user_id'])].append(row)
    summaries: list[dict[str, Any]] = []
    compatibility: list[dict[str, Any]] = []
    for user_id in sorted(by_user):
        rows = sorted(by_user[user_id], key=lambda row: (row['timestamp'], row['source_index']))
        positives = [row for row in rows if row['label'] == 1]
        negatives = [row for row in rows if row['label'] == 0]
        split_positive: dict[str, list[dict[str, Any]]] = {}
        for split in SPLIT_SLOTS:
            split_positive[split] = [{'item_id': int(row['item_id']), 'thinkrec_iid': int(row['thinkrec_iid']), 'rating': int(row['rating']), 'timestamp': int(row['timestamp'])} for row in positives if row['split'] == split]
        negative_public = [{'item_id': int(row['item_id']), 'thinkrec_iid': int(row['thinkrec_iid']), 'rating': int(row['rating']), 'timestamp': int(row['timestamp']), 'split': str(row['split'])} for row in negatives]
        counts = {'total_interactions': len(rows), 'positive_interactions': len(positives), 'negative_side_interactions': len(negatives), 'train_positive': len(split_positive['train']), 'valid_positive': len(split_positive['valid']), 'test_positive': len(split_positive['test'])}
        for split in SPLIT_SLOTS:
            counts[f'{split}_total'] = sum((row['split'] == split for row in rows))
            counts[f'{split}_negative'] = sum((row['split'] == split and row['label'] == 0 for row in rows))
        summaries.append({'user_id': user_id, 'thinkrec_uid': int(rows[0]['thinkrec_uid']), 'counts': counts})
        compatibility.append({'user_id': user_id, 'thinkrec_uid': int(rows[0]['thinkrec_uid']), 'counts': counts, 'positive_time_boundaries': {'last_train_positive_timestamp': split_positive['train'][-1]['timestamp'] if split_positive['train'] else None, 'first_valid_positive_timestamp': split_positive['valid'][0]['timestamp'] if split_positive['valid'] else None, 'last_valid_positive_timestamp': split_positive['valid'][-1]['timestamp'] if split_positive['valid'] else None, 'first_test_positive_timestamp': split_positive['test'][0]['timestamp'] if split_positive['test'] else None}, 'train_positive': split_positive['train'], 'valid_positive': split_positive['valid'], 'test_positive': split_positive['test'], 'negative_side_info': negative_public})
    return (summaries, compatibility)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratings', required=True, type=Path)
    parser.add_argument('--movies', required=True, type=Path)
    parser.add_argument('--outdir', required=True, type=Path)
    parser.add_argument('--skip-expected-count-assertions', action='store_true')
    args = parser.parse_args()
    if args.outdir.exists() and any(args.outdir.iterdir()):
        raise FileExistsError(f'Refusing to overwrite non-empty versioned output directory: {args.outdir}')
    args.outdir.mkdir(parents=True, exist_ok=True)
    movies = read_movies(args.movies)
    all_rows = read_ratings(args.ratings, set(movies))
    min_dt = datetime.fromtimestamp(min((row['timestamp'] for row in all_rows)), tz=timezone.utc)
    user_map = first_seen_mapping((row['user_id'] for row in all_rows))
    item_map = first_seen_mapping((row['item_id'] for row in all_rows))
    active_rows: list[dict[str, Any]] = []
    excluded_by_slot = Counter()
    for raw in all_rows:
        slot = month_slot(raw['timestamp'], min_dt.year, min_dt.month)
        split = split_for_slot(slot)
        if split is None:
            excluded_by_slot[slot] += 1
            continue
        row: dict[str, Any] = dict(raw)
        row.update({'thinkrec_uid': user_map[row['user_id']], 'thinkrec_iid': item_map[row['item_id']], 'label': 1 if row['rating'] >= 4 else 0, 'month_slot': slot, 'split': split, 'flag': SPLIT_FLAGS[split]})
        active_rows.append(row)
    active_rows.sort(key=lambda row: (row['thinkrec_uid'], row['timestamp'], row['source_index']))
    train_users = {row['thinkrec_uid'] for row in active_rows if row['split'] == 'train'}
    train_items = {row['thinkrec_iid'] for row in active_rows if row['split'] == 'train'}
    for row in active_rows:
        row['not_cold'] = int(row['split'] == 'train' or (row['thinkrec_uid'] in train_users and row['thinkrec_iid'] in train_items))
    add_strict_histories(active_rows)
    rows_by_split = {split: [row for row in active_rows if row['split'] == split] for split in SPLIT_SLOTS}
    if not args.skip_expected_count_assertions:
        assert_expected_counts(rows_by_split)
    frames = {split: pickle_frame(rows) for split, rows in rows_by_split.items()}
    frames['train'].to_pickle(args.outdir / 'train_ood2.pkl')
    frames['valid'].to_pickle(args.outdir / 'valid_ood2.pkl')
    frames['test'].to_pickle(args.outdir / 'test_ood2.pkl')
    frames['valid'].sample(frac=0.25, random_state=2025).to_pickle(args.outdir / 'valid_small_ood2.pkl')
    frames['test'].sample(frac=0.25, random_state=2025).to_pickle(args.outdir / 'test_small_ood2.pkl')
    frames['test'].sample(n=200, random_state=2025).to_pickle(args.outdir / 'test_tiny_ood2.pkl')
    id2title = {str(thinkrec_iid): {'title': movies[original_iid]['title'], 'keywords': movies[original_iid]['keywords']} for original_iid, thinkrec_iid in item_map.items() if original_iid != 0}
    json_dump(args.outdir / 'id2title.json', id2title)
    public_rows = [public_interaction(row) for row in active_rows]
    jsonl_dump(args.outdir / 'all_interactions.jsonl', public_rows)
    for split in SPLIT_SLOTS:
        jsonl_dump(args.outdir / f'{split}_interactions.jsonl', (public_interaction(row) for row in rows_by_split[split]))
    positive_index_by_user: Counter[int] = Counter()
    negative_index_by_user: Counter[int] = Counter()
    split_positive_index: dict[tuple[int, str], int] = defaultdict(int)
    positives_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    negatives: list[dict[str, Any]] = []
    for row in active_rows:
        user_id = int(row['user_id'])
        if row['label'] == 1:
            split = str(row['split'])
            positives_by_split[split].append(positive_row(row, positive_index_by_user[user_id], split_positive_index[user_id, split]))
            positive_index_by_user[user_id] += 1
            split_positive_index[user_id, split] += 1
        else:
            negatives.append(negative_row(row, negative_index_by_user[user_id]))
            negative_index_by_user[user_id] += 1
    for split in SPLIT_SLOTS:
        jsonl_dump(args.outdir / f'{split}_positive_interactions.jsonl', positives_by_split[split])
    jsonl_dump(args.outdir / 'negative_side_info.jsonl', negatives)
    user_summaries, compatibility_rows = build_user_compatibility_rows(active_rows)
    jsonl_dump(args.outdir / 'user_split_summary.jsonl', user_summaries)
    jsonl_dump(args.outdir / 'user_positive_splits.jsonl', compatibility_rows)
    jsonl_dump(args.outdir / 'eligible_users.jsonl', ({'user_id': row['user_id'], 'thinkrec_uid': row['thinkrec_uid'], **row['counts']} for row in user_summaries))
    jsonl_dump(args.outdir / 'dropped_users.jsonl', [])
    user_mapping = {'original_to_thinkrec': {str(k): v for k, v in user_map.items()}, 'thinkrec_to_original': {str(v): k for k, v in user_map.items()}}
    item_mapping = {'original_to_thinkrec': {str(k): v for k, v in item_map.items()}, 'thinkrec_to_original': {str(v): k for k, v in item_map.items()}}
    json_dump(args.outdir / 'user_id_mapping.json', user_mapping)
    json_dump(args.outdir / 'item_id_mapping.json', item_mapping)
    split_stats = {}
    for split, rows in rows_by_split.items():
        positive = sum((row['label'] for row in rows))
        split_stats[split] = {'interactions': len(rows), 'positive': positive, 'negative': len(rows) - positive, 'users': len({row['user_id'] for row in rows}), 'items': len({row['item_id'] for row in rows}), 'not_cold': sum((row['not_cold'] for row in rows)), 'cold': sum((1 - row['not_cold'] for row in rows))}
    summary = {'dataset': 'ML-1M', 'splits': split_stats, 'active': {'users': len({row['user_id'] for row in active_rows}), 'items': len({row['item_id'] for row in active_rows}), 'interactions': len(active_rows)}, 'full_raw_mapping': {'users_excluding_padding': len(user_map) - 1, 'items_excluding_padding': len(item_map) - 1}, 'excluded_interactions_by_month_slot': {str(key): value for key, value in sorted(excluded_by_slot.items())}, 'small_outputs': {'valid_small': len(frames['valid'].sample(frac=0.25, random_state=2025)), 'test_small': len(frames['test'].sample(frac=0.25, random_state=2025)), 'test_tiny': 200}}
    config = {'source': {'ratings': str(args.ratings), 'movies': str(args.movies), 'ratings_sha256': sha256_file(args.ratings), 'movies_sha256': sha256_file(args.movies)}, 'utc_month_slots': {split: list(slots) for split, slots in SPLIT_SLOTS.items()}, 'calendar_windows': {'train': ['2001-06-01T00:00:00Z', '2002-04-01T00:00:00Z'], 'valid': ['2002-04-01T00:00:00Z', '2002-09-01T00:00:00Z'], 'test': ['2002-09-01T00:00:00Z', '2003-02-01T00:00:00Z']}, 'label': {'positive': 'rating >= 4', 'negative': 'rating <= 3'}, 'filtering': {'minimum_interactions': None, 'cold_start_deletion': False, 'valid_test_not_cold_annotation': True}, 'software': {'python': os.sys.version, 'numpy': np.__version__, 'pandas': pd.__version__}}
    json_dump(args.outdir / 'split_summary.json', summary)
    json_dump(args.outdir / 'split_config.json', config)
    write_checksums(args.outdir)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
if __name__ == '__main__':
    main()
