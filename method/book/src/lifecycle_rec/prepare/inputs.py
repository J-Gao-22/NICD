from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import html
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator
import numpy as np
import pandas as pd
YEAR_2017_START = 1483228800
YEAR_2018_START = 1514764800
LIST_FIELDS = ('surface_genres', 'core_themes', 'narrative_style', 'emotional_tone', 'story_keywords', 'audience_preference_facets', 'semantic_tags', 'semantic_terms_for_embedding', 'semantic_signal_terms', 'named_entities')
LENSES = (('C1', 'dominant_feedback_pattern', 'Focus on the strongest overall feedback pattern in this segment. If the segment is mixed, identify the main positive pattern and the main negative contrast. If the segment is mostly positive, emphasize positive preference. If mostly negative, emphasize rejection pattern.'), ('C2', 'positive_negative_contrast', 'Focus on what separates positive interactions from negative interactions. Identify semantic patterns that are liked versus patterns that are rejected or risky. Use this lens to make the state useful for binary Yes/No recommendation.'), ('C3', 'fine_grained_decision_signal', 'Focus on finer-grained signals not captured by the dominant pattern. Look for specific themes, tones, story styles, author signals, or semantic terms that distinguish positive from negative interactions. If no distinct signal exists, return insufficient_distinct_evidence.'))
GENERIC_TERMS = {'book', 'books', 'read', 'reading', 'story', 'stories', 'author', 'authors', 'novel', 'novels', 'fiction', 'nonfiction', 'non fiction', 'kindle', 'edition', 'amazon', 'product', 'literature', 'general', 'unknown', 'none'}
BOOK_CATEGORY_ROOTS = {'book', 'books', 'kindle store'}
CATALOG_NOISE_TERMS = {'isbn', 'brand', 'brand new', 'exactly', 'same', 'listed', 'please', 'notes', 'double', 'check', 'carefully', 'before', 'ordering', 'printed', 'pages', 'black', 'white', 'minor', 'self', 'wear', 'sale', 'restriction', 'delivered', 'delivery', 'working', 'days', 'dhl', 'fedex', 'ups', 'usps', 'marks', 'condition'}
CATALOG_BOILERPLATE_MARKERS = ('exactly same isbn as listed', 'please double check isbn', 'international softcover edition', 'fast delivery through dhl', 'all orders received before', 'money back if not satisfied')

def clean(value: Any) -> str:
    return re.sub('\\s+', ' ', html.unescape(str(value or '')).replace('\n', ' ')).strip()

def clean_author(value: Any) -> str:
    author = clean(value)
    match = re.fullmatch("Visit Amazon(?:'s|’) (.+?) Page", author, flags=re.IGNORECASE)
    return clean(match.group(1)) if match else author

def norm(value: Any) -> str:
    return re.sub('[^a-z0-9]+', ' ', clean(value).casefold()).strip()

def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [clean(x) for x in value if clean(x)]
    return [clean(value)] if clean(value) else []

def dedupe(values: Iterable[Any], limit: int | None=None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = clean(value)
        key = norm(text)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(text)
        if limit is not None and len(out) >= limit:
            break
    return out

def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')

def append_jsonl(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')

def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def parse_raw_response(value: Any) -> dict[str, Any]:
    text = str(value or '').strip()
    if text.startswith('```'):
        text = re.sub('^```(?:json)?\\s*', '', text)
        text = re.sub('\\s*```$', '', text)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        match = re.search('\\{.*\\}', text, flags=re.S)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}

def truncate_words(value: Any, max_words: int) -> str:
    words = clean(value).split()
    if len(words) <= max_words:
        return ' '.join(words)
    return ' '.join(words[:max_words]).rstrip(' ,.;:') + '...'

def is_catalog_boilerplate(value: Any) -> bool:
    text = clean(value).casefold()
    return any((marker in text for marker in CATALOG_BOILERPLATE_MARKERS))

def clean_semantic_list(values: Iterable[Any], limit: int | None=None) -> list[str]:
    cleaned = [value for value in dedupe(values) if norm(value) not in CATALOG_NOISE_TERMS]
    return cleaned[:limit] if limit is not None else cleaned

def title_fallback_terms(title: Any) -> list[str]:
    text = re.sub('\\([^)]*\\)', ' ', clean(title))
    text = re.sub('\\b(?:book|volume|vol)\\s*\\d+\\b', ' ', text, flags=re.I)
    words = [word for word in re.findall("[A-Za-z][A-Za-z'-]{2,}", text) if norm(word) not in GENERIC_TERMS]
    terms = [clean(text)] if clean(text) else []
    terms.extend(words)
    return dedupe(terms, 6)

def build_embedding_terms(row: dict[str, Any]) -> list[str]:
    blocked = {norm(row.get('title'))}
    blocked.update((norm(x) for x in as_list(row.get('authors'))))
    candidates: list[str] = []
    for field in ('semantic_signal_terms', 'semantic_tags', 'core_themes', 'story_keywords', 'narrative_style', 'emotional_tone', 'audience_preference_facets', 'semantic_terms_for_embedding'):
        candidates.extend(as_list(row.get(field)))
    out = []
    for value in dedupe(candidates):
        key = norm(value)
        if key and key not in GENERIC_TERMS and (key not in CATALOG_NOISE_TERMS) and (key not in blocked):
            out.append(value)
        if len(out) >= 18:
            break
    return out

def render_embedding_text(row: dict[str, Any]) -> str:
    lines = [f"Book: {clean(row.get('title'))}."]
    fields = (('Categories', as_list(row.get('categories'))[:6]), ('Author signals', as_list(row.get('authors'))[:3]), ('Preference profile', clean(row.get('recommendation_profile'))), ('Core themes', as_list(row.get('core_themes'))[:8]), ('Narrative style', as_list(row.get('narrative_style'))[:6]), ('Emotional tone', as_list(row.get('emotional_tone'))[:6]), ('Audience preference facets', as_list(row.get('audience_preference_facets'))[:6]), ('Semantic interest terms', as_list(row.get('semantic_terms_for_embedding'))[:18]), ('Story cues', as_list(row.get('story_keywords'))[:10]), ('Short description', truncate_words(row.get('description'), 35)))
    for label, value in fields:
        text = ', '.join(value) if isinstance(value, list) else clean(value)
        if text:
            lines.append(f"{label}: {text}{('' if text.endswith(('.', '!', '?')) else '.')}")
    return '\n'.join(lines)

def prepare_profiles(args: argparse.Namespace) -> None:
    started = time.time()
    source = {int(row['item_id']): row for row in read_jsonl(args.source_profile)}
    merged = {int(row['item_id']): row for row in read_jsonl(args.merged_profile)}
    if set(source) != set(merged):
        raise RuntimeError(f'profile id mismatch: source_only={len(set(source) - set(merged))} merged_only={len(set(merged) - set(source))}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = args.output_dir / 'item_profiles.canonical.jsonl'
    text_path = args.output_dir / 'item_text_balanced_semantic.jsonl'
    stats: Counter[str] = Counter()
    lengths: list[int] = []
    with canonical_path.open('w', encoding='utf-8', newline='\n') as profile_out, text_path.open('w', encoding='utf-8', newline='\n') as text_out:
        for item_id in sorted(source):
            src = source[item_id]
            old = merged[item_id]
            raw = parse_raw_response(old.get('raw_response'))
            nested = src.get('llm_profile') if isinstance(src.get('llm_profile'), dict) else {}
            row: dict[str, Any] = {'item_id': item_id, 'asin': clean(old.get('asin') or src.get('asin')), 'title': clean(old.get('title') or src.get('title')), 'authors': dedupe(as_list(old.get('authors')) + as_list(src.get('authors')), 4), 'categories': dedupe(as_list(old.get('categories')) + as_list(src.get('categories')), 8), 'description': clean(old.get('description') or src.get('description')), 'profile_branch': old.get('profile_branch'), 'content_evidence_status': old.get('content_evidence_status'), 'recommendation_profile': clean(old.get('recommendation_profile') or raw.get('recommendation_profile') or nested.get('recommendation_profile'))}
            for field in LIST_FIELDS:
                current = as_list(old.get(field))
                recovered = as_list(raw.get(field))
                fallback = as_list(nested.get(field))
                chosen = current or recovered or fallback
                row[field] = dedupe(chosen)
                if not current and recovered:
                    stats[f'recovered_{field}_from_raw_response'] += 1
                if row[field]:
                    stats[f'nonempty_{field}'] += 1
            for field in ('story_keywords', 'semantic_tags', 'semantic_terms_for_embedding', 'semantic_signal_terms'):
                row[field] = clean_semantic_list(row[field])
            if is_catalog_boilerplate(row['description']):
                row['description'] = ''
                stats['removed_catalog_boilerplate_description'] += 1
            if is_catalog_boilerplate(row['recommendation_profile']):
                row['recommendation_profile'] = ''
                stats['removed_catalog_boilerplate_recommendation_profile'] += 1
            rebuilt_terms = build_embedding_terms(row)
            if not row['semantic_terms_for_embedding']:
                row['semantic_terms_for_embedding'] = rebuilt_terms
                stats['rebuilt_semantic_terms_for_embedding'] += 1
            else:
                row['semantic_terms_for_embedding'] = clean_semantic_list(row['semantic_terms_for_embedding'] + rebuilt_terms, 18)
            if not row['semantic_signal_terms']:
                row['semantic_signal_terms'] = dedupe(row['semantic_terms_for_embedding'] + as_list(row['core_themes']) + as_list(row['categories']) + title_fallback_terms(row['title']), 12)
                stats['filled_empty_semantic_signal_terms_conservatively'] += 1
            row['semantic_signal_terms'] = clean_semantic_list(row['semantic_signal_terms'])[:12]
            if not row['semantic_terms_for_embedding']:
                row['semantic_terms_for_embedding'] = list(row['semantic_signal_terms'])
                stats['filled_empty_embedding_terms_from_conservative_signals'] += 1
            row['semantic_signal_terms_count'] = len(row['semantic_signal_terms'])
            row['canonical_profile_version'] = 'amazon_book_72b_prep_v1'
            row['llm_profile'] = {field: row[field] for field in LIST_FIELDS}
            row['llm_profile']['recommendation_profile'] = row['recommendation_profile']
            text = render_embedding_text(row)
            lengths.append(len(text))
            append_jsonl(profile_out, row)
            append_jsonl(text_out, {'item_id': item_id, 'title': row['title'], 'text_type': 'book_balanced_semantic_v1', 'text': text})
    report = {'version': 'amazon_book_profile_and_embedding_text_v1', 'rows': len(source), 'unique_item_ids': len(source), 'canonical_profile': str(canonical_path), 'embedding_text': str(text_path), 'stats': dict(stats), 'text_chars': {'min': min(lengths), 'max': max(lengths), 'mean': sum(lengths) / len(lengths)}, 'source_files': {'source_profile': str(args.source_profile), 'merged_profile': str(args.merged_profile), 'source_profile_sha256': file_sha256(args.source_profile), 'merged_profile_sha256': file_sha256(args.merged_profile)}, 'elapsed_sec': time.time() - started}
    write_json(args.output_dir / 'profile_prep_report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

@dataclass(frozen=True)
class PklRow:
    split: str
    row_index: int
    uid: int
    iid: int
    label: int
    timestamp: int
    his: tuple[int, ...]
    title: str
    flag: int
    not_cold: int

def load_pkl_rows(data_dir: Path) -> tuple[list[PklRow], dict[int, list[PklRow]], dict[str, int]]:
    rows: list[PklRow] = []
    by_uid: dict[int, list[PklRow]] = defaultdict(list)
    counts: dict[str, int] = {}
    for split in ('train', 'valid', 'test'):
        frame = pd.read_pickle(data_dir / f'{split}_ood2.pkl')
        counts[split] = len(frame)
        for row_index, row in enumerate(frame.itertuples(index=False)):
            item = PklRow(split=split, row_index=row_index, uid=int(row.uid), iid=int(row.iid), label=int(row.label), timestamp=int(row.timestamp), his=tuple((int(x) for x in row.his if int(x) != 0)), title=clean(row.title), flag=int(row.flag), not_cold=int(row.not_cold))
            rows.append(item)
            by_uid[item.uid].append(item)
    for user_rows in by_uid.values():
        user_rows.sort(key=lambda x: (x.timestamp, {'train': 0, 'valid': 1, 'test': 2}[x.split], x.row_index))
    return (rows, by_uid, counts)

def iter_books_csv(path: Path) -> Iterator[tuple[int, str, str, float, int]]:
    with path.open('r', encoding='utf-8', errors='replace', newline='') as handle:
        for source_order, row in enumerate(csv.reader(handle)):
            if len(row) != 4:
                continue
            asin, raw_uid, rating_raw, timestamp_raw = row
            try:
                rating = float(rating_raw)
                timestamp = int(timestamp_raw)
            except (TypeError, ValueError, OverflowError):
                continue
            if YEAR_2017_START <= timestamp < YEAR_2018_START:
                yield (source_order, clean(asin), clean(raw_uid), rating, timestamp)

def iter_identity_reviews(path: Path) -> Iterator[tuple[int, str, str, float, int]]:
    if path.suffix.lower() == '.csv':
        yield from iter_books_csv(path)
        return
    opener = gzip.open if path.suffix.lower() == '.gz' else open
    with opener(path, 'rt', encoding='utf-8', errors='replace') as handle:
        for source_order, line in enumerate(handle):
            try:
                row = json.loads(line)
                asin = clean(row.get('asin'))
                raw_uid = clean(row.get('reviewerID'))
                rating = float(row.get('overall'))
                timestamp = int(row.get('unixReviewTime'))
            except (json.JSONDecodeError, TypeError, ValueError, OverflowError):
                continue
            if asin and raw_uid:
                yield (source_order, asin, raw_uid, rating, timestamp)

def choose_probes(user_rows: list[PklRow], count: int=3) -> list[PklRow]:
    ranked = sorted(user_rows, key=lambda x: (-len(x.his), -x.timestamp, x.iid, x.row_index))
    chosen = ranked[:1]
    for row in (user_rows[0], user_rows[-1]):
        if row not in chosen:
            chosen.append(row)
    for row in ranked:
        if row not in chosen:
            chosen.append(row)
        if len(chosen) >= count:
            break
    return chosen[:count]

def probe_signature(row: PklRow, iid_to_asin: dict[int, str]) -> tuple[str, int, int]:
    return (iid_to_asin[row.iid], row.timestamp, row.label)

def event_matches_history(events: list[dict[str, Any]], row: PklRow) -> bool:
    positions = [i for i, event in enumerate(events) if event['iid'] == row.iid and event['timestamp'] == row.timestamp and (event['label'] == row.label)]
    for position in positions:
        expected = tuple((event['iid'] for event in events[:position] if event['label'] == 1 and event['iid'] != row.iid))
        if expected == row.his:
            return True
    return False

def restore_interactions(args: argparse.Namespace) -> None:
    started = time.time()
    profiles = {int(row['item_id']): row for row in read_jsonl(args.canonical_profile)}
    iid_to_asin = {iid: clean(row.get('asin')) for iid, row in profiles.items()}
    asin_to_iid = {asin: iid for iid, asin in iid_to_asin.items() if asin}
    rows, by_uid, split_counts = load_pkl_rows(args.data_dir)
    probes_by_uid = {uid: choose_probes(user_rows) for uid, user_rows in by_uid.items()}
    candidate_raw_by_uid: dict[int, set[str]] = defaultdict(set)
    first_pass_rows = 0
    if args.identity_map:
        exact_identity = {int(uid): clean(raw_uid) for uid, raw_uid in json.loads(args.identity_map.read_text(encoding='utf-8')).items()}
        if set(exact_identity) != set(by_uid) or len(set(exact_identity.values())) != len(exact_identity):
            raise RuntimeError('Exact identity map does not provide a one-to-one raw uid for every pkl uid')
        candidate_raw_by_uid.update({uid: {raw_uid} for uid, raw_uid in exact_identity.items()})
    else:
        signature_to_uids: dict[tuple[str, int, int], list[int]] = defaultdict(list)
        for uid, probes in probes_by_uid.items():
            for row in probes:
                signature_to_uids[probe_signature(row, iid_to_asin)].append(uid)
        for _, asin, raw_uid, rating, timestamp in iter_books_csv(args.raw_reviews):
            first_pass_rows += 1
            key = (asin, timestamp, int(rating >= 5))
            for uid in signature_to_uids.get(key, ()):
                candidate_raw_by_uid[uid].add(raw_uid)
    missing_candidates = sorted(set(by_uid) - set(candidate_raw_by_uid))
    if missing_candidates:
        raise RuntimeError(f'No raw candidates for {len(missing_candidates)} pkl users: {missing_candidates[:20]}')
    candidate_raw_uids = {raw_uid for values in candidate_raw_by_uid.values() for raw_uid in values}
    events_by_raw: dict[str, list[dict[str, Any]]] = defaultdict(list)
    second_pass_rows = 0
    for source_order, asin, raw_uid, rating, timestamp in iter_books_csv(args.raw_reviews):
        second_pass_rows += 1
        iid = asin_to_iid.get(asin)
        if iid is None or raw_uid not in candidate_raw_uids:
            continue
        events_by_raw[raw_uid].append({'iid': iid, 'rating': rating, 'label': int(rating >= 5), 'timestamp': timestamp, 'source_order': source_order})
    for events in events_by_raw.values():
        events.sort(key=lambda x: (x['timestamp'], x['source_order']))
    scored_by_uid: dict[int, list[tuple[str, int, int]]] = {}
    no_full_match: dict[int, list[tuple[str, int, int]]] = {}
    for uid in sorted(by_uid):
        scored = []
        for raw_uid in candidate_raw_by_uid[uid]:
            events = events_by_raw.get(raw_uid, [])
            probe_score = sum((event_matches_history(events, probe) for probe in probes_by_uid[uid]))
            if not probe_score:
                continue
            full_score = sum((event_matches_history(events, row) for row in by_uid[uid]))
            scored.append((raw_uid, full_score, probe_score))
        scored.sort(key=lambda x: (-x[1], -x[2], x[0]))
        exact = [entry for entry in scored if entry[1] == len(by_uid[uid])]
        if exact:
            scored_by_uid[uid] = exact
        else:
            no_full_match[uid] = scored[:10]
    if no_full_match:
        diagnostic = {'no_full_match_count': len(no_full_match), 'no_full_match_preview': {str(k): v for k, v in list(no_full_match.items())[:50]}}
        write_json(args.output_dir / 'raw_user_mapping_failure.json', diagnostic)
        raise RuntimeError(f'No exact full-row raw mapping for {len(no_full_match)} users')

    def event_sequence_hash(raw_uid: str) -> str:
        payload = [(event['iid'], event['rating'], event['timestamp']) for event in events_by_raw.get(raw_uid, [])]
        return hashlib.sha256(json.dumps(payload, separators=(',', ':')).encode('utf-8')).hexdigest()
    raw_by_uid: dict[int, str] = {}
    equivalent_tie_users: dict[int, list[str]] = {}
    used_raw: set[str] = set()
    ordered_uids = sorted(scored_by_uid, key=lambda uid: (len(scored_by_uid[uid]), uid))
    for uid in ordered_uids:
        available = [entry for entry in scored_by_uid[uid] if entry[0] not in used_raw]
        if not available:
            continue
        best_score = (available[0][1], available[0][2])
        tied = [entry[0] for entry in available if (entry[1], entry[2]) == best_score]
        if len(tied) > 1:
            sequence_hashes = {event_sequence_hash(raw_uid) for raw_uid in tied}
            if len(sequence_hashes) != 1:
                continue
            equivalent_tie_users[uid] = tied
        chosen_raw = sorted(tied)[0]
        raw_by_uid[uid] = chosen_raw
        used_raw.add(chosen_raw)
    unmatched = sorted(set(by_uid) - set(raw_by_uid))
    identity_diagnostic: dict[str, Any] = {'used': False}
    if unmatched and args.identity_reviews:
        identity_started = time.time()
        identity_first_selected_order: dict[str, int] = {}
        identity_2017_counts: Counter[str] = Counter()
        identity_rows = 0
        for source_order, asin, raw_uid, _, timestamp in iter_identity_reviews(args.identity_reviews):
            identity_rows += 1
            if raw_uid not in candidate_raw_uids or not YEAR_2017_START <= timestamp < YEAR_2018_START:
                continue
            identity_2017_counts[raw_uid] += 1
            if asin in asin_to_iid and raw_uid not in identity_first_selected_order:
                identity_first_selected_order[raw_uid] = source_order
        anchors = {uid: identity_first_selected_order[raw_uid] for uid, raw_uid in raw_by_uid.items() if uid not in equivalent_tie_users and raw_uid in identity_first_selected_order}
        ordered_anchors = sorted(anchors.items())
        anchor_violations = [{'left_uid': left_uid, 'left_order': left_order, 'right_uid': right_uid, 'right_order': right_order} for (left_uid, left_order), (right_uid, right_order) in zip(ordered_anchors, ordered_anchors[1:]) if left_order >= right_order]
        identity_diagnostic = {'used': True, 'source': str(args.identity_reviews), 'source_sha256': file_sha256(args.identity_reviews), 'rows_scanned': identity_rows, 'candidate_users_seen': len(identity_2017_counts), 'ordered_anchor_count': len(anchors), 'anchor_order_violation_count': len(anchor_violations), 'anchor_order_violations_preview': anchor_violations[:20], 'elapsed_sec': time.time() - identity_started}
        if anchor_violations:
            write_json(args.output_dir / 'raw_user_mapping_failure.json', identity_diagnostic)
            raise RuntimeError('Identity-source order disagrees with unambiguous pkl fingerprints')
        unresolved_details: dict[str, Any] = {}
        for uid in unmatched:
            lower = max((order for anchor_uid, order in anchors.items() if anchor_uid < uid), default=-1)
            upper = min((order for anchor_uid, order in anchors.items() if anchor_uid > uid), default=math.inf)
            candidates = []
            for raw_uid, full_score, probe_score in scored_by_uid[uid]:
                if raw_uid in used_raw:
                    continue
                first_order = identity_first_selected_order.get(raw_uid)
                candidates.append({'raw_uid': raw_uid, 'full_score': full_score, 'probe_score': probe_score, 'identity_2017_count': identity_2017_counts.get(raw_uid, 0), 'identity_first_selected_order': first_order, 'inside_anchor_interval': first_order is not None and lower < first_order < upper})
            eligible = [row for row in candidates if row['identity_2017_count'] > 20 and row['inside_anchor_interval']]
            unresolved_details[str(uid)] = {'lower_anchor_order': lower, 'upper_anchor_order': None if math.isinf(upper) else upper, 'candidates': candidates, 'eligible': eligible}
            if len(eligible) == 1:
                chosen_raw = eligible[0]['raw_uid']
                raw_by_uid[uid] = chosen_raw
                used_raw.add(chosen_raw)
                anchors[uid] = identity_first_selected_order[chosen_raw]
        identity_diagnostic['tie_resolution'] = unresolved_details
        identity_diagnostic['resolved_by_identity_order'] = len(set(unmatched) & set(raw_by_uid))
    unmatched = sorted(set(by_uid) - set(raw_by_uid))
    if unmatched:
        diagnostic = {'unmatched_count': len(unmatched), 'unmatched_preview': unmatched[:50], 'candidate_preview': {str(k): scored_by_uid.get(k, [])[:10] for k in unmatched[:20]}, 'identity_diagnostic': identity_diagnostic}
        write_json(args.output_dir / 'raw_user_mapping_failure.json', diagnostic)
        raise RuntimeError(f'Could not uniquely fingerprint {len(unmatched)} users; see mapping failure report')
    output_path = args.output_dir / 'all_interactions.restored_recent30.jsonl'
    mapping_path = args.output_dir / 'pkl_uid_to_raw_uid.json'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(mapping_path, {str(uid): raw_uid for uid, raw_uid in sorted(raw_by_uid.items())})
    stats: Counter[str] = Counter()
    history_lengths: list[int] = []
    with output_path.open('w', encoding='utf-8', newline='\n') as out:
        for uid in sorted(by_uid):
            events = events_by_raw[raw_by_uid[uid]]
            positions: dict[tuple[int, int, int], list[int]] = defaultdict(list)
            for index, event in enumerate(events):
                positions[event['iid'], event['timestamp'], event['label']].append(index)
            occurrences: Counter[tuple[int, int, int]] = Counter()
            assigned_positions: dict[int, str] = {}
            restored_rows: list[tuple[PklRow, int]] = []
            for row in by_uid[uid]:
                key = (row.iid, row.timestamp, row.label)
                occurrence = occurrences[key]
                matches = positions.get(key, [])
                if occurrence >= len(matches):
                    stats['target_event_missing'] += 1
                    continue
                position = matches[occurrence]
                occurrences[key] += 1
                restored_rows.append((row, position))
                assigned_positions[position] = row.split
            if len(restored_rows) != len(by_uid[uid]):
                continue
            for row, position in restored_rows:
                target = events[position]
                expected_positive = tuple((event['iid'] for event in events[:position] if event['label'] == 1 and event['iid'] != row.iid))
                if expected_positive != row.his:
                    stats['positive_history_mismatch'] += 1
                    continue
                recent: list[tuple[int, dict[str, Any]]] = []
                target_title_key = norm(profiles.get(row.iid, {}).get('title'))
                for event_index in range(position - 1, -1, -1):
                    event = events[event_index]
                    if event['iid'] == row.iid:
                        stats['excluded_target_iid_history_event'] += 1
                        continue
                    historical_title_key = norm(profiles.get(event['iid'], {}).get('title'))
                    if target_title_key and historical_title_key == target_title_key:
                        stats['excluded_target_title_equivalent_history_event'] += 1
                        continue
                    recent.append((event_index, event))
                    if len(recent) >= 30:
                        break
                recent.reverse()
                sample_id = f'book::{row.split}::row_{row.row_index}::u{uid}_i{row.iid}_t{row.timestamp}'
                record = {'sample_id': sample_id, 'split': row.split, 'row_index': row.row_index, 'source_index': row.row_index, 'user_id': uid, 'item_id': row.iid, 'label': row.label, 'rating': target['rating'], 'timestamp': row.timestamp, 'month_slot': int(pd.Timestamp(row.timestamp, unit='s', tz='UTC').month), 'flag': row.flag, 'not_cold': row.not_cold, 'history_item_ids': [event['iid'] for _, event in recent], 'his_label': [event['label'] for _, event in recent], 'history_ratings': [event['rating'] for _, event in recent], 'history_timestamps': [event['timestamp'] for _, event in recent], 'history_splits': [assigned_positions.get(index, 'raw_recovered') for index, _ in recent], 'thinkrec_uid': uid, 'thinkrec_iid': row.iid, 'thinkrec_positive_history_length': len(row.his)}
                append_jsonl(out, record)
                stats[f'written_{row.split}'] += 1
                stats[f'label_{row.label}'] += 1
                stats['history_positive'] += sum(record['his_label'])
                stats['history_negative'] += len(record['his_label']) - sum(record['his_label'])
                history_lengths.append(len(recent))
    expected_total = sum(split_counts.values())
    if sum((stats[f'written_{split}'] for split in split_counts)) != expected_total:
        raise RuntimeError(f"Restored row count mismatch: wrote={sum((stats[f'written_{s}'] for s in split_counts))} expected={expected_total}; stats={dict(stats)}")
    report = {'version': 'amazon_book_restored_interactions_recent30_v1', 'raw_reviews': str(args.raw_reviews), 'raw_reviews_sha256': file_sha256(args.raw_reviews), 'identity_map': str(args.identity_map) if args.identity_map else None, 'identity_map_sha256': file_sha256(args.identity_map) if args.identity_map else None, 'pkl_sha256': {split: file_sha256(args.data_dir / f'{split}_ood2.pkl') for split in split_counts}, 'split_counts_expected': split_counts, 'stats': dict(stats), 'pkl_users': len(by_uid), 'mapped_raw_users': len(raw_by_uid), 'candidate_raw_users': len(candidate_raw_uids), 'equivalent_sequence_tie_users': len(equivalent_tie_users), 'identity_resolution': identity_diagnostic, 'raw_rows_scanned_first_pass_2017': first_pass_rows, 'raw_rows_scanned_second_pass_2017': second_pass_rows, 'history_length_recent30': {'min': min(history_lengths), 'max': max(history_lengths), 'mean': sum(history_lengths) / len(history_lengths)}, 'output': str(output_path), 'elapsed_sec': time.time() - started}
    write_json(args.output_dir / 'restored_interactions_report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

def mean_vec(ids: list[int], embeddings: dict[int, np.ndarray]) -> np.ndarray | None:
    vectors = [embeddings[i] for i in ids if i in embeddings]
    if not vectors:
        return None
    vector = np.mean(np.stack(vectors, axis=0), axis=0)
    length = float(np.linalg.norm(vector))
    return vector / length if length > 0 else vector

def cosine_distance(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return 0.0
    return max(0.0, min(1.0, 1.0 - float(np.dot(left, right))))

def boundary_scores(row: dict[str, Any], embeddings: dict[int, np.ndarray]) -> dict[int, dict[str, float]]:
    ids = [int(x) for x in row['history_item_ids']]
    labels = [int(x) for x in row['his_label']]
    timestamps = [float(x) for x in row['history_timestamps']]
    log_gaps = [math.log1p(max(0.0, timestamps[i] - timestamps[i - 1])) for i in range(1, len(ids))]
    max_gap = max(log_gaps, default=0.0)
    out = {}
    for boundary in range(1, len(ids)):
        lo, hi = (max(0, boundary - 4), min(len(ids), boundary + 4))
        semantic = cosine_distance(mean_vec(ids[lo:boundary], embeddings), mean_vec(ids[boundary:hi], embeddings))
        left_pol = sum(labels[lo:boundary]) / max(1, boundary - lo)
        right_pol = sum(labels[boundary:hi]) / max(1, hi - boundary)
        polarity = abs(left_pol - right_pol)
        time_score = log_gaps[boundary - 1] / max_gap if max_gap > 0 else 0.0
        out[boundary] = {'score': 0.4 * semantic + 0.35 * polarity + 0.25 * time_score, 'semantic': semantic, 'polarity': polarity, 'time': time_score}
    return out

def valid_cuts(length: int, count: int) -> list[tuple[int, ...]]:
    if count == 1:
        return [()] if length <= 15 else []
    if count == 2:
        return [(cut,) for cut in range(4, length - 3) if all((4 <= size <= 15 for size in (cut, length - cut)))]
    return [(first, second) for first in range(4, length - 7) for second in range(first + 4, length - 3) if all((4 <= size <= 15 for size in (first, second - first, length - second)))]

def build_segments(row: dict[str, Any], embeddings: dict[int, np.ndarray]) -> tuple[list[dict[str, Any]], dict[int, dict[str, float]]]:
    length = len(row['history_item_ids'])
    if not length:
        return ([], {})
    scores = boundary_scores(row, embeddings)
    if length <= 8:
        chosen: tuple[int, ...] = ()
    else:
        best: dict[int, tuple[tuple[int, ...], float]] = {}
        for count in (1, 2, 3):
            candidates = valid_cuts(length, count)
            if candidates:
                best[count] = max(((cuts, sum((scores[cut]['score'] for cut in cuts)) / len(cuts)) if cuts else (cuts, 0.0) for cuts in candidates), key=lambda value: (value[1], tuple((-cut for cut in value[0]))))
        minimum = 1 if length <= 15 else 2
        if 3 in best and best[3][1] >= 0.66:
            chosen = best[3][0]
        elif 2 in best and (minimum == 2 or best[2][1] >= 0.36):
            chosen = best[2][0]
        elif minimum == 1:
            chosen = ()
        else:
            chosen = best[2][0]
    ends = list(chosen) + [length]
    segments = []
    start = 0
    for number, end in enumerate(ends, 1):
        position = 'only' if len(ends) == 1 else 'first' if number == 1 else 'last' if number == len(ends) else 'middle'
        segments.append({'segment_id': f'S{number}', 'start': start, 'end': end, 'position': position})
        start = end
    return (segments, scores)

def compact_categories(item: dict[str, Any]) -> list[str]:
    categories: list[str] = []
    for value in as_list(item.get('categories')) + as_list(item.get('surface_genres')):
        text = clean(value)
        key = norm(text)
        if not key or key in BOOK_CATEGORY_ROOTS:
            continue
        tokens = set(key.split())
        if any((tokens == set(norm(existing).split()) or tokens < set(norm(existing).split()) or set(norm(existing).split()) < tokens for existing in categories)):
            continue
        categories.append(text)
        if len(categories) >= 2:
            break
    return categories

def semantic_evidence(item: dict[str, Any]) -> str:
    status = clean(item.get('content_evidence_status')).casefold()
    if status == 'adequate':
        return 'adequate'
    if status == 'external_kg_supported':
        return 'external_supported'
    return 'weak'

def item_terms(item: dict[str, Any]) -> list[str]:
    values = compact_categories(item) + as_list(item.get('semantic_signal_terms'))[:10]
    return [term for value in values if (term := norm(value)) and term not in BOOK_CATEGORY_ROOTS]

def common_terms(ids: list[int], labels: list[int], items: dict[int, dict[str, Any]], positive: bool) -> list[tuple[str, int, int, int, int, str]]:
    own = [iid for iid, label in zip(ids, labels) if (label == 1) == positive]
    other = [iid for iid, label in zip(ids, labels) if (label == 1) != positive]
    counts: Counter[str] = Counter()
    other_counts: Counter[str] = Counter()
    for iid in own:
        counts.update(sorted(set(item_terms(items.get(iid, {})))))
    for iid in other:
        other_counts.update(sorted(set(item_terms(items.get(iid, {})))))
    ordered = sorted(counts.items(), key=lambda pair: (-pair[1], other_counts[pair[0]], pair[0]))
    out = []
    for term, count in ordered[:8]:
        if count < 2 and len(own) > 3:
            continue
        role = 'positive_discriminative' if positive and other_counts[term] == 0 else 'negative_rejection_signal' if not positive and other_counts[term] == 0 else 'shared_non_discriminative'
        out.append((term, count, len(own), other_counts[term], len(other), role))
    return out[:6]

def state_hint(labels: list[int]) -> str:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives >= 2 and negatives <= 1:
        return 'positive_preference_state'
    if negatives >= 2 and positives <= 1:
        return 'negative_rejection_state'
    return 'mixed_feedback_state'

def without_target_phrase(values: Iterable[Any], target_title_key: str, limit: int) -> list[str]:
    kept = []
    for value in values:
        text = clean(value)
        if text and (not target_title_key or target_title_key not in norm(text)):
            kept.append(text)
    if target_title_key:
        removed = True
        while removed and target_title_key in norm(', '.join(kept)):
            removed = False
            for width in range(2, len(kept) + 1):
                for start in range(0, len(kept) - width + 1):
                    if target_title_key in norm(' '.join(kept[start:start + width])):
                        del kept[start:start + width]
                        removed = True
                        break
                if removed:
                    break
    return kept[:limit]

def render_item(iid: int, order: int, rating: float, timestamp: int, item: dict[str, Any], target_title_key: str) -> str:
    title = clean(item.get('title')) or 'unknown'
    if target_title_key and target_title_key in norm(title):
        title = 'none'
    authors = without_target_phrase([clean_author(value) for value in as_list(item.get('authors'))], target_title_key, 1)
    categories = without_target_phrase(compact_categories(item), target_title_key, 2)
    semantic_terms = without_target_phrase(as_list(item.get('semantic_signal_terms')), target_title_key, 10)
    return '\n'.join([f'- title: {title}', f'  rating: {rating:g}', f'  order: {order}', f"  categories: {', '.join(categories) or 'none'}.", f"  semantic_terms: {', '.join(semantic_terms) or 'none'}.", f"  authors: {', '.join(authors) or 'none'}", f'  semantic_evidence: {semantic_evidence(item)}'])

def render_pack(row: dict[str, Any], segment: dict[str, Any], items: dict[int, dict[str, Any]], target_title: str) -> str:
    start, end = (segment['start'], segment['end'])
    ids = [int(x) for x in row['history_item_ids'][start:end]]
    labels = [int(x) for x in row['his_label'][start:end]]
    ratings = [float(x) for x in row['history_ratings'][start:end]]
    timestamps = [int(x) for x in row['history_timestamps'][start:end]]
    target_title_key = norm(target_title)
    positive = [(iid, rating, timestamp) for iid, label, rating, timestamp in zip(ids, labels, ratings, timestamps) if label == 1]
    negative = [(iid, rating, timestamp) for iid, label, rating, timestamp in zip(ids, labels, ratings, timestamps) if label == 0]
    lines = ['[SEGMENT CONTEXT]', f"segment_id: {segment['segment_id']}", f"segment_position: {segment['position']}", f'segment_length: {len(ids)}', f'positive_count: {len(positive)}', f'negative_count: {len(negative)}', f'feedback_state_hint: {state_hint(labels)}', '', '[POSITIVE INTERACTIONS]']
    if positive:
        lines.extend((render_item(iid, order, rating, timestamp, items.get(iid, {}), target_title_key) for order, (iid, rating, timestamp) in enumerate(positive, 1)))
    else:
        lines.append('none')
    lines.extend(['', '[NEGATIVE INTERACTIONS]'])
    if negative:
        lines.extend((render_item(iid, order, rating, timestamp, items.get(iid, {}), target_title_key) for order, (iid, rating, timestamp) in enumerate(negative, 1)))
    else:
        lines.append('none')
    lines.extend(['', '[POSITIVE COMMON TERMS]'])
    positive_terms = [row for row in common_terms(ids, labels, items, True) if not target_title_key or target_title_key not in norm(row[0])]
    lines.extend((f'- {term} | support={count}/{denominator} | negative_overlap={other_count}/{other_denominator} | role={role}' for term, count, denominator, other_count, other_denominator, role in positive_terms))
    if not positive_terms:
        lines.append('none')
    lines.extend(['', '[NEGATIVE COMMON TERMS]'])
    negative_terms = [row for row in common_terms(ids, labels, items, False) if not target_title_key or target_title_key not in norm(row[0])]
    lines.extend((f'- {term} | support={count}/{denominator} | positive_overlap={other_count}/{other_denominator} | role={role}' for term, count, denominator, other_count, other_denominator, role in negative_terms))
    if not negative_terms:
        lines.append('none')
    return '\n'.join(lines)

def load_embeddings(directory: Path) -> dict[int, np.ndarray]:
    array = np.load(directory / 'item_embeddings.npy', mmap_mode='r')
    ids = json.loads((directory / 'item_ids.json').read_text(encoding='utf-8'))
    if len(ids) != len(array):
        raise RuntimeError(f'embedding ids/array mismatch: ids={len(ids)} rows={len(array)}')
    return {int(iid): array[index] for index, iid in enumerate(ids)}

def stable_sample_key(sample_id: str, seed: int) -> str:
    return hashlib.sha256(f'{seed}:{sample_id}'.encode('utf-8')).hexdigest()

def normalize_title(value: Any) -> str:
    return re.sub('[^a-z0-9]+', ' ', clean(value).casefold()).strip()

def encoded_token_count(encoded: Any) -> int:
    if isinstance(encoded, dict):
        encoded = encoded.get('input_ids')
    elif hasattr(encoded, 'input_ids'):
        encoded = encoded.input_ids
    if encoded is None:
        raise RuntimeError('Tokenizer output does not contain input_ids')
    if hasattr(encoded, 'shape'):
        return int(encoded.shape[-1])
    if encoded and isinstance(encoded[0], (list, tuple)):
        return len(encoded[0])
    return len(encoded)

def prepare_teacher(args: argparse.Namespace) -> None:
    started = time.time()
    items = {int(row['item_id']): row for row in read_jsonl(args.canonical_profile)}
    embeddings = load_embeddings(args.embedding_dir)
    missing_embeddings = sorted(set(items) - set(embeddings))
    if missing_embeddings:
        raise RuntimeError(f'Missing embeddings for {len(missing_embeddings)} items: {missing_embeddings[:20]}')
    eligible = [row for row in read_jsonl(args.all_interactions) if row.get('split') == 'train' and len(row.get('history_item_ids') or []) >= args.min_history_len]
    eligible.sort(key=lambda row: stable_sample_key(str(row['sample_id']), args.seed))
    selected = eligible[:args.teacher_samples]
    if len(selected) != args.teacher_samples:
        raise RuntimeError(f'Only {len(selected)} eligible rows for requested {args.teacher_samples}')
    template = args.prompt.read_text(encoding='utf-8')
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.output_dir / 'teacher_sample_list.jsonl'
    segment_path = args.output_dir / 'state_segments.jsonl'
    pack_path = args.output_dir / 'state_packs.jsonl'
    message_path = args.output_dir / 'state_messages.jsonl'
    smoke_message_path = args.output_dir / f'state_messages_v3_smoke{args.smoke_samples}.jsonl'
    segment_distribution: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    lens_counts: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    title_overlap_warnings: list[dict[str, Any]] = []
    pack_lengths: list[int] = []
    prompt_token_lengths: list[int] = []
    with sample_path.open('w', encoding='utf-8', newline='\n') as sample_out, segment_path.open('w', encoding='utf-8', newline='\n') as segment_out, pack_path.open('w', encoding='utf-8', newline='\n') as pack_out, message_path.open('w', encoding='utf-8', newline='\n') as message_out, smoke_message_path.open('w', encoding='utf-8', newline='\n') as smoke_message_out:
        for sample_index, row in enumerate(selected):
            target_id = int(row['item_id'])
            target = items[target_id]
            history_ids = [int(x) for x in row['history_item_ids']]
            if target_id in history_ids:
                errors.append({'sample_id': row['sample_id'], 'type': 'target_item_id_in_history'})
                continue
            segments, scores = build_segments(row, embeddings)
            segment_distribution[str(len(segments))] += 1
            label_counts[str(int(row['label']))] += 1
            append_jsonl(sample_out, {'sample_id': row['sample_id'], 'user_id': int(row['user_id']), 'target_item_id': target_id, 'label': int(row['label']), 'rating': float(row['rating']), 'timestamp': int(row['timestamp']), 'history_len': len(history_ids), 'num_segments': len(segments), 'selection_key': stable_sample_key(str(row['sample_id']), args.seed)})
            target_title = normalize_title(target.get('title'))
            for segment in segments:
                start, end = (segment['start'], segment['end'])
                segment_ids = history_ids[start:end]
                pack = render_pack(row, segment, items, clean(target.get('title')))
                pack_hash = hashlib.sha256(pack.encode('utf-8')).hexdigest()
                pack_lengths.append(len(pack))
                if target_id in segment_ids:
                    errors.append({'sample_id': row['sample_id'], 'segment_id': segment['segment_id'], 'type': 'target_item_id_in_segment'})
                exact_title_item_ids = [iid for iid in segment_ids if target_title and normalize_title(items.get(iid, {}).get('title')) == target_title]
                if exact_title_item_ids:
                    errors.append({'sample_id': row['sample_id'], 'segment_id': segment['segment_id'], 'type': 'target_title_exact_in_historical_title_field', 'historical_item_ids': exact_title_item_ids, 'target_title': target.get('title')})
                if clean(target.get('asin')) and clean(target.get('asin')) in pack:
                    errors.append({'sample_id': row['sample_id'], 'segment_id': segment['segment_id'], 'type': 'target_asin_in_pack'})
                if any((marker in pack for marker in ('target_iid:', 'target_item_id:', 'target_label:', 'sample_id:'))):
                    errors.append({'sample_id': row['sample_id'], 'segment_id': segment['segment_id'], 'type': 'target_control_marker_in_pack'})
                if target_title and len(target_title) >= 8 and (target_title in normalize_title(pack)):
                    errors.append({'sample_id': row['sample_id'], 'segment_id': segment['segment_id'], 'target_item_id': target_id, 'target_title': target.get('title'), 'type': 'target_title_phrase_in_pack_after_sanitization'})
                append_jsonl(segment_out, {'sample_id': row['sample_id'], 'split': 'train', 'user_id': int(row['user_id']), 'segment_id': segment['segment_id'], 'segment_position': segment['position'], 'start': start, 'end': end, 'item_ids': segment_ids, 'boundary_scores': scores})
                append_jsonl(pack_out, {'sample_id': row['sample_id'], 'split': 'train', 'user_id': int(row['user_id']), 'segment_id': segment['segment_id'], 'pack_hash': pack_hash, 'pack_text': pack})
                for candidate_id, lens, instruction in LENSES:
                    system_prompt = template.replace('{{candidate_id}}', candidate_id).replace('{{analysis_lens}}', lens).replace('{{lens_instruction}}', instruction)
                    messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': pack}]
                    prompt_tokens = None
                    if tokenizer is not None:
                        token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                        prompt_tokens = encoded_token_count(token_ids)
                        prompt_token_lengths.append(prompt_tokens)
                        if prompt_tokens + args.max_output_tokens > args.max_model_len:
                            errors.append({'sample_id': row['sample_id'], 'segment_id': segment['segment_id'], 'candidate_id': candidate_id, 'type': 'context_length_overflow', 'prompt_tokens': prompt_tokens, 'max_output_tokens': args.max_output_tokens, 'max_model_len': args.max_model_len})
                    for message in messages:
                        content = message['content']
                        if any((marker in content for marker in ('target_iid:', 'target_item_id:', 'target_label:', 'sample_id:'))):
                            errors.append({'sample_id': row['sample_id'], 'segment_id': segment['segment_id'], 'candidate_id': candidate_id, 'type': 'target_control_marker_in_message'})
                    message_record = {'record_id': f"{row['sample_id']}::{segment['segment_id']}::{candidate_id}", 'sample_id': row['sample_id'], 'split': 'train', 'user_id': int(row['user_id']), 'item_id': target_id, 'segment_id': segment['segment_id'], 'candidate_id': candidate_id, 'analysis_lens': lens, 'pack_hash': pack_hash, 'prompt_tokens': prompt_tokens, 'messages': messages}
                    append_jsonl(message_out, message_record)
                    if sample_index < args.smoke_samples:
                        append_jsonl(smoke_message_out, message_record)
                    lens_counts[candidate_id] += 1
    leakage = {'audit_version': 'book_state_teacher_field_level_leakage_v3_book_compatible_pack', 'hard_error_count': len(errors), 'hard_error_type_counts': dict(Counter((error['type'] for error in errors))), 'hard_errors_preview': errors[:100], 'title_overlap_warning_count': len(title_overlap_warnings), 'title_overlap_warnings_preview': title_overlap_warnings[:100]}
    write_json(args.output_dir / 'leakage_audit.json', leakage)
    if errors:
        raise RuntimeError(f'Teacher input leakage audit failed with {len(errors)} hard errors')
    token_length_report = None
    if prompt_token_lengths:
        token_length_report = {'min': min(prompt_token_lengths), 'p50': float(np.percentile(prompt_token_lengths, 50)), 'p95': float(np.percentile(prompt_token_lengths, 95)), 'p99': float(np.percentile(prompt_token_lengths, 99)), 'max': max(prompt_token_lengths), 'max_model_len': args.max_model_len, 'reserved_output_tokens': args.max_output_tokens, 'max_allowed_prompt_tokens': args.max_model_len - args.max_output_tokens}
    inference_config = {'version': 'amazon_book_state_72b_inference_config_v3_book_compatible_pack', 'model_path': str(args.tokenizer) if args.tokenizer else 'Qwen/Qwen2.5-72B-Instruct-AWQ', 'served_model_name': 'Qwen2.5-72B-Instruct-AWQ', 'endpoint': 'http://127.0.0.1:8010/v1', 'temperature': 0.0, 'max_output_tokens': args.max_output_tokens, 'max_model_len': args.max_model_len, 'logprobs': True, 'top_logprobs': 5, 'max_attempts': 2, 'workers': 10, 'json_mode': False, 'smoke_messages': str(smoke_message_path), 'full_messages': str(message_path)}
    write_json(args.output_dir / 'state_72b_inference_config.json', inference_config)
    report = {'version': 'amazon_book_state_teacher_inputs_v3_book_compatible_pack', 'teacher_samples': len(selected), 'eligible_train_rows': len(eligible), 'selection': {'seed': args.seed}, 'label_counts': dict(label_counts), 'segment_count_distribution': dict(segment_distribution), 'segments': sum((segment_distribution[str(i)] * i for i in range(1, 4))), 'messages': sum(lens_counts.values()), 'lens_counts': dict(lens_counts), 'pack_chars': {'min': min(pack_lengths), 'max': max(pack_lengths), 'mean': sum(pack_lengths) / len(pack_lengths)}, 'prompt_tokens': token_length_report, 'segmentation': {'history': 'target-preceding recent 30', 'boundary': '0.40 embedding cosine distance + 0.35 polarity difference + 0.25 normalized log time gap', 'window': 4, 'max_segments': 3, 'segment_length': [4, 15], 'two_segment_threshold': 0.36, 'three_segment_threshold': 0.66}, 'inputs': {'all_interactions': str(args.all_interactions), 'all_interactions_sha256': file_sha256(args.all_interactions), 'canonical_profile': str(args.canonical_profile), 'canonical_profile_sha256': file_sha256(args.canonical_profile), 'embedding_dir': str(args.embedding_dir), 'embedding_meta_sha256': file_sha256(args.embedding_dir / 'embedding_meta.json'), 'prompt': str(args.prompt), 'prompt_sha256': file_sha256(args.prompt)}, 'outputs': {'sample_list': str(sample_path), 'segments': str(segment_path), 'packs': str(pack_path), 'messages': str(message_path), 'smoke_messages': str(smoke_message_path), 'inference_config': str(args.output_dir / 'state_72b_inference_config.json')}, 'leakage_audit': leakage, 'elapsed_sec': time.time() - started}
    write_json(args.output_dir / 'teacher_input_report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    profiles = sub.add_parser('profiles')
    profiles.add_argument('--source-profile', type=Path, required=True)
    profiles.add_argument('--merged-profile', type=Path, required=True)
    profiles.add_argument('--output-dir', type=Path, required=True)
    profiles.set_defaults(func=prepare_profiles)
    restore = sub.add_parser('restore')
    restore.add_argument('--data-dir', type=Path, required=True)
    restore.add_argument('--canonical-profile', type=Path, required=True)
    restore.add_argument('--raw-reviews', type=Path, required=True)
    restore.add_argument('--identity-map', type=Path)
    restore.add_argument('--identity-reviews', type=Path)
    restore.add_argument('--output-dir', type=Path, required=True)
    restore.set_defaults(func=restore_interactions)
    return parser

def main() -> None:
    args = build_parser().parse_args()
    args.func(args)
if __name__ == '__main__':
    main()
