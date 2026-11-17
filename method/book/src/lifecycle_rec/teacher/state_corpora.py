import argparse
import gzip
import hashlib
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
STATE_SYSTEM_PROMPT = 'You analyze one short segment of book interactions and generate one compact interest state node. Use only the supplied pack, without external knowledge, unstated plots, or permanent preference claims.\n\nRatings >=5 are positive; ratings <5 are the binary negative class. A 4-star rating is not a poor rating. The negative class supports only tentative local caution, never an explanation of why a reader rated a book that way.\n\nRules:\n1. Copy feedback_state_hint into state_type. This is a compatibility enum; negative_rejection_state does not authorize rejection language in prose. order is the original position within this segment.\n2. Separate positive and negative evidence. For absent polarity, use an empty pattern list, reasoning "No positive interactions are present." or "No negative interactions are present.", and match cues ["none"].\n3. Prefer semantic_terms; categories provide context. For weak items, use only supplied categories or concrete supplied phrases. Never expand a title fragment into a theme. Fewer useful terms are better than padding. An item with no usable semantic evidence need not support a pattern.\n4. Each polarity permits at most TWO compact patterns. Each pattern cites exactly ONE verbatim book title. Every support_terms entry must be an exact complete element of categories or semantic_terms of at least one cited book, and EVERY cited book must supply at least one support term. Do not use uncited books to justify the pattern. Preserve punctuation and spelling.\n5. Use 1-4 support_terms and 1-4 match cues, or fewer when evidence is limited. Match cues must be exact complete terms from any supplied book of the same polarity; they need not repeat the selected pattern support_terms. Do not fill quotas. Never output "none" as a support term.\n6. COMMON TERMS are counts, not causal evidence. A term shared across polarities does not establish a contrast. Avoid making shared broad genres the principal explanation. If all evidence is shared, explicitly state that no distinct semantic contrast is supported and use match cues ["none"]. Terms seen once support one local observation, not a recurring pattern.\n7. A single book supports an observation about that book only; write "one positive interaction" or "one below-5 interaction". Do not infer a general preference, persistent tendency, or author preference from it.\n8. In prose use "local positive signal" and "tentative local caution". Do not use dislike/disliked/disliking, hate/hated, avoid/avoided/avoidance, reject/rejected/rejection, strong preference, rated poorly, low-rated, poorly rated, or permanent claims. This restriction excludes schema keys, exact book titles, and copied semantic phrases. A suitable negative statement is: "These below-5 interactions provide tentative local caution; the reason for the ratings is unknown."\n9. Author cues require the same author in at least two books of the same polarity. Variants that differ only in case, spaces, periods, or hyphens may be treated as aliases, but output one exact supplied spelling. Otherwise ["none"]. Do not output director or actor fields for books.\n10. semantic_evidence is confidence metadata. Never mention weak, adequate, external_supported, evidence quality, or metadata status in output prose.\n11. Keep the state label brief, summary at most two sentences, and each reasoning at most one sentence. Ground prose in the cited books and support terms. Do not infer a contrast from edition/volume numbering, names, promotional praise, or generic series markers.\n12. Return ONLY JSON in the schema below. Generate exactly one proposed state node; do not add candidate metadata, analysis commentary, or markdown.\n\n{\n  "interest_state_node": {\n    "state_type": "copy feedback_state_hint",\n    "state_label": "...",\n    "state_summary": "...",\n    "positive_analysis": {\n      "reasoning": "...",\n      "liked_patterns": [{"pattern": "...", "evidence_titles": ["exact title"], "support_terms": ["exact term"]}]\n    },\n    "negative_analysis": {\n      "reasoning": "...",\n      "rejected_or_risky_patterns": [{"pattern": "...", "evidence_titles": ["exact title"], "support_terms": ["exact term"]}]\n    },\n    "decision_cues": {\n      "positive_match_cues": ["exact support term or none"],\n      "negative_match_cues": ["exact support term or none"],\n      "author_cues": {"liked_authors": ["none"], "rejected_authors": ["none"]}\n    }\n  }\n}\n\nUse local positive signal, not strong/clear/persistent positive preference. No claim of preference intensity or permanence is justified by these observations. Remove equivalent duplicate phrases before returning JSON.'

def read_jsonl(path):
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(str(path) + '.tmp')
    with open(temp, 'w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(canonical(row) + '\n')
    temp.replace(path)

def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(str(path) + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temp.replace(path)

def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def digest(obj):
    text = obj if isinstance(obj, str) else canonical(obj)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def segnum(value):
    return int(re.sub('\\D', '', str(value)) or 0)

def file_digest(path):
    result = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()

def state_node(candidate):
    source = (candidate.get('model_output') or {}).get('candidate') or candidate.get('selected_candidate') or {}
    node = json.loads(canonical({key: value for key, value in source.items() if key not in {'candidate_id', 'analysis_lens', 'candidate_status', 'insufficient_reason'}}))
    decision_cues = node.get('decision_cues') or {}
    decision_cues.pop('director_actor_cues', None)
    return {'interest_state_node': node}

def validate_node(output):
    node = output.get('interest_state_node')
    required = {'state_type', 'state_label', 'state_summary', 'positive_analysis', 'negative_analysis', 'decision_cues'}
    if not isinstance(node, dict) or set(node) != required:
        return 'bad_top_level_schema'
    if set(node['positive_analysis']) != {'reasoning', 'liked_patterns'}:
        return 'bad_positive_schema'
    if set(node['negative_analysis']) != {'reasoning', 'rejected_or_risky_patterns'}:
        return 'bad_negative_schema'
    cues = node['decision_cues']
    if set(cues) != {'positive_match_cues', 'negative_match_cues', 'author_cues'}:
        return 'bad_decision_cues_schema'
    if set(cues['author_cues']) != {'liked_authors', 'rejected_authors'}:
        return 'bad_author_cues_schema'

    def all_keys(value):
        if isinstance(value, dict):
            for key, child in value.items():
                yield str(key).lower()
                yield from all_keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from all_keys(child)
    forbidden_keys = {'director_actor_cues', 'liked_actors', 'liked_directors', 'rejected_actors', 'rejected_directors', 'actors', 'directors'}
    if forbidden_keys & set(all_keys(output)):
        return 'movie_field_leakage'
    return None

def final_sample_map():
    final = {}
    baseline_rows = list(read_jsonl(BASELINE))
    for row in baseline_rows:
        if row.get('prediction') == row.get('label'):
            final[row['sample_id']] = {'status': 'base_correct', 'label': row['label'], 'prediction': row['prediction']}
    for row in read_jsonl(EDGE_DIR / 'repair_final_results.jsonl'):
        if row.get('final_correct') is True:
            final[row['sample_id']] = {'status': 'edge_repaired_success', 'label': row['label'], 'prediction': row['final_prediction']}
    for row in read_jsonl(STATE_DIR / 'state_repair_final_results.jsonl'):
        if row.get('final_correct') is True:
            final[row['sample_id']] = {'status': 'state_repaired_success', 'label': row['label'], 'prediction': row['final_prediction']}
    return (baseline_rows, final)

def split_records(records, seed=2026):
    keys = sorted({row['split_group_key'] for row in records})
    random.Random(seed).shuffle(keys)
    train_end = int(len(keys) * 0.8)
    valid_end = train_end + int(len(keys) * 0.1)
    assignment = {key: 'train' if idx < train_end else 'valid' if idx < valid_end else 'test' for idx, key in enumerate(keys)}
    splits = {'train': [], 'valid': [], 'test': []}
    for row in records:
        copied = dict(row)
        copied['split'] = assignment[row['split_group_key']]
        splits[copied['split']].append(copied)
    return splits

def main():
    global BASELINE, EDGE_DIR, STATE_DIR, C1_SELECTED, CANDIDATES, MESSAGES
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--transition-repair-dir', type=Path, required=True)
    parser.add_argument('--state-repair-dir', type=Path, required=True)
    parser.add_argument('--selected-states', type=Path, required=True)
    parser.add_argument('--state-candidates', type=Path, required=True)
    parser.add_argument('--state-messages', type=Path, required=True)
    args = parser.parse_args()
    BASELINE = args.baseline
    EDGE_DIR = args.transition_repair_dir
    STATE_DIR = args.state_repair_dir
    C1_SELECTED = args.selected_states
    CANDIDATES = args.state_candidates
    MESSAGES = args.state_messages
    out_dir = args.output_dir
    print('stage=load_final_samples', flush=True)
    baseline_rows, final_samples = final_sample_map()
    if len(baseline_rows) != 6000 or len(final_samples) != 4142:
        raise RuntimeError(f'unexpected sample totals: baseline={len(baseline_rows)} final={len(final_samples)}')
    print('stage=load_state_repairs', flush=True)
    state_success = {}
    for row in read_jsonl(STATE_DIR / 'state_repair_attempts.jsonl'):
        if row.get('success') is True:
            if row['sample_id'] in state_success:
                raise RuntimeError(f"multiple successful state repairs: {row['sample_id']}")
            state_success[row['sample_id']] = row
    if len(state_success) != 253:
        raise RuntimeError(f'unexpected successful state repairs: {len(state_success)}')
    print('stage=load_c1_index', flush=True)
    c1_rows = {}
    segments_by_sample = defaultdict(list)
    for row in read_jsonl(C1_SELECTED):
        sid = row['sample_id']
        if sid not in final_samples:
            continue
        key = (sid, row['segment_id'], 'C1')
        c1_rows[key] = {'sample_id': row['sample_id'], 'segment_id': row['segment_id'], 'selected_candidate': row['selected_candidate'], 'source_pack_hash': row['source_pack_hash']}
        segments_by_sample[sid].append(row['segment_id'])
    selected_cids = {}
    for sid in final_samples:
        for segment_id in segments_by_sample.get(sid, []):
            selected_cids[sid, segment_id] = 'C1'
        if final_samples[sid]['status'] == 'state_repaired_success':
            action = state_success.get(sid)
            if not action:
                raise RuntimeError(f'missing successful state action: {sid}')
            selected_cids[sid, action['target_node']] = action['new_candidate_id']
    missing_samples = sorted(set(final_samples) - set(segments_by_sample))
    if missing_samples:
        raise RuntimeError(f'final samples without segments: {len(missing_samples)}')
    needed_alternatives = {(sid, segment_id, cid) for (sid, segment_id), cid in selected_cids.items() if cid != 'C1'}
    print(f'stage=scan_alternatives needed={len(needed_alternatives)}', flush=True)
    alternative_rows = {}
    with gzip.open(CANDIDATES, 'rt', encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            key = (row['sample_id'], row['segment_id'], row['candidate_id'])
            if key in needed_alternatives:
                alternative_rows[key] = {'sample_id': row['sample_id'], 'segment_id': row['segment_id'], 'candidate_id': row['candidate_id'], 'pack_hash': row['pack_hash'], 'model_output': row['model_output']}
    if set(alternative_rows) != needed_alternatives:
        raise RuntimeError(f'missing alternative candidates: {len(needed_alternatives - set(alternative_rows))}')
    print('stage=validate_selected_nodes', flush=True)
    selected = {}
    for (sid, segment_id), cid in selected_cids.items():
        key = (sid, segment_id, cid)
        source = c1_rows.get(key) if cid == 'C1' else alternative_rows.get(key)
        if not source:
            raise RuntimeError(f'missing selected candidate: {key}')
        output = state_node(source)
        error = validate_node(output)
        if error:
            raise RuntimeError(f'invalid selected node {key}: {error}')
        selected[key] = (source, output)
    needed_message_keys = set(selected)
    print(f'stage=scan_messages needed={len(needed_message_keys)}', flush=True)
    message_rows = {}
    with open(MESSAGES, encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            key = (row['sample_id'], row['segment_id'], row['candidate_id'])
            if key in needed_message_keys:
                message_rows[key] = {'sample_id': row['sample_id'], 'segment_id': row['segment_id'], 'candidate_id': row['candidate_id'], 'user_id': row.get('user_id'), 'item_id': row.get('item_id'), 'pack_hash': row['pack_hash'], 'user_text': row['messages'][1]['content']}
    if set(message_rows) != needed_message_keys:
        raise RuntimeError(f'missing exact messages: {len(needed_message_keys - set(message_rows))}')
    print('stage=assemble_records', flush=True)
    records = []
    selected_rows = []
    conflicts = []
    removed_movie_compatibility_fields = 0
    for key in sorted(selected, key=lambda x: (x[0], segnum(x[1]), x[2])):
        sid, segment_id, cid = key
        source, output = selected[key]
        raw_candidate = (source.get('model_output') or {}).get('candidate') or source.get('selected_candidate') or {}
        movie_cues = (raw_candidate.get('decision_cues') or {}).get('director_actor_cues')
        if movie_cues is not None:
            movie_values = [value for values in movie_cues.values() for value in values]
            if any((str(value).lower() != 'none' for value in movie_values)):
                conflicts.append({'sample_id': sid, 'segment_id': segment_id, 'candidate_id': cid, 'reason': 'nonempty_movie_field'})
                continue
            removed_movie_compatibility_fields += 1
        message = message_rows[key]
        source_pack_hash = source.get('source_pack_hash') if cid == 'C1' else source.get('pack_hash')
        if message.get('pack_hash') != source_pack_hash:
            conflicts.append({'sample_id': sid, 'segment_id': segment_id, 'candidate_id': cid, 'reason': 'pack_hash_mismatch', 'message_pack_hash': message.get('pack_hash'), 'candidate_pack_hash': source_pack_hash})
            continue
        user_text = message['user_text']
        if not user_text.startswith('[SEGMENT CONTEXT]'):
            conflicts.append({'sample_id': sid, 'segment_id': segment_id, 'candidate_id': cid, 'reason': 'bad_user_pack'})
            continue
        if '[TARGET' in user_text.upper() or 'target item' in user_text.lower():
            conflicts.append({'sample_id': sid, 'segment_id': segment_id, 'candidate_id': cid, 'reason': 'target_leakage_marker'})
            continue
        input_hash = digest(user_text)
        output_hash = digest(output)
        final = final_samples[sid]
        record_id = f'{sid}::{segment_id}::{cid}::{input_hash[:12]}'
        records.append({'record_id': record_id, 'sample_id': sid, 'user_id': message.get('user_id'), 'item_id': message.get('item_id'), 'segment_id': segment_id, 'split_group_key': message['pack_hash'], 'source_label': final['label'], 'teacher_recommendation_pred': final['prediction'], 'pack_hash': message['pack_hash'], 'input_hash': input_hash, 'output_hash': output_hash, 'messages': [{'role': 'system', 'content': STATE_SYSTEM_PROMPT}, {'role': 'user', 'content': user_text}, {'role': 'assistant', 'content': canonical(output)}], 'quality': {'final_source_status': final['status'], 'selected_candidate_id': cid, 'sample_weight': 1.0}, 'provenance': {'teacher_model': 'Qwen2.5-72B-Instruct-AWQ', 'repair_source': final['status'], 'source_message_file': str(MESSAGES), 'source_candidate_file': str(C1_SELECTED if cid == 'C1' else CANDIDATES)}})
        selected_rows.append({'record_id': record_id, 'sample_id': sid, 'user_id': message.get('user_id'), 'item_id': message.get('item_id'), 'segment_id': segment_id, 'selected_candidate_id': cid, 'source_pack_hash': message['pack_hash'], 'final_source_status': final['status'], 'selected_state_node': output['interest_state_node']})
    if conflicts:
        write_jsonl(out_dir / 'segment_interest_node_sft_conflicts.jsonl', conflicts)
        raise RuntimeError(f'corpus conflicts detected: {len(conflicts)}')
    expected_records = sum((len(value) for value in segments_by_sample.values()))
    if len(records) != expected_records or len(selected_rows) != expected_records:
        raise RuntimeError(f'record total mismatch: expected={expected_records} records={len(records)}')
    if len({row['record_id'] for row in records}) != len(records):
        raise RuntimeError('duplicate record_id')
    if len({(row['sample_id'], row['segment_id']) for row in records}) != len(records):
        raise RuntimeError('duplicate sample-segment pair')
    print(f'stage=split_and_write records={len(records)}', flush=True)
    splits = split_records(records)
    split_keys = {name: {row['split_group_key'] for row in rows} for name, rows in splits.items()}
    if any((split_keys[a] & split_keys[b] for a, b in (('train', 'valid'), ('train', 'test'), ('valid', 'test')))):
        raise RuntimeError('split group leakage')
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / 'selected_state_nodes.jsonl', selected_rows)
    write_jsonl(out_dir / 'segment_interest_node_sft_all.jsonl', records)
    for split, rows in splits.items():
        write_jsonl(out_dir / f'segment_interest_node_sft_{split}.jsonl', rows)
    write_jsonl(out_dir / 'segment_interest_node_sft_conflicts.jsonl', [])
    report = {'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'baseline_samples': len(baseline_rows), 'final_correct_samples': len(final_samples), 'records': len(records), 'failures': 0, 'split_counts': {name: len(rows) for name, rows in splits.items()}, 'split_group_counts': {name: len(keys) for name, keys in split_keys.items()}, 'final_sample_status_counts': dict(Counter((value['status'] for value in final_samples.values()))), 'record_status_counts': dict(Counter((row['quality']['final_source_status'] for row in records))), 'candidate_counts': dict(Counter((row['quality']['selected_candidate_id'] for row in records))), 'state_repair_candidate_counts': dict(Counter((action['new_candidate_id'] for action in state_success.values()))), 'unique_samples_in_records': len({row['sample_id'] for row in records}), 'unique_pack_hashes': len({row['pack_hash'] for row in records}), 'system_prompt_sha256': digest(STATE_SYSTEM_PROMPT), 'removed_empty_movie_compatibility_fields': removed_movie_compatibility_fields, 'all_complete': True}
    write_json(out_dir / 'segment_interest_node_sft_report.json', report)
    with open(out_dir / 'SHA256SUMS', 'w', encoding='utf-8') as handle:
        for path in sorted(out_dir.iterdir()):
            if path.is_file() and path.name not in {'SHA256SUMS', 'training_preflight_audit.json'}:
                handle.write(f'{file_digest(path)}  {path.name}\n')
    print(json.dumps({'output_dir': str(out_dir), **report}, ensure_ascii=False, indent=2), flush=True)
if __name__ == '__main__':
    main()
