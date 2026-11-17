from __future__ import annotations
import gzip
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator
from transformers import AutoTokenizer
from lifecycle_rec.teacher import transition_support as edgev3
VERSION = 'amazon_book_state_entropy_repair_edge_regen_v1'
MAX_STATE_ATTEMPTS = 2
FIELDS = ('avg_token_nll', 'approx_topk_entropy_mean', 'low_conf_token_rate')

def read(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    tmp.replace(path)

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    tmp.replace(path)

def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def segnum(value: Any) -> int:
    return int(re.sub('\\D', '', str(value)) or 0)

def stable(values: list[Any], limit: int) -> list[str]:
    out, seen = ([], set())
    for value in values:
        value = str(value).strip()
        folded = value.casefold()
        if value and folded != 'none' and (folded not in seen):
            out.append(value)
            seen.add(folded)
        if len(out) >= limit:
            break
    return out

def compare(left: list[str], right: list[str], limit: int=6) -> dict[str, list[str]]:
    left_keys = {value.casefold() for value in left}
    right_keys = {value.casefold() for value in right}
    return {'retained': [value for value in left if value.casefold() in right_keys][:limit], 'emerging': [value for value in right if value.casefold() not in left_keys][:limit], 'fading': [value for value in left if value.casefold() not in right_keys][:limit]}

def authors(state: dict[str, Any]) -> tuple[list[str], list[str]]:
    cues = (state.get('decision_cues') or {}).get('author_cues') or {}
    return (stable(cues.get('liked_authors') or [], 6), stable(cues.get('rejected_authors') or [], 6))

def object_terms(rows: list[Any], pattern_key: str) -> list[str]:
    values = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        if row.get(pattern_key):
            values.append(row[pattern_key])
        values.extend(row.get('support_terms') or [])
    return values

def polarity_terms(state: dict[str, Any], polarity: str) -> list[str]:
    cues = state.get('decision_cues') or {}
    if polarity == 'positive':
        analysis = state.get('positive_analysis') or {}
        values = list(cues.get('positive_match_cues') or [])
        values += object_terms(analysis.get('liked_patterns') or analysis.get('positive_patterns') or [], 'pattern')
    else:
        analysis = state.get('negative_analysis') or {}
        values = list(cues.get('negative_match_cues') or [])
        values += object_terms(analysis.get('rejected_or_risky_patterns') or analysis.get('negative_patterns') or [], 'pattern')
    return stable(values, 40)

def transition_analysis(from_state: dict[str, Any], to_state: dict[str, Any]) -> dict[str, Any]:
    from_positive, to_positive = (polarity_terms(from_state, 'positive'), polarity_terms(to_state, 'positive'))
    from_negative, to_negative = (polarity_terms(from_state, 'negative'), polarity_terms(to_state, 'negative'))
    positive = compare(from_positive, to_positive, 8)
    negative = compare(from_negative, to_negative, 6)
    positive_keys = {value.casefold() for value in from_positive + to_positive}
    negative_keys = {value.casefold() for value in from_negative + to_negative}
    conflicts = stable([value for value in from_positive + to_positive + from_negative + to_negative if value.casefold() in positive_keys & negative_keys], 8)
    from_liked, from_rejected = authors(from_state)
    to_liked, to_rejected = authors(to_state)
    return {'temporal_direction': 'FROM is earlier and TO is later. Set differences describe state representation only, not preference strength or causal change.', 'from_state_type': from_state.get('state_type') or '', 'to_state_type': to_state.get('state_type') or '', 'program_suggested_transition_type': edgev3.suggest_type(positive, negative, conflicts), 'positive_signal_comparison': {'retained_positive_terms': positive['retained'], 'emerging_positive_terms': positive['emerging'], 'fading_positive_terms': positive['fading']}, 'negative_caution_comparison': {'retained_negative_terms': negative['retained'], 'emerging_negative_terms': negative['emerging'], 'fading_negative_terms': negative['fading']}, 'polarity_conflict_terms': conflicts, 'author_signal_comparison': {'positive_authors': compare(from_liked, to_liked), 'negative_caution_authors': compare(from_rejected, to_rejected), 'evidence_rule': 'Authors appear only when repeated same-polarity state evidence produced an author cue.'}, 'program_notes': ['Broad category or tone terms must not become hard negative evidence.', 'Fading means present in FROM and absent from TO state output. It does not prove weaker interest.', 'Emerging means absent from FROM and present in TO state output. It does not prove stronger interest.', 'Below-5 interactions provide non-causal tentative local caution only.', 'Do not infer author preference from an author absent from the state author cues.']}

def norm_value(values: list[float], value: Any) -> float:
    if not values or not isinstance(value, (int, float)) or (not math.isfinite(value)):
        return 0.0
    low, high = (min(values), max(values))
    return 0.0 if high <= low else max(0.0, min(1.0, (float(value) - low) / (high - low)))

def main() -> None:
    global SOURCE, BASE_JUDGE, EDGE_ONLY, OUT, TOKENIZER, PROMPT, SCHEMA
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--state-candidates', type=Path, required=True)
    parser.add_argument('--judge-dir', type=Path, required=True)
    parser.add_argument('--transition-repair-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--prompt', type=Path, required=True)
    parser.add_argument('--schema', type=Path, required=True)
    args = parser.parse_args()
    SOURCE = args.state_candidates
    BASE_JUDGE = args.judge_dir
    EDGE_ONLY = args.transition_repair_dir
    OUT = args.output_dir
    TOKENIZER = args.tokenizer
    PROMPT = args.prompt
    SCHEMA = args.schema
    baseline = {row['sample_id']: row for row in read(BASE_JUDGE / 'judge_results.jsonl')}
    edge_final = {row['sample_id']: row for row in read(EDGE_ONLY / 'repair_final_results.jsonl')}
    remaining = {sample_id for sample_id, row in baseline.items() if row.get('prediction') != row.get('label') and (not edge_final.get(sample_id, {}).get('final_correct'))}
    field_values = {field: [] for field in FIELDS}
    source_rows = 0
    for row in read(SOURCE):
        source_rows += 1
        summary = row.get('logprob_summary') or {}
        for field in FIELDS:
            value = summary.get(field)
            if isinstance(value, (int, float)) and math.isfinite(value):
                field_values[field].append(float(value))
    states: dict[tuple[str, str, str], dict[str, Any]] = {}
    segments: dict[str, set[str]] = defaultdict(set)
    for row in read(SOURCE):
        sample_id = str(row['sample_id'])
        if sample_id not in remaining:
            continue
        candidate = (row.get('model_output') or {}).get('candidate') or {}
        compact = {'sample_id': sample_id, 'user_id': row.get('user_id'), 'item_id': row.get('item_id'), 'segment_id': str(row['segment_id']), 'candidate_id': str(row['candidate_id']), 'candidate': candidate, 'pack_hash': row.get('pack_hash'), 'logprob_summary': {field: (row.get('logprob_summary') or {}).get(field) for field in FIELDS}}
        states[sample_id, compact['segment_id'], compact['candidate_id']] = compact
        segments[sample_id].add(compact['segment_id'])

    def uncertainty(summary: dict[str, Any]) -> float:
        return 0.5 * norm_value(field_values['avg_token_nll'], summary.get('avg_token_nll')) + 0.3 * norm_value(field_values['approx_topk_entropy_mean'], summary.get('approx_topk_entropy_mean')) + 0.2 * norm_value(field_values['low_conf_token_rate'], summary.get('low_conf_token_rate'))
    state_uncertainty = {key: uncertainty(row['logprob_summary']) for key, row in states.items()}
    attempts = []
    for sample_id in sorted(remaining):
        ordered = sorted(segments.get(sample_id, []), key=segnum)
        queue = []
        for index, segment in enumerate(ordered):
            current_key = (sample_id, segment, 'C1')
            current_uncertainty = state_uncertainty[current_key]
            replacements = []
            for candidate_id in ('C2', 'C3'):
                key = (sample_id, segment, candidate_id)
                candidate = states.get(key)
                if not candidate or candidate['candidate'].get('candidate_status') != 'proposed':
                    continue
                replacement_uncertainty = state_uncertainty[key]
                if replacement_uncertainty <= current_uncertainty + 0.05:
                    replacements.append((candidate_id, replacement_uncertainty))
            if replacements:
                candidate_id, replacement_uncertainty = min(replacements, key=lambda item: item[1])
                recency = 1.0 + (0.4 * index / max(1, len(ordered) - 1) if len(ordered) > 1 else 0.2)
                queue.append({'segment': segment, 'old_uncertainty': current_uncertainty, 'new_candidate_id': candidate_id, 'new_uncertainty': replacement_uncertainty, 'priority': current_uncertainty * recency})
        for attempt_id, action in enumerate(sorted(queue, key=lambda row: -row['priority'])[:MAX_STATE_ATTEMPTS], 1):
            attempts.append({'repair_attempt_key': f"{sample_id}::state_{action['segment']}_{action['new_candidate_id']}::attempt{attempt_id}", 'sample_id': sample_id, 'attempt_id': attempt_id, 'repair_action': 'state_replacement_with_edge_regeneration', 'target_node': action['segment'], 'old_candidate_id': 'C1', 'new_candidate_id': action['new_candidate_id'], 'old_uncertainty': action['old_uncertainty'], 'new_uncertainty': action['new_uncertainty'], 'priority': action['priority'], 'label': int(baseline[sample_id]['label'])})
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
    source_hash = file_hash(SOURCE)
    prompt_hash = file_hash(PROMPT)
    schema_hash = file_hash(SCHEMA)
    packs, failures = ([], [])
    for action in attempts:
        sample_id = action['sample_id']
        replacement_segment = action['target_node']
        ordered = sorted(segments[sample_id], key=segnum)
        selected = {}
        for segment in ordered:
            candidate_id = action['new_candidate_id'] if segment == replacement_segment else 'C1'
            selected[segment] = states[sample_id, segment, candidate_id]
        for from_segment, to_segment in zip(ordered, ordered[1:]):
            if replacement_segment not in (from_segment, to_segment):
                continue
            identity = {'repair_attempt_key': action['repair_attempt_key'], 'sample_id': sample_id, 'user_id': selected[from_segment]['user_id'], 'item_id': selected[from_segment]['item_id'], 'from_segment': from_segment, 'to_segment': to_segment}
            try:
                from_state = edgev3.state_core_payload({'segment_id': from_segment, 'selected_candidate': selected[from_segment]['candidate']})
                to_state = edgev3.state_core_payload({'segment_id': to_segment, 'selected_candidate': selected[to_segment]['candidate']})
                analysis = transition_analysis(from_state, to_state)
                if not any(analysis['positive_signal_comparison'].values()) and (not any(analysis['negative_caution_comparison'].values())):
                    failures.append({**identity, 'failure_type': 'insufficient_transition_analysis'})
                    continue
                row = {**identity, 'prompt_version': VERSION, 'edge_task': {'edge_id': f'{from_segment}->{to_segment}', 'from_state_id': from_segment, 'to_state_id': to_segment, 'expected_output': 'Analyze positive and tentative negative-caution updates between adjacent book-interest states.'}, 'from_state_node': from_state, 'to_state_node': to_state, 'transition_analysis_data': analysis, 'source_file_hashes': {'state_candidates': source_hash, 'edge_prompt_e1': prompt_hash, 'edge_schema': schema_hash}}
                row['pack_text'] = edgev3.build_pack_text(row)
                row['pack_token_count'] = len(tokenizer.encode(row['pack_text']))
                row['pack_hash'] = hashlib.sha256(row['pack_text'].encode()).hexdigest()
                packs.append(row)
            except Exception as exc:
                failures.append({**identity, 'failure_type': 'pack_build_exception', 'error': repr(exc)})
    smoke_attempt_keys = []
    for label in (0, 1):
        for action in attempts:
            if action['label'] == label and action['repair_attempt_key'] not in smoke_attempt_keys:
                smoke_attempt_keys.append(action['repair_attempt_key'])
                if sum((next((row['label'] for row in attempts if row['repair_attempt_key'] == key)) == label for key in smoke_attempt_keys)) >= 5:
                    break
    smoke_key_set = set(smoke_attempt_keys)
    smoke_actions = [row for row in attempts if row['repair_attempt_key'] in smoke_key_set]
    smoke_packs = [row for row in packs if row['repair_attempt_key'] in smoke_key_set]
    forbidden = ('target_title', 'target_semantic', 'target_profile', 'target_label', 'gold_item', 'recommendation_judge')
    leakage = []
    for pack in packs:
        hits = [marker for marker in forbidden if marker in pack['pack_text'].casefold()]
        if hits:
            leakage.append({'repair_attempt_key': pack['repair_attempt_key'], 'sample_id': pack['sample_id'], 'from_segment': pack['from_segment'], 'to_segment': pack['to_segment'], 'markers': hits})
    write_jsonl(OUT / 'state_repair_action_queue.jsonl', attempts)
    write_jsonl(OUT / 'repair_transition_packs.jsonl', packs)
    write_jsonl(OUT / 'edge_regeneration_pack_failures_v1.jsonl', failures)
    write_jsonl(OUT / 'edge_regeneration_leakage_failures_v1.jsonl', leakage)
    write_jsonl(OUT / 'smoke_state_repair_action_queue.jsonl', smoke_actions)
    write_jsonl(OUT / 'smoke_edge_regeneration_packs_v1.jsonl', smoke_packs)
    report = {'version': VERSION, 'source_candidate_rows': source_rows, 'remaining_after_edge_only': len(remaining), 'remaining_with_state_candidates': len(segments), 'state_repair_attempts': len(attempts), 'samples_with_state_repair_attempts': len({row['sample_id'] for row in attempts}), 'edge_regeneration_packs': len(packs), 'edge_regeneration_pack_failures': len(failures), 'leakage_failures': len(leakage), 'attempt_candidate_counts': dict(Counter((row['new_candidate_id'] for row in attempts))), 'attempt_label_counts': dict(Counter((str(row['label']) for row in attempts))), 'smoke_attempts': len(smoke_actions), 'smoke_edge_packs': len(smoke_packs), 'pack_token_counts': {'min': min((row['pack_token_count'] for row in packs)) if packs else None, 'max': max((row['pack_token_count'] for row in packs)) if packs else None, 'mean': sum((row['pack_token_count'] for row in packs)) / len(packs) if packs else None}, 'max_state_attempts_per_sample': MAX_STATE_ATTEMPTS, 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S')}
    write_json(OUT / 'state_repair_pack_build_report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
