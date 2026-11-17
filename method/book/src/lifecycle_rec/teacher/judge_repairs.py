from __future__ import annotations
import argparse
import concurrent.futures as cf
import gzip
import importlib.util
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator
FORBIDDEN_FIELD_RE = re.compile('(?im)^\\s*(?:label|rating|target_item_id|item_id|is_gold|gold|prediction|candidate_source)\\s*:')

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

def segnum(value: Any) -> int:
    return int(re.sub('\\D', '', str(value)) or 0)

def auc(labels: list[int], scores: list[float]) -> float | None:
    pairs = sorted(zip(scores, labels), key=lambda pair: pair[0])
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum += average_rank * sum((pairs[pos][1] for pos in range(index, end)))
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)

def main() -> None:
    global STATE_SOURCE, EDGE_ROOT, BASELINE, EDGE_ONLY, ROOT, JUDGE_ENDPOINT, REGEN
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('smoke', 'run'), required=True)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--state-candidates', type=Path, required=True)
    parser.add_argument('--transition-dir', type=Path, required=True)
    parser.add_argument('--judge-dir', type=Path, required=True)
    parser.add_argument('--transition-repair-dir', type=Path, required=True)
    parser.add_argument('--state-repair-dir', type=Path, required=True)
    parser.add_argument('--judge-endpoint', type=str, required=True)
    args = parser.parse_args()
    STATE_SOURCE = args.state_candidates
    EDGE_ROOT = args.transition_dir
    BASELINE = args.judge_dir
    EDGE_ONLY = args.transition_repair_dir
    ROOT = args.state_repair_dir
    JUDGE_ENDPOINT = args.judge_endpoint
    REGEN = ROOT / 'transitions'
    out = ROOT / ('state_repair_judge_smoke' if args.phase == 'smoke' else 'judged_states')
    out.mkdir(parents=True, exist_ok=True)
    from lifecycle_rec.teacher import judge as renderer
    from lifecycle_rec.teacher import judge_client as judge_runner
    judge_runner.ENDPOINT = JUDGE_ENDPOINT
    prompts = {str(row['sample_id']): row for row in read(BASELINE / 'judge_prompts.jsonl')}
    baseline = {str(row['sample_id']): row for row in read(BASELINE / 'judge_results.jsonl')}
    edge_finals = {str(row['sample_id']): row for row in read(EDGE_ONLY / 'repair_final_results.jsonl')}
    remaining = [sample_id for sample_id in prompts if baseline[sample_id].get('prediction') != baseline[sample_id].get('label') and (not edge_finals.get(sample_id, {}).get('final_correct'))]
    all_actions = list(read(ROOT / 'state_repair_action_queue.jsonl'))
    selected_actions = list(read(ROOT / 'smoke_state_repair_action_queue.jsonl')) if args.phase == 'smoke' else all_actions
    actions_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for action in selected_actions:
        actions_by_sample[str(action['sample_id'])].append(action)
    for rows in actions_by_sample.values():
        rows.sort(key=lambda row: int(row['attempt_id']))
    selected_samples = list(actions_by_sample) if args.phase == 'smoke' else remaining
    c1_states: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read(EDGE_ROOT / 'selected_state_nodes.jsonl'):
        c1_states[str(row['sample_id'])].append(row)
    for rows in c1_states.values():
        rows.sort(key=lambda row: segnum(row['segment_id']))
    required_replacements = {(str(row['sample_id']), str(row['target_node']), str(row['new_candidate_id'])) for row in selected_actions}
    replacements: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in read(STATE_SOURCE):
        key = (str(row['sample_id']), str(row['segment_id']), str(row['candidate_id']))
        if key in required_replacements:
            replacements[key] = (row.get('model_output') or {}).get('candidate') or {}
    missing_replacements = sorted(required_replacements - set(replacements))
    if missing_replacements:
        raise RuntimeError(f'missing replacement states: {missing_replacements[:10]}')
    original_edges = {(str(row['sample_id']), str(row['from_segment']), str(row['to_segment'])): row for row in read(EDGE_ROOT / 'E1/transition_candidates.jsonl')}
    regen_packs = {(str(row['repair_attempt_key']), str(row['from_segment']), str(row['to_segment'])): str(row['pack_hash']) for row in read(ROOT / 'repair_transition_packs.jsonl')}
    regen_edges = {str(row['pack_hash']): row for row in read(REGEN / 'transition_candidates.jsonl')}

    def original_messages(sample_id: str) -> list[dict[str, str]]:
        parts = ['Lifecycle interest chain before the target:']
        states = c1_states[sample_id]
        for index, state in enumerate(states):
            if index:
                previous, current = (str(states[index - 1]['segment_id']), str(state['segment_id']))
                parts.append(renderer.render_edge(original_edges[sample_id, previous, current]))
            parts.append(renderer.render_state(state))
        original_user = prompts[sample_id]['messages'][1]['content']
        target_offset = original_user.find('Target book:')
        return [prompts[sample_id]['messages'][0], {'role': 'user', 'content': '\n\n'.join(parts) + '\n\n' + original_user[target_offset:]}]
    parity_failures = []
    for sample_id, prompt in prompts.items():
        if original_messages(sample_id) != prompt['messages']:
            parity_failures.append(sample_id)
            if len(parity_failures) >= 10:
                break
    if parity_failures:
        raise RuntimeError(f'baseline prompt parity failed: {parity_failures}')

    def action_messages(action: dict[str, Any]) -> list[dict[str, str]]:
        sample_id = str(action['sample_id'])
        target_segment = str(action['target_node'])
        candidate_id = str(action['new_candidate_id'])
        action_key = str(action['repair_attempt_key'])
        states = c1_states[sample_id]
        parts = ['Lifecycle interest chain before the target:']
        for index, state in enumerate(states):
            segment = str(state['segment_id'])
            if index:
                previous = str(states[index - 1]['segment_id'])
                if target_segment in (previous, segment):
                    pack_hash = regen_packs[action_key, previous, segment]
                    edge = regen_edges[pack_hash]
                else:
                    edge = original_edges[sample_id, previous, segment]
                parts.append(renderer.render_edge(edge))
            if segment == target_segment:
                replacement = replacements[sample_id, segment, candidate_id]
                parts.append(renderer.render_state({'segment_id': segment, 'selected_candidate': replacement}))
            else:
                parts.append(renderer.render_state(state))
        original_user = prompts[sample_id]['messages'][1]['content']
        target_offset = original_user.find('Target book:')
        return [prompts[sample_id]['messages'][0], {'role': 'user', 'content': '\n\n'.join(parts) + '\n\n' + original_user[target_offset:]}]
    audit_failures, action_prompt_rows = ([], [])
    for action in selected_actions:
        messages = action_messages(action)
        user_prompt = messages[1]['content']
        target_offset = user_prompt.find('Target book:')
        chain = user_prompt[:target_offset]
        title_match = re.search('(?m)^Title:\\s*(.+)$', user_prompt[target_offset:])
        target_title = title_match.group(1).strip().casefold() if title_match else ''
        field_leaks = FORBIDDEN_FIELD_RE.findall(chain)
        replacement = replacements[str(action['sample_id']), str(action['target_node']), str(action['new_candidate_id'])]
        evidence_titles = set()
        for analysis_name, pattern_name in (('positive_analysis', 'liked_patterns'), ('negative_analysis', 'rejected_or_risky_patterns')):
            analysis = replacement.get(analysis_name) or {}
            for pattern in analysis.get(pattern_name) or []:
                for title in pattern.get('evidence_titles') or []:
                    normalized = renderer.clean(title).casefold()
                    if normalized:
                        evidence_titles.add(normalized)
        target_in_chain = bool(target_title and target_title in evidence_titles)
        if field_leaks or target_in_chain:
            audit_failures.append({'repair_attempt_key': action['repair_attempt_key'], 'sample_id': action['sample_id'], 'field_leaks': field_leaks, 'target_title_in_chain': target_in_chain})
        action_prompt_rows.append({**action, 'messages': messages})
    write_jsonl(out / 'state_repair_prompts.jsonl', action_prompt_rows)
    write_jsonl(out / 'leakage_failures.jsonl', audit_failures)
    if audit_failures:
        raise RuntimeError(f'state repair prompt leakage failures: {len(audit_failures)}')
    prompt_by_key = {row['repair_attempt_key']: row for row in action_prompt_rows}

    def run_sample(sample_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        original = baseline[sample_id]
        label = int(original['label'])
        attempts = []
        for action in actions_by_sample.get(sample_id, [])[:2]:
            prompt_row = prompt_by_key[action['repair_attempt_key']]
            result = judge_runner.call_model(prompt_row, int(action['attempt_id']))
            prediction = result.get('prediction')
            success = prediction == label
            attempt = {**action, 'judge_before_prediction': original.get('prediction'), 'judge_before_score': original.get('score'), 'judge_after_prediction': prediction, 'judge_after_score': result.get('score'), 'score_source': result.get('score_source'), 'success': success, 'valid_prediction': prediction is not None, 'response_text': result.get('response_text'), 'latency_sec': result.get('latency_sec'), 'error': result.get('error')}
            attempts.append(attempt)
            if success:
                return ({'sample_id': sample_id, 'label': label, 'original_prediction': original.get('prediction'), 'original_score': original.get('score'), 'final_prediction': prediction, 'final_score': result.get('score'), 'final_score_source': result.get('score_source'), 'final_correct': True, 'final_status': 'repaired_success_state_edge_regen', 'num_attempts': len(attempts), 'successful_attempt_id': action['attempt_id'], 'repair_attempt_key': action['repair_attempt_key'], 'target_node': action['target_node'], 'new_candidate_id': action['new_candidate_id']}, attempts)
        return ({'sample_id': sample_id, 'label': label, 'original_prediction': original.get('prediction'), 'original_score': original.get('score'), 'final_prediction': original.get('prediction'), 'final_score': original.get('score'), 'final_score_source': original.get('score_source'), 'final_correct': False, 'final_status': 'state_repair_failed_or_no_eligible_action', 'num_attempts': len(attempts), 'successful_attempt_id': None}, attempts)
    final_path = out / 'state_repair_final_results.jsonl'
    attempt_path = out / 'state_repair_attempts.jsonl'
    existing = {row['sample_id']: row for row in read(final_path)} if final_path.exists() else {}
    pending = [sample_id for sample_id in selected_samples if sample_id not in existing]
    started = time.time()
    with final_path.open('a', encoding='utf-8') as final_handle, attempt_path.open('a', encoding='utf-8') as attempt_handle:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_sample, sample_id): sample_id for sample_id in pending}
            for completed, future in enumerate(cf.as_completed(futures), 1):
                final, attempts = future.result()
                for attempt in attempts:
                    attempt_handle.write(json.dumps(attempt, ensure_ascii=False) + '\n')
                final_handle.write(json.dumps(final, ensure_ascii=False) + '\n')
                attempt_handle.flush()
                final_handle.flush()
                existing[final['sample_id']] = final
                if completed % 25 == 0 or completed == len(pending):
                    print(json.dumps({'completed': completed, 'pending_at_start': len(pending)}), flush=True)
    phase_finals = [existing[sample_id] for sample_id in selected_samples]
    phase_attempts = list(read(attempt_path)) if attempt_path.exists() else []
    successful = [row for row in phase_finals if row['final_correct']]
    report: dict[str, Any] = {'experiment': 'amazon_book_entropy_repair_v1_edge_then_state_with_edge_regeneration', 'phase': args.phase, 'baseline_total': len(baseline), 'baseline_correct': 3654, 'baseline_accuracy': 0.609, 'baseline_auc': 0.637560542822109, 'edge_repaired_success': sum((row.get('final_correct') for row in edge_finals.values())), 'remaining_after_edge_only': len(remaining), 'selected_samples': len(selected_samples), 'samples_with_state_actions': sum((bool(actions_by_sample.get(sample_id)) for sample_id in selected_samples)), 'state_repair_attempts': sum((row['num_attempts'] for row in phase_finals)), 'valid_state_repair_attempts': sum((row.get('valid_prediction') for row in phase_attempts)), 'state_repaired_success': len(successful), 'state_repair_success_rate': len(successful) / len(phase_finals) if phase_finals else None, 'baseline_prompt_parity_failures': 0, 'leakage_failures': 0, 'workers': args.workers, 'temperature': 0.0, 'max_tokens': 192, 'top_logprobs': 20, 'elapsed_sec_this_run': time.time() - started, 'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S')}
    if args.phase == 'run':
        labels, predictions, scores = ([], [], [])
        for sample_id in prompts:
            labels.append(int(prompts[sample_id]['label']))
            if edge_finals.get(sample_id, {}).get('final_correct'):
                predictions.append(edge_finals[sample_id]['final_prediction'])
                scores.append(float(edge_finals[sample_id]['final_score']))
            elif existing.get(sample_id, {}).get('final_correct'):
                predictions.append(existing[sample_id]['final_prediction'])
                scores.append(float(existing[sample_id]['final_score']))
            else:
                predictions.append(baseline[sample_id]['prediction'])
                scores.append(float(baseline[sample_id]['score']))
        correct = sum((prediction == label for prediction, label in zip(predictions, labels)))
        report.update({'total_repaired_success': report['edge_repaired_success'] + len(successful), 'correct_after_edge_then_state_repair': correct, 'accuracy_after_edge_then_state_repair': correct / len(labels), 'auc_after_edge_then_state_repair_logprob': auc(labels, scores)})
    write_json(out / 'state_repair_report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
