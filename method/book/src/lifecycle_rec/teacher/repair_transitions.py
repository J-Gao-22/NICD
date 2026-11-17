from __future__ import annotations
import argparse
import concurrent.futures as cf
import importlib.util
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any
FIELDS = ('avg_token_nll', 'approx_topk_entropy_mean', 'low_conf_token_rate')
MAX_ATTEMPTS = 2

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n')
    tmp.replace(path)

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    tmp.replace(path)

def segment_number(segment_id: str) -> int:
    return int(re.sub('\\D', '', segment_id) or 0)

def norm_value(values: list[Any], value: Any) -> float:
    finite = [float(item) for item in values if isinstance(item, (int, float)) and math.isfinite(item)]
    if not finite or not isinstance(value, (int, float)) or (not math.isfinite(value)):
        return 0.0
    low, high = (min(finite), max(finite))
    return 0.0 if high <= low else max(0.0, min(1.0, (float(value) - low) / (high - low)))

def select_smoke(sample_ids: list[str], baseline: dict[str, dict[str, Any]], limit: int) -> list[str]:
    selected: list[str] = []
    target_per_label = limit // 2
    for label in (0, 1):
        for sample_id in sample_ids:
            if int(baseline[sample_id]['label']) == label and sample_id not in selected:
                selected.append(sample_id)
                if sum((int(baseline[sid]['label']) == label for sid in selected)) >= target_per_label:
                    break
    for sample_id in sample_ids:
        if len(selected) >= limit:
            break
        if sample_id not in selected:
            selected.append(sample_id)
    return selected

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
    global EDGE_ROOT, BASELINE, OUT_ROOT, JUDGE_ENDPOINT
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('smoke', 'run'), required=True)
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--transition-dir', type=Path, required=True)
    parser.add_argument('--judge-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--judge-endpoint', type=str, required=True)
    args = parser.parse_args()
    EDGE_ROOT = args.transition_dir
    BASELINE = args.judge_dir
    OUT_ROOT = args.output_dir
    JUDGE_ENDPOINT = args.judge_endpoint
    from lifecycle_rec.teacher import judge as renderer
    from lifecycle_rec.teacher import judge_client as judge_runner
    judge_runner.ENDPOINT = JUDGE_ENDPOINT
    prompts = {str(row['sample_id']): row for row in read_jsonl(BASELINE / 'judge_prompts.jsonl')}
    baseline = {str(row['sample_id']): row for row in read_jsonl(BASELINE / 'judge_results.jsonl')}
    failed = [sample_id for sample_id in prompts if baseline[sample_id].get('prediction') != baseline[sample_id].get('label')]
    states_by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(EDGE_ROOT / 'selected_state_nodes.jsonl'):
        states_by_sample.setdefault(str(row['sample_id']), []).append(row)
    for rows in states_by_sample.values():
        rows.sort(key=lambda row: segment_number(str(row['segment_id'])))
    edges: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    summaries: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    raw_summaries: list[dict[str, Any]] = []
    for candidate_id in ('E1', 'E2', 'E3'):
        for row in read_jsonl(EDGE_ROOT / f'{candidate_id}/transition_candidates.jsonl'):
            key = (str(row['sample_id']), str(row['from_segment']), str(row['to_segment']), candidate_id)
            edges[key] = row
            summary = row.get('logprob_summary') or {}
            summaries[key] = summary
            raw_summaries.append(summary)
    field_values = {field: [row.get(field) for row in raw_summaries] for field in FIELDS}

    def uncertainty(summary: dict[str, Any]) -> float:
        return 0.5 * norm_value(field_values['avg_token_nll'], summary.get('avg_token_nll')) + 0.3 * norm_value(field_values['approx_topk_entropy_mean'], summary.get('approx_topk_entropy_mean')) + 0.2 * norm_value(field_values['low_conf_token_rate'], summary.get('low_conf_token_rate'))
    edge_uncertainty = {key: uncertainty(summary) for key, summary in summaries.items()}

    def make_messages(sample_id: str, selected_edges: dict[tuple[str, str], str]) -> list[dict[str, str]]:
        states = states_by_sample[sample_id]
        parts = ['Lifecycle interest chain before the target:']
        for index, state in enumerate(states):
            if index:
                previous = str(states[index - 1]['segment_id'])
                current = str(state['segment_id'])
                candidate_id = selected_edges.get((previous, current), 'E1')
                parts.append(renderer.render_edge(edges[sample_id, previous, current, candidate_id]))
            parts.append(renderer.render_state(state))
        original_user = prompts[sample_id]['messages'][1]['content']
        target_offset = original_user.find('Target book:')
        if target_offset < 0:
            raise ValueError(f'missing Target book marker: {sample_id}')
        user_prompt = '\n\n'.join(parts) + '\n\n' + original_user[target_offset:]
        return [{'role': 'system', 'content': prompts[sample_id]['messages'][0]['content']}, {'role': 'user', 'content': user_prompt}]
    parity_failures = []
    for sample_id, row in prompts.items():
        if make_messages(sample_id, {}) != row['messages']:
            parity_failures.append(sample_id)
            if len(parity_failures) >= 10:
                break
    if parity_failures:
        raise RuntimeError(f'baseline prompt parity failed for: {parity_failures}')

    def build_queue(sample_id: str) -> list[dict[str, Any]]:
        queue = []
        states = states_by_sample[sample_id]
        for index in range(len(states) - 1):
            from_segment = str(states[index]['segment_id'])
            to_segment = str(states[index + 1]['segment_id'])
            current_key = (sample_id, from_segment, to_segment, 'E1')
            replacements = []
            for candidate_id in ('E2', 'E3'):
                key = (sample_id, from_segment, to_segment, candidate_id)
                if key in edges:
                    replacements.append({'candidate_id': candidate_id, 'uncertainty': edge_uncertainty[key]})
            replacements.sort(key=lambda row: row['uncertainty'])
            if replacements:
                current_uncertainty = edge_uncertainty[current_key]
                queue.append({'sample_id': sample_id, 'from_segment': from_segment, 'to_segment': to_segment, 'old_candidate_id': 'E1', 'old_uncertainty': current_uncertainty, 'replacement_candidates': replacements, 'priority': current_uncertainty * (1.0 + 0.2 * index)})
        queue.sort(key=lambda row: -row['priority'])
        return queue
    queues = {sample_id: build_queue(sample_id) for sample_id in failed}
    eligible = [sample_id for sample_id in failed if queues[sample_id]]
    selected = select_smoke(eligible, baseline, args.limit) if args.phase == 'smoke' else failed
    out = OUT_ROOT / ('smoke' if args.phase == 'smoke' else 'full')
    out.mkdir(parents=True, exist_ok=True)
    queue_rows = []
    for sample_id in selected:
        for rank, action in enumerate(queues[sample_id], 1):
            queue_rows.append({**action, 'queue_rank': rank})
    write_jsonl(out / 'repair_action_queue.jsonl', queue_rows)
    write_json(out / 'input_audit.json', {'baseline_prompts': len(prompts), 'baseline_results': len(baseline), 'baseline_failed': len(failed), 'failed_with_edge_action': len(eligible), 'failed_without_edge_action': len(failed) - len(eligible), 'selected_for_phase': len(selected), 'baseline_prompt_parity_failures': 0, 'edge_counts': dict(Counter((key[3] for key in edges))), 'phase': args.phase})

    def run_sample(sample_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        original = baseline[sample_id]
        label = int(original['label'])
        attempts: list[dict[str, Any]] = []
        attempt_id = 0
        for action in queues[sample_id]:
            for replacement in action['replacement_candidates']:
                if attempt_id >= MAX_ATTEMPTS:
                    break
                attempt_id += 1
                edge_selection = {(action['from_segment'], action['to_segment']): replacement['candidate_id']}
                prompt_row = {**prompts[sample_id], 'messages': make_messages(sample_id, edge_selection)}
                result = judge_runner.call_model(prompt_row, attempt_id)
                prediction = result.get('prediction')
                success = prediction == label
                attempt = {'sample_id': sample_id, 'attempt_id': attempt_id, 'label': label, 'repair_action': 'edge_replacement', 'target_node': f"{action['from_segment']}->{action['to_segment']}", 'old_candidate_id': 'E1', 'new_candidate_id': replacement['candidate_id'], 'old_uncertainty': action['old_uncertainty'], 'new_uncertainty': replacement['uncertainty'], 'judge_before_prediction': original.get('prediction'), 'judge_before_score': original.get('score'), 'judge_after_prediction': prediction, 'judge_after_score': result.get('score'), 'score_source': result.get('score_source'), 'success': success, 'valid_prediction': prediction is not None, 'response_text': result.get('response_text'), 'error': result.get('error'), 'latency_sec': result.get('latency_sec')}
                attempts.append(attempt)
                if success:
                    return ({'sample_id': sample_id, 'label': label, 'original_prediction': original.get('prediction'), 'original_score': original.get('score'), 'final_prediction': prediction, 'final_score': result.get('score'), 'final_score_source': result.get('score_source'), 'final_correct': True, 'final_status': 'repaired_success_edge_replacement', 'num_attempts': attempt_id, 'successful_attempt_id': attempt_id, 'target_node': attempt['target_node'], 'new_candidate_id': replacement['candidate_id']}, attempts)
            if attempt_id >= MAX_ATTEMPTS:
                break
        return ({'sample_id': sample_id, 'label': label, 'original_prediction': original.get('prediction'), 'original_score': original.get('score'), 'final_prediction': original.get('prediction'), 'final_score': original.get('score'), 'final_score_source': original.get('score_source'), 'final_correct': False, 'final_status': 'edge_repair_failed_or_no_eligible_action', 'num_attempts': attempt_id, 'successful_attempt_id': None}, attempts)
    final_path = out / 'repair_final_results.jsonl'
    attempt_path = out / 'repair_attempts.jsonl'
    existing_finals = {row['sample_id']: row for row in read_jsonl(final_path)} if final_path.exists() else {}
    pending = [sample_id for sample_id in selected if sample_id not in existing_finals]
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
                existing_finals[final['sample_id']] = final
                if completed % 25 == 0 or completed == len(pending):
                    print(json.dumps({'completed_this_run': completed, 'pending_at_start': len(pending)}), flush=True)
    selected_finals = [existing_finals[sample_id] for sample_id in selected]
    successful = [row for row in selected_finals if row['final_correct']]
    valid_attempts = [row for row in read_jsonl(attempt_path) if row.get('valid_prediction')]
    report: dict[str, Any] = {'experiment': 'amazon_book_entropy_repair_v1_edge_only', 'phase': args.phase, 'baseline_total': len(baseline), 'baseline_correct': sum((row.get('prediction') == row.get('label') for row in baseline.values())), 'baseline_accuracy': sum((row.get('prediction') == row.get('label') for row in baseline.values())) / len(baseline), 'baseline_auc': auc([int(prompts[sid]['label']) for sid in prompts], [float(baseline[sid]['score']) for sid in prompts]), 'baseline_failed': len(failed), 'selected_failed_samples': len(selected), 'selected_with_edge_action': sum((bool(queues[sid]) for sid in selected)), 'repair_final_rows': len(selected_finals), 'repair_attempts': sum((row['num_attempts'] for row in selected_finals)), 'valid_repair_attempts': len(valid_attempts), 'repaired_success': len(successful), 'repair_success_rate_selected': len(successful) / len(selected_finals) if selected_finals else None, 'workers': args.workers, 'max_attempts_per_failed_sample': MAX_ATTEMPTS, 'temperature': 0.0, 'max_tokens': 192, 'top_logprobs': 20, 'elapsed_sec_this_run': time.time() - started, 'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S')}
    if args.phase == 'run':
        predictions, scores, labels = ([], [], [])
        for sample_id in prompts:
            labels.append(int(prompts[sample_id]['label']))
            repaired = existing_finals.get(sample_id)
            if repaired and repaired['final_correct']:
                predictions.append(repaired['final_prediction'])
                scores.append(float(repaired['final_score']))
            else:
                predictions.append(baseline[sample_id]['prediction'])
                scores.append(float(baseline[sample_id]['score']))
        correct = sum((prediction == label for prediction, label in zip(predictions, labels)))
        report.update({'correct_after_edge_repair': correct, 'accuracy_after_edge_repair': correct / len(labels), 'auc_after_edge_repair_logprob': auc(labels, scores)})
    write_json(out / 'repair_report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
