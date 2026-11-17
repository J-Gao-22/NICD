import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER', '0')
os.environ.setdefault('CC', '/usr/bin/gcc')
os.environ.setdefault('CXX', '/usr/bin/g++')
from lifecycle_rec.prepare.state_messages import base, finalize_user, first_row, read_jsonl
from lifecycle_rec.generators.state_format import extract_json, validate_schema
from lifecycle_rec.generators.grounding import validate as validate_grounding
from lifecycle_rec.generators.repair import repair_output

def state_schema():
    string = {'type': 'string'}
    strings = {'type': 'array', 'items': string}
    pattern = {'type': 'object', 'additionalProperties': False, 'properties': {'evidence_titles': strings, 'pattern': string, 'support_terms': strings}, 'required': ['evidence_titles', 'pattern', 'support_terms']}

    def analysis(pattern_key):
        return {'type': 'object', 'additionalProperties': False, 'properties': {pattern_key: {'type': 'array', 'items': pattern}, 'reasoning': string}, 'required': [pattern_key, 'reasoning']}
    author_cues = {'type': 'object', 'additionalProperties': False, 'properties': {'liked_authors': strings, 'rejected_authors': strings}, 'required': ['liked_authors', 'rejected_authors']}
    cues = {'type': 'object', 'additionalProperties': False, 'properties': {'author_cues': author_cues, 'positive_match_cues': strings, 'negative_match_cues': strings}, 'required': ['author_cues', 'positive_match_cues', 'negative_match_cues']}
    node = {'type': 'object', 'additionalProperties': False, 'properties': {'state_type': {'type': 'string', 'enum': ['positive_preference_state', 'negative_rejection_state', 'mixed_feedback_state']}, 'state_label': string, 'state_summary': string, 'positive_analysis': analysis('liked_patterns'), 'negative_analysis': analysis('rejected_or_risky_patterns'), 'decision_cues': cues}, 'required': ['state_type', 'state_label', 'state_summary', 'positive_analysis', 'negative_analysis', 'decision_cues']}
    return {'type': 'object', 'additionalProperties': False, 'properties': {'interest_state_node': node}, 'required': ['interest_state_node']}

def validate_repair_shape(value):
    if not isinstance(value, dict) or not isinstance(value.get('interest_state_node'), dict):
        return ['invalid_state_node_type']
    node = value['interest_state_node']
    for analysis_key, pattern_key in (('positive_analysis', 'liked_patterns'), ('negative_analysis', 'rejected_or_risky_patterns')):
        analysis = node.get(analysis_key)
        if not isinstance(analysis, dict) or not isinstance(analysis.get(pattern_key), list):
            return [f'invalid_{analysis_key}_type']
        for pattern in analysis[pattern_key]:
            if not isinstance(pattern, dict) or not isinstance(pattern.get('evidence_titles'), list) or (not isinstance(pattern.get('support_terms'), list)):
                return [f'invalid_{pattern_key}_type']
            if not all((isinstance(item, str) for item in pattern['evidence_titles'] + pattern['support_terms'])):
                return [f'invalid_{pattern_key}_item_type']
    cues = node.get('decision_cues')
    if not isinstance(cues, dict) or not isinstance(cues.get('author_cues'), dict):
        return ['invalid_cues_type']
    for key in ('positive_match_cues', 'negative_match_cues'):
        if not isinstance(cues.get(key), list) or not all((isinstance(item, str) for item in cues[key])):
            return [f'invalid_{key}_type']
    for key in ('liked_authors', 'rejected_authors'):
        authors = cues['author_cues'].get(key)
        if not isinstance(authors, list) or not all((isinstance(item, str) for item in authors)):
            return [f'invalid_{key}_type']
    return []

def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')

def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')

def load_done(output_dir, splits):
    done = set()
    for split in splits:
        path = output_dir / f'{split}_state_nodes.jsonl'
        if path.exists():
            for row in read_jsonl(path):
                done.add((str(row['sample_id']), str(row['segment_id'])))
    return done

def make_records(row, items, embeddings, system_prompt, max_model_len, max_tokens, tokenizer):
    target_id = int(row['item_id'])
    history_ids = [int(value) for value in row['history_item_ids']]
    target = items[target_id]
    if target_id in history_ids:
        return ([], [{'sample_id': row['sample_id'], 'type': 'target_item_id_in_history'}])
    segments, scores = base.build_segments(row, embeddings)
    target_title = base.normalize_title(target.get('title'))
    records, errors = ([], [])
    for segment in segments:
        start, end = (segment['start'], segment['end'])
        segment_ids = history_ids[start:end]
        pack = finalize_user(base.render_pack(row, segment, items, base.clean(target.get('title'))))
        local_errors = []
        if target_id in segment_ids:
            local_errors.append('target_item_id_in_segment')
        if target_title and len(target_title) >= 8 and (target_title in base.normalize_title(pack)):
            local_errors.append('target_title_phrase_in_pack')
        if base.clean(target.get('asin')) and base.clean(target.get('asin')) in pack:
            local_errors.append('target_asin_in_pack')
        if any((marker in pack for marker in ('target_iid:', 'target_item_id:', 'target_label:', 'sample_id:'))):
            local_errors.append('target_control_marker_in_pack')
        messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': pack}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        if prompt_tokens + max_tokens > max_model_len:
            local_errors.append('context_length_overflow')
        if local_errors:
            errors.append({'sample_id': row['sample_id'], 'segment_id': segment['segment_id'], 'types': local_errors, 'prompt_tokens': prompt_tokens})
            continue
        records.append({'record_id': f"{row['sample_id']}::{segment['segment_id']}::C1", 'sample_id': row['sample_id'], 'split': row['split'], 'row_index': int(row['row_index']), 'user_id': int(row['user_id']), 'item_id': target_id, 'source_label': int(row['label']), 'rating': float(row['rating']), 'timestamp': int(row['timestamp']), 'history_recent30_len': len(history_ids), 'num_segments': len(segments), 'segment_id': segment['segment_id'], 'segment_position': segment['position'], 'segment_start': start, 'segment_end': end, 'segment_item_ids': segment_ids, 'boundary_scores': scores, 'pack_hash': hashlib.sha256(pack.encode()).hexdigest(), 'messages': messages, 'prompt': prompt})
    return (records, errors)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all-interactions', type=Path, required=True)
    parser.add_argument('--canonical-profile', type=Path, required=True)
    parser.add_argument('--embedding-dir', type=Path, required=True)
    parser.add_argument('--sft-reference', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--base-model', default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--lora', type=Path, required=True)
    parser.add_argument('--splits', default='train,valid,test')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--max-model-len', type=int, default=4096)
    parser.add_argument('--max-num-seqs', type=int, default=128)
    parser.add_argument('--max-num-batched-tokens', type=int, default=16384)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.92)
    parser.add_argument('--report-every', type=int, default=2048)
    parser.add_argument('--max-new-segments', type=int, default=0)
    parser.add_argument('--only-failures-dir', type=Path)
    parser.add_argument('--baseline-output-dir', type=Path)
    parser.add_argument('--json-object', action='store_true')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = [value.strip() for value in args.splits.split(',') if value.strip()]
    write_json(args.output_dir / 'run_config.json', {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
    items = {int(row['item_id']): row for row in read_jsonl(args.canonical_profile)}
    embeddings = base.load_embeddings(args.embedding_dir)
    if set(items) - set(embeddings):
        raise RuntimeError(f'missing embeddings: {len(set(items) - set(embeddings))}')
    system_prompt = first_row(args.sft_reference)['messages'][0]['content']
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import StructuredOutputsParams
    llm = LLM(model=args.base_model, trust_remote_code=True, dtype='bfloat16', max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_num_batched_tokens, gpu_memory_utilization=args.gpu_memory_utilization, enable_lora=True, disable_log_stats=True)
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens, structured_outputs=StructuredOutputsParams(json=state_schema()) if args.json_object else None)
    lora_request = LoRARequest('book_state_grounded_full', 1, str(args.lora))
    done = load_done(args.output_dir, splits)
    retry_keys = None
    retry_samples = None
    if args.only_failures_dir:
        if not args.baseline_output_dir:
            raise ValueError('--baseline-output-dir is required with --only-failures-dir')
        retry_keys = {(str(row['sample_id']), str(row['segment_id'])) for split in splits for row in read_jsonl(args.only_failures_dir / f'{split}_state_node_failures.jsonl')}
        retry_keys.difference_update(load_done(args.baseline_output_dir, splits))
        retry_samples = {sample_id for sample_id, _ in retry_keys}
    stats = Counter({'resume_done': len(done)})
    if retry_keys is not None:
        stats['retry_targets'] = len(retry_keys)
    started = time.time()
    batch = []

    def flush(rows):
        if not rows:
            return
        outputs = llm.generate([row['prompt'] for row in rows], sampling, lora_request=lora_request, use_tqdm=False)
        for row, result in zip(rows, outputs):
            raw = result.outputs[0].text if result.outputs else ''
            parsed = extract_json(raw)
            errors = ['json_parse_failed'] if parsed is None else validate_schema(parsed) + validate_repair_shape(parsed)
            repair_actions = {}
            repaired_text = raw
            if not errors:
                compact = json.dumps(parsed, ensure_ascii=False, separators=(',', ':'))
                repaired_text, repair_counts = repair_output(compact, row['messages'][1]['content'])
                repair_actions = dict(repair_counts)
                errors = validate_schema(json.loads(repaired_text))
                errors.extend(validate_grounding(row['record_id'], repaired_text, {'messages': row['messages']}))
            split = row['split']
            if errors:
                append_jsonl(args.output_dir / f'{split}_state_node_failures.jsonl', {'record_id': row['record_id'], 'sample_id': row['sample_id'], 'segment_id': row['segment_id'], 'pack_hash': row['pack_hash'], 'errors': sorted(set(errors)), 'raw_output': raw})
                stats[f'failure_{split}'] += 1
                continue
            model_output = json.loads(repaired_text)
            changed = repaired_text != json.dumps(parsed, ensure_ascii=False, separators=(',', ':'))
            append_jsonl(args.output_dir / f'{split}_state_nodes.jsonl', {key: value for key, value in row.items() if key not in {'messages', 'prompt'}} | {'model_output': model_output, 'grounding_repaired': changed, 'grounding_repair_actions': repair_actions, 'raw_output_hash': hashlib.sha256(raw.encode()).hexdigest()})
            stats[f'success_{split}'] += 1
            stats[f'repaired_{split}'] += int(changed)
        stats['attempted'] += len(rows)
        elapsed = max(time.time() - started, 1e-06)
        if stats['attempted'] % args.report_every < len(rows):
            print(json.dumps({'event': 'progress', 'stats': dict(stats), 'segments_per_sec': stats['attempted'] / elapsed}), flush=True)
        write_json(args.output_dir / 'progress.json', {'stats': dict(stats), 'elapsed_sec': elapsed})
    stop = False
    for source_row in read_jsonl(args.all_interactions):
        split = str(source_row.get('split'))
        if split not in splits or not source_row.get('history_item_ids'):
            continue
        if retry_samples is not None and str(source_row['sample_id']) not in retry_samples:
            continue
        stats[f'samples_seen_{split}'] += 1
        records, input_errors = make_records(source_row, items, embeddings, system_prompt, args.max_model_len, args.max_tokens, tokenizer)
        for error in input_errors:
            append_jsonl(args.output_dir / 'input_failures.jsonl', error)
            stats['input_failure'] += 1
        for row in records:
            key = (str(row['sample_id']), str(row['segment_id']))
            if retry_keys is not None and key not in retry_keys:
                continue
            if key in done:
                stats['skipped_done'] += 1
                continue
            if args.max_new_segments and stats['attempted'] + len(batch) >= args.max_new_segments:
                stop = True
                break
            batch.append(row)
            if len(batch) >= args.batch_size:
                flush(batch)
                batch = []
        if stop:
            break
    flush(batch)
    report = {'stats': dict(stats), 'elapsed_sec': time.time() - started, 'finished_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    write_json(args.output_dir / 'generation_report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
if __name__ == '__main__':
    main()
