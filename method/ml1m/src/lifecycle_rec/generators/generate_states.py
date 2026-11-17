from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any
os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER', '0')
os.environ.setdefault('VLLM_ATTENTION_BACKEND', 'FLASH_ATTN')
BROAD_NEGATIVE_TERMS = {'action', 'adventure', 'comedy', 'drama', 'romance', 'thriller', 'sci-fi', 'science fiction', 'fantasy', 'crime'}

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [jsonable(v) for v in obj]
    return obj

def stable_hash_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def extract_json(text: str) -> dict[str, Any] | None:
    text = (text or '').strip()
    if not text:
        return None
    if text.startswith('```'):
        text = text.strip('`')
        if '\n' in text:
            text = text.split('\n', 1)[1]
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    start = text.find('{')
    end = text.rfind('}')
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None

def validate_state(obj: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    node = obj.get('interest_state_node')
    if not isinstance(node, dict):
        return ['missing_interest_state_node']
    for key in ['state_type', 'state_label', 'state_summary', 'positive_analysis', 'negative_analysis', 'decision_cues']:
        if key not in node:
            errs.append(f'missing_{key}')
    pa = node.get('positive_analysis')
    na = node.get('negative_analysis')
    dc = node.get('decision_cues')
    if not isinstance(pa, dict):
        errs.append('positive_analysis_not_object')
    elif not isinstance(pa.get('liked_patterns', []), list):
        errs.append('liked_patterns_not_list')
    if not isinstance(na, dict):
        errs.append('negative_analysis_not_object')
    elif not isinstance(na.get('rejected_or_risky_patterns', []), list):
        errs.append('rejected_patterns_not_list')
    if not isinstance(dc, dict):
        errs.append('decision_cues_not_object')
    else:
        if not isinstance(dc.get('positive_match_cues', []), list):
            errs.append('positive_match_cues_not_list')
        if not isinstance(dc.get('negative_match_cues', []), list):
            errs.append('negative_match_cues_not_list')
        if not isinstance(dc.get('director_actor_cues', {}), dict):
            errs.append('director_actor_cues_not_object')
    return errs

def downweight_broad_negative_cues(obj: dict[str, Any]) -> dict[str, Any]:
    node = obj.get('interest_state_node')
    if not isinstance(node, dict):
        return obj
    dc = node.get('decision_cues')
    if not isinstance(dc, dict):
        return obj
    cues = dc.get('negative_match_cues')
    if not isinstance(cues, list):
        return obj
    kept: list[Any] = []
    low: list[Any] = []
    for cue in cues:
        norm = str(cue).strip().lower()
        if norm in BROAD_NEGATIVE_TERMS:
            low.append(cue)
        else:
            kept.append(cue)
    dc['negative_match_cues'] = kept
    if low:
        dc['low_priority_broad_negative_cues'] = low
    return obj

def render_prompt(tokenizer: Any, row: dict[str, Any]) -> str:
    messages = row['messages']
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def done_keys(path: Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for r in read_jsonl(path):
        keys.add((str(r.get('sample_id')), str(r.get('segment_id'))))
    return keys

def generate_split(args: argparse.Namespace, llm: Any, tokenizer: Any, sampling: Any, lora_request: Any, split: str) -> dict[str, Any]:
    msg_path = args.message_dir / f'{split}_state_segment_messages.jsonl'
    rows = read_jsonl(msg_path)
    out_path = args.output_dir / f'{split}_state_nodes.jsonl'
    fail_path = args.output_dir / f'{split}_state_node_failures.jsonl'
    raw_dir = args.output_dir / 'raw_outputs' / split
    raw_dir.mkdir(parents=True, exist_ok=True)
    done = done_keys(out_path)
    jobs = [r for r in rows if (str(r.get('sample_id')), str(r.get('segment_id'))) not in done]
    if args.limit:
        jobs = jobs[:args.limit]
    stats = {'split': split, 'input_segments': len(rows), 'already_done': len(rows) - len(jobs), 'pending_segments': len(jobs), 'success': 0, 'failure': 0, 'start': time.time()}
    print(json.dumps({'event': 'split_start', **stats}, ensure_ascii=False), flush=True)
    for offset in range(0, len(jobs), args.batch_size):
        batch = jobs[offset:offset + args.batch_size]
        prompts = [render_prompt(tokenizer, r) for r in batch]
        outputs = llm.generate(prompts, sampling, lora_request=lora_request)
        for row, out in zip(batch, outputs):
            raw = out.outputs[0].text if out.outputs else ''
            raw_hash = stable_hash_text(raw)
            parsed = extract_json(raw)
            errors = ['json_parse_failed'] if parsed is None else validate_state(parsed)
            if not errors and parsed is not None:
                post = downweight_broad_negative_cues(json.loads(json.dumps(parsed, ensure_ascii=False)))
                out_row = {'split': split, 'sample_id': row.get('sample_id'), 'user_id': row.get('user_id'), 'item_id': row.get('item_id'), 'segment_id': row.get('segment_id'), 'segment_position': row.get('segment_position'), 'source_label': row.get('source_label'), 'rating': row.get('rating'), 'history_recent30_len': row.get('history_recent30_len'), 'num_segments': row.get('num_segments'), 'pack_hash': row.get('pack_hash'), 'model': args.base_model, 'lora': str(args.lora), 'raw_output_hash': raw_hash, 'model_output': parsed, 'postprocessed_output': post}
                append_jsonl(out_path, out_row)
                stats['success'] += 1
            else:
                fail_row = {'split': split, 'sample_id': row.get('sample_id'), 'user_id': row.get('user_id'), 'item_id': row.get('item_id'), 'segment_id': row.get('segment_id'), 'pack_hash': row.get('pack_hash'), 'raw_output_hash': raw_hash, 'errors': errors, 'raw_output': raw}
                append_jsonl(fail_path, fail_row)
                stats['failure'] += 1
            if args.save_raw:
                (raw_dir / f"{row.get('sample_id')}__{row.get('segment_id')}__{raw_hash[:12]}.txt").write_text(raw, encoding='utf-8')
        done_new = min(offset + len(batch), len(jobs))
        write_json(args.output_dir / 'progress.json', {'current_split': split, 'done_new': done_new, 'stats': stats})
        if done_new % args.report_every == 0 or done_new == len(jobs):
            elapsed = max(1e-06, time.time() - stats['start'])
            print(json.dumps({'event': 'progress', 'split': split, 'done_new': done_new, 'pending_segments': len(jobs), 'success': stats['success'], 'failure': stats['failure'], 'segments_per_sec': done_new / elapsed}, ensure_ascii=False), flush=True)
    stats['elapsed_sec'] = time.time() - stats['start']
    return stats

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--message-dir', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--base-model', default='Qwen/Qwen2.5-7B-Instruct')
    ap.add_argument('--lora', type=Path, required=True)
    ap.add_argument('--splits', default='train,valid,test')
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--max-tokens', type=int, default=768)
    ap.add_argument('--max-model-len', type=int, default=4096)
    ap.add_argument('--gpu-memory-utilization', type=float, default=0.92)
    ap.add_argument('--max-num-seqs', type=int, default=128)
    ap.add_argument('--report-every', type=int, default=2048)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--save-raw', action='store_true')
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / 'run_config.json', jsonable(vars(args) | {'created_at': time.strftime('%Y-%m-%d %H:%M:%S')}))
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(model=args.base_model, trust_remote_code=True, dtype='bfloat16', max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs, gpu_memory_utilization=args.gpu_memory_utilization, enable_lora=True, disable_log_stats=True)
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens)
    lora_request = LoRARequest('states', 1, str(args.lora))
    reports = []
    for split in [s.strip() for s in args.splits.split(',') if s.strip()]:
        reports.append(generate_split(args, llm, tokenizer, sampling, lora_request, split))
    report = {'splits': reports, 'finished_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    write_json(args.output_dir / 'generation_report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
if __name__ == '__main__':
    main()
