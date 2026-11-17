from __future__ import annotations
import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from pathlib import Path
from typing import Any
from transformers import AutoTokenizer
from lifecycle_rec.teacher import transition_packs as edgev3
from lifecycle_rec.teacher import io as stategen

def summarize_logprobs(choice: dict[str, Any]) -> dict[str, Any]:
    import math
    lp = choice.get('logprobs') or {}
    content = lp.get('content') or []
    sampled = []
    entropies = []
    low_conf = 0
    for tok in content:
        val = tok.get('logprob')
        if isinstance(val, (int, float)):
            sampled.append(float(val))
            if val < -1.0:
                low_conf += 1
        tops = tok.get('top_logprobs') or []
        vals = [x.get('logprob') for x in tops if isinstance(x.get('logprob'), (int, float))]
        if vals:
            m = max(vals)
            ps = [math.exp(v - m) for v in vals]
            z = sum(ps)
            if z > 0:
                probs = [p / z for p in ps]
                entropies.append(-sum((p * math.log(max(p, 1e-12)) for p in probs)))
    return {'token_count_for_scoring': len(content), 'sampled_logprob_count': len(sampled), 'mean_sampled_token_logprob': sum(sampled) / len(sampled) if sampled else None, 'avg_token_nll': -sum(sampled) / len(sampled) if sampled else None, 'approx_topk_entropy_mean': sum(entropies) / len(entropies) if entropies else None, 'topk_entropy_count': len(entropies), 'low_conf_token_rate': low_conf / len(sampled) if sampled else None}

def request_completion(endpoint: str, model_name: str, messages: list[dict[str, str]], schema: dict[str, Any], max_tokens: int, temperature: float, timeout: int, seed: int, logprobs: bool=False, top_logprobs: int=0) -> tuple[str, dict[str, Any], dict[str, Any]]:
    payload = {'model': model_name, 'messages': messages, 'temperature': temperature, 'top_p': 1.0, 'max_tokens': max_tokens, 'seed': seed, 'response_format': {'type': 'json_schema', 'json_schema': {'name': 'edge_node_v3', 'schema': schema}}, 'guided_json': schema}
    if logprobs:
        payload['logprobs'] = True
        payload['top_logprobs'] = int(top_logprobs)
    data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(endpoint.rstrip('/') + '/chat/completions', data=data, headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode('utf-8'))
    choice = body['choices'][0]
    return (choice['message']['content'], body.get('usage') or {}, summarize_logprobs(choice))

def parse_json(raw: str) -> dict[str, Any]:
    normalized, _ = stategen.normalize_model_json_text(raw)
    return json.loads(normalized)

def identity(pack: dict[str, Any]) -> dict[str, Any]:
    return {'sample_id': pack.get('sample_id'), 'user_id': pack.get('user_id'), 'item_id': pack.get('item_id'), 'from_segment': pack.get('from_segment'), 'to_segment': pack.get('to_segment'), 'pack_hash': pack.get('pack_hash')}

def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')

def load_done(path: Path) -> set[str]:
    done = set()
    if not path.exists():
        return done
    for row in stategen.read_jsonl(path):
        key = f"{row.get('sample_id')}||{row.get('from_segment')}||{row.get('to_segment')}||{row.get('pack_hash')}"
        done.add(key)
    return done

def pack_key(pack: dict[str, Any]) -> str:
    return f"{pack.get('sample_id')}||{pack.get('from_segment')}||{pack.get('to_segment')}||{pack.get('pack_hash')}"

def run_one(pack: dict[str, Any], prompt: str, schema: dict[str, Any], args: argparse.Namespace, tokenizer) -> dict[str, Any]:
    messages = [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': pack['pack_text']}]
    attempts = []
    final_errors: list[str] = []
    raw_dir = args.output_dir / 'raw_edge_outputs'
    for attempt_id in range(1, args.max_attempts + 1):
        if attempt_id > 1:
            messages = [{'role': 'system', 'content': prompt + '\n\nRetry instruction: Your previous answer failed validation. Regenerate the full JSON, keep fields concise, keep negative signals non-causal, and do not use hard negative recommendation directives.'}, {'role': 'user', 'content': pack['pack_text']}]
        started = time.time()
        status = 'failed'
        failure_type = None
        raw_text = ''
        usage: dict[str, Any] = {}
        logprob_summary: dict[str, Any] = {}
        parsed: dict[str, Any] | None = None
        errors: list[str] = []
        warnings: list[str] = []
        try:
            raw_text, usage, logprob_summary = request_completion(args.endpoint, args.model_name, messages, schema, args.max_tokens, args.temperature, args.timeout, args.seed + attempt_id - 1, args.logprobs, args.top_logprobs)
            parsed = parse_json(raw_text)
            schema_errors, rule_errors, warnings = edgev3.validate_edge_node(parsed, pack, schema)
            errors = schema_errors + rule_errors
            if errors:
                failure_type = 'validation_error'
            else:
                status = 'success'
        except TimeoutError:
            failure_type = 'timeout'
            errors = ['timeout']
        except urllib.error.URLError as exc:
            failure_type = 'request_error'
            errors = [repr(exc)]
        except Exception as exc:
            failure_type = 'parse_or_request_error'
            errors = [repr(exc)]
        raw_name = f"{pack['sample_id']}__{pack['from_segment']}__{pack['to_segment']}__attempt{attempt_id}.txt"
        raw_path = raw_dir / raw_name
        raw_path.write_text(raw_text, encoding='utf-8')
        attempt = {**identity(pack), 'generation_variant': args.generation_variant, 'seed': args.seed + attempt_id - 1, 'attempt_id': attempt_id, 'status': status, 'failure_type': failure_type, 'errors': errors[:20], 'warnings': warnings[:20], 'raw_output_path': str(raw_path), 'prompt_token_count': usage.get('prompt_tokens') or pack.get('pack_token_count'), 'output_token_count': usage.get('completion_tokens') or (len(tokenizer.encode(raw_text)) if raw_text else 0), 'elapsed_sec': time.time() - started, 'model_name': args.model_name, 'logprob_summary': logprob_summary, 'logprob_config': {'logprobs': bool(args.logprobs), 'top_logprobs': int(args.top_logprobs)}}
        attempts.append(attempt)
        if status == 'success' and parsed is not None:
            return {'ok': True, 'attempts': attempts, 'node': {**identity(pack), 'generation_variant': args.generation_variant, 'edge_candidate_id': args.generation_variant, 'seed': args.seed + attempt_id - 1, 'model_name': args.model_name, 'model_output': parsed, 'logprob_summary': logprob_summary, 'validator_errors': [], 'validator_warnings': warnings, 'raw_output_path': str(raw_path), 'created_at': stategen.utc_now()}}
        final_errors = errors
    return {'ok': False, 'attempts': attempts, 'failure': {**identity(pack), 'generation_variant': args.generation_variant, 'edge_candidate_id': args.generation_variant, 'seed': args.seed, 'terminal_failure_type': attempts[-1].get('failure_type') if attempts else 'unknown', 'validation_errors': final_errors, 'attempts': attempts, 'failed_at': stategen.utc_now()}}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--packs', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--prompt', type=Path, required=True)
    ap.add_argument('--schema', type=Path, required=True)
    ap.add_argument('--endpoint', default='http://127.0.0.1:8010/v1')
    ap.add_argument('--model-name', default='Qwen2.5-72B-Instruct-AWQ')
    ap.add_argument('--tokenizer', default='Qwen/Qwen2.5-72B-Instruct-AWQ')
    ap.add_argument('--temperature', type=float, default=0.2)
    ap.add_argument('--max-tokens', type=int, default=1536)
    ap.add_argument('--timeout', type=int, default=300)
    ap.add_argument('--max-attempts', type=int, default=2)
    ap.add_argument('--parallel', type=int, default=2)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--seed', type=int, default=20260725)
    ap.add_argument('--generation-variant', default='E1')
    ap.add_argument('--logprobs', action='store_true')
    ap.add_argument('--top-logprobs', type=int, default=0)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'raw_edge_outputs').mkdir(parents=True, exist_ok=True)
    nodes_path = args.output_dir / 'transition_candidates.jsonl'
    attempts_path = args.output_dir / 'edge_generation_attempts.jsonl'
    failures_path = args.output_dir / 'transition_failures.jsonl'
    logprobs_path = args.output_dir / 'transition_logprobs.jsonl'
    for path in (nodes_path, attempts_path, failures_path, logprobs_path):
        path.touch(exist_ok=True)
    prompt = args.prompt.read_text(encoding='utf-8')
    schema = json.loads(args.schema.read_text(encoding='utf-8'))
    packs = stategen.read_jsonl(args.packs)
    if args.limit > 0:
        packs = packs[:args.limit]
    done = load_done(nodes_path) | load_done(failures_path)
    todo = [p for p in packs if pack_key(p) not in done]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    report = {'packs_total': len(packs), 'done_existing': len(packs) - len(todo), 'todo': len(todo), 'parallel': args.parallel, 'model_name': args.model_name, 'generation_variant': args.generation_variant, 'seed': args.seed, 'started_at': stategen.utc_now()}
    stategen.atomic_write_json(args.output_dir / 'edge_generation_run_report.json', report)
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = [ex.submit(run_one, pack, prompt, schema, args, tokenizer) for pack in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            result = fut.result()
            for attempt in result['attempts']:
                append_jsonl(attempts_path, attempt)
            if result['ok']:
                append_jsonl(nodes_path, result['node'])
                append_jsonl(logprobs_path, {**identity(result['node']), 'generation_variant': result['node'].get('generation_variant'), 'edge_candidate_id': result['node'].get('edge_candidate_id'), 'status': 'success', 'logprob_summary': result['node'].get('logprob_summary'), 'logprob_config': {'logprobs': bool(args.logprobs), 'top_logprobs': int(args.top_logprobs)}, 'raw_output_path': result['node'].get('raw_output_path'), 'model_name': args.model_name})
            else:
                append_jsonl(failures_path, result['failure'])
            if i == 1 or i % 20 == 0:
                nodes = stategen.read_jsonl(nodes_path)
                failures = stategen.read_jsonl(failures_path)
                attempts = stategen.read_jsonl(attempts_path)
                current = {**report, 'completed_now': i, 'edge_nodes': len(nodes), 'failures': len(failures), 'attempts': len(attempts), 'attempt_status_counts': dict(Counter((a.get('status') for a in attempts))), 'failure_type_counts': dict(Counter((f.get('terminal_failure_type') for f in failures))), 'updated_at': stategen.utc_now(), 'logprobs': bool(args.logprobs), 'top_logprobs': int(args.top_logprobs)}
                stategen.atomic_write_json(args.output_dir / 'edge_generation_run_report.json', current)
                print(json.dumps(current, ensure_ascii=False), flush=True)
    nodes = stategen.read_jsonl(nodes_path)
    failures = stategen.read_jsonl(failures_path)
    attempts = stategen.read_jsonl(attempts_path)
    final = {**report, 'edge_nodes': len(nodes), 'failures': len(failures), 'attempts': len(attempts), 'attempt_status_counts': dict(Counter((a.get('status') for a in attempts))), 'failure_type_counts': dict(Counter((f.get('terminal_failure_type') for f in failures))), 'finished_at': stategen.utc_now(), 'logprobs': bool(args.logprobs), 'top_logprobs': int(args.top_logprobs)}
    stategen.atomic_write_json(args.output_dir / 'edge_generation_run_report.json', final)
    print(json.dumps(final, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
