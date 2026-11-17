from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
from typing import Any
from lifecycle_rec.generators import transition_inputs as edge_pipe

def render_prompt(tokenizer: Any, job: dict[str, Any], retry_errors: list[str] | None=None) -> str:
    user = edge_pipe.EDGE_RUNTIME_USER_INSTRUCTION
    if retry_errors:
        user += '\nYour previous answer failed validation: ' + ', '.join(retry_errors) + '. Regenerate complete JSON only. Keep from_state_id/to_state_id exact. ' + 'Use neutral non-causal caution wording. ' + 'Do not use forbidden negative directive words in any field: ' + 'avoid, downweight, reject, penalize, less suitable, rank lower, deprioritize.\n'
    user += '\n\n' + job['pack_text']
    messages = [{'role': 'system', 'content': edge_pipe.EDGE_RUNTIME_SYSTEM_PROMPT}, {'role': 'user', 'content': user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def parse_and_validate(raw: str, job: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    parsed = edge_pipe.extract_json(raw)
    if parsed is None:
        return (None, ['json_parse_failed'])
    parsed = edge_pipe.minimal_structural_repair(parsed)
    errs = edge_pipe.edge_sft.validate_output(parsed, job['from_segment'], job['to_segment'])
    if errs:
        return (parsed, errs)
    return (parsed, [])

def output_row(job: dict[str, Any], raw: str, parsed: dict[str, Any], attempt: int) -> dict[str, Any]:
    base = {k: job[k] for k in ['sample_id', 'split', 'user_id', 'item_id', 'from_segment', 'to_segment', 'edge_id', 'pack_hash']}
    return {**base, 'attempt': attempt, 'raw_output': raw, 'edge': parsed['edge_node']}

def failure_row(job: dict[str, Any], raw: str, errors: list[str]) -> dict[str, Any]:
    base = {k: job[k] for k in ['sample_id', 'split', 'user_id', 'item_id', 'from_segment', 'to_segment', 'edge_id', 'pack_hash']}
    return {**base, 'raw_output': raw, 'errors': errors}

def load_jobs(args: argparse.Namespace, split: str) -> list[dict[str, Any]]:
    state_rows = edge_pipe.read_jsonl(args.generated_dir / f'{split}_state_nodes.jsonl')
    jobs = edge_pipe.make_edge_jobs(state_rows)
    done = {(r['sample_id'], r['from_segment'], r['to_segment']) for r in edge_pipe.read_jsonl(args.edge_output_dir / f'{split}_edge_mintrim_outputs.jsonl')}
    jobs = [j for j in jobs if (j['sample_id'], j['from_segment'], j['to_segment']) not in done]
    if args.limit:
        jobs = jobs[:args.limit]
    return jobs

def generate_split(args: argparse.Namespace, llm: Any, tokenizer: Any, sampling: Any, lora_request: Any, split: str) -> dict[str, Any]:
    jobs = load_jobs(args, split)
    out_path = args.edge_output_dir / f'{split}_edge_mintrim_outputs.jsonl'
    fail_path = args.edge_output_dir / f'{split}_edge_mintrim_failures.jsonl'
    stats = {'split': split, 'pending_edges': len(jobs), 'success': 0, 'failure': 0, 'start': time.time()}
    print(json.dumps({'event': 'split_start', **stats}, ensure_ascii=False), flush=True)
    for offset in range(0, len(jobs), args.batch_size):
        batch = jobs[offset:offset + args.batch_size]
        prompts = [render_prompt(tokenizer, j) for j in batch]
        outputs = llm.generate(prompts, sampling, lora_request=lora_request)
        retry_jobs: list[dict[str, Any]] = []
        retry_errs: list[list[str]] = []
        retry_raws: list[str] = []
        for job, out in zip(batch, outputs):
            raw = out.outputs[0].text if out.outputs else ''
            parsed, errs = parse_and_validate(raw, job)
            if not errs and parsed is not None:
                edge_pipe.append_jsonl(out_path, output_row(job, raw, parsed, 1))
                stats['success'] += 1
            else:
                retry_jobs.append(job)
                retry_errs.append(errs)
                retry_raws.append(raw)
        if retry_jobs and args.max_attempts >= 2:
            retry_prompts = [render_prompt(tokenizer, j, e) for j, e in zip(retry_jobs, retry_errs)]
            retry_outputs = llm.generate(retry_prompts, sampling, lora_request=lora_request)
            for job, prev_raw, prev_err, out in zip(retry_jobs, retry_raws, retry_errs, retry_outputs):
                raw = out.outputs[0].text if out.outputs else ''
                parsed, errs = parse_and_validate(raw, job)
                if not errs and parsed is not None:
                    edge_pipe.append_jsonl(out_path, output_row(job, raw, parsed, 2))
                    stats['success'] += 1
                else:
                    edge_pipe.append_jsonl(fail_path, failure_row(job, raw or prev_raw, errs or prev_err))
                    stats['failure'] += 1
        elif retry_jobs:
            for job, raw, errs in zip(retry_jobs, retry_raws, retry_errs):
                edge_pipe.append_jsonl(fail_path, failure_row(job, raw, errs))
                stats['failure'] += 1
        done_new = min(offset + len(batch), len(jobs))
        if done_new % args.report_every == 0 or done_new == len(jobs):
            elapsed = max(1e-06, time.time() - stats['start'])
            print(json.dumps({'event': 'progress', 'split': split, 'done_new': done_new, 'pending_new': len(jobs), 'success': stats['success'], 'failure': stats['failure'], 'edge_per_sec': done_new / elapsed}, ensure_ascii=False), flush=True)
    return stats
