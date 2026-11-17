from __future__ import annotations
import argparse
import concurrent.futures as cf
import copy
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from lifecycle_rec.teacher import transition_support as edgev3
from lifecycle_rec.teacher import transition_client as base

def read(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(x) for x in f if x.strip()]

def append_jsonl(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    handle.flush()

def record_key(row, variant=None):
    variant = variant or row.get('generation_variant')
    pack_hash = row.get('pack_hash')
    if pack_hash:
        return (variant, pack_hash)
    return (variant, row.get('sample_id'), row.get('from_segment'), row.get('to_segment'))

def completed_keys(path):
    if not path.exists():
        return set()
    keys = set()
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                keys.add(record_key(row))
    return keys

def write_report(path, report):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)

def stable(values):
    out, seen = ([], set())
    for value in values:
        value = str(value).strip()
        key = value.casefold()
        if value and key != 'none' and (key not in seen):
            out.append(value)
            seen.add(key)
    return out

def dynamic_schema(base_schema, pack, variant):
    schema = copy.deepcopy(base_schema)
    a = pack['transition_analysis_data']
    pos = a['positive_signal_comparison']
    neg = a['negative_caution_comparison']
    conflicts = stable(a.get('polarity_conflict_terms') or [])
    positive = {k: stable(pos.get(f'{k}_positive_terms') or []) for k in ('retained', 'emerging', 'fading')}
    negative = {k: stable(neg.get(f'{k}_negative_terms') or []) for k in ('retained', 'emerging', 'fading')}
    authors = a['author_signal_comparison']
    pa = authors['positive_authors']
    na = authors['negative_caution_authors']

    def enum_items(node, values, max_items):
        values = stable(values)
        node['maxItems'] = min(max_items, len(values))
        node['minItems'] = 1 if values else 0
        node['items'] = {'type': 'string', 'enum': values or ['__no_supported_signal__'], 'maxLength': 140}
    node = schema['properties']['edge_node']['properties']
    positive_caps = {'E1': 3, 'E2': 4, 'E3': 2}
    negative_caps = {'E1': 2, 'E2': 2, 'E3': 3}
    pos_cap = positive_caps[variant]
    neg_cap = negative_caps[variant]
    for phase in ('retained', 'emerging', 'fading'):
        definition = 'fading_signal' if phase == 'fading' else 'positive_signal'
        name = f'positive_{phase}_signal'
        schema['definitions'][name] = copy.deepcopy(schema['definitions'][definition])
        schema['definitions'][name]['properties']['signal'] = {'type': 'string', 'enum': positive[phase], 'maxLength': 140} if positive[phase] else {'type': 'string', 'enum': ['__no_supported_signal__']}
        arr = node['positive_transition']['properties'][phase]
        arr['maxItems'] = min(arr['maxItems'], len(positive[phase]), pos_cap)
        arr['minItems'] = 1 if positive[phase] else 0
        arr['items'] = {'$ref': f'#/definitions/{name}'}
        name = f'negative_{phase}_signal'
        schema['definitions'][name] = copy.deepcopy(schema['definitions']['negative_signal'])
        schema['definitions'][name]['properties']['signal'] = {'type': 'string', 'enum': negative[phase], 'maxLength': 140} if negative[phase] else {'type': 'string', 'enum': ['__no_supported_signal__']}
        arr = node['negative_caution_transition']['properties'][phase]
        arr['maxItems'] = min(arr['maxItems'], len(negative[phase]), neg_cap)
        arr['minItems'] = 1 if negative[phase] else 0
        arr['items'] = {'$ref': f'#/definitions/{name}'}
    conflict_array = node['polarity_conflicts']
    conflict_array['maxItems'] = min(conflict_array['maxItems'], len(conflicts))
    conflict_array['minItems'] = 1 if conflicts else 0
    conflict_array['items']['properties']['signal'] = {'type': 'string', 'enum': conflicts or ['__no_supported_signal__'], 'maxLength': 140}
    author_node = node['author_signal_transition']['properties']
    enum_items(author_node['retained_positive_authors'], pa['retained'], 3)
    enum_items(author_node['emerging_positive_authors'], pa['emerging'], 3)
    enum_items(author_node['negative_caution_authors'], na['retained'] + na['emerging'] + na['fading'], 3)
    author_values = stable(pa['retained'] + pa['emerging'] + na['retained'] + na['emerging'] + na['fading'])
    if author_values:
        author_node['analysis'] = {'type': 'string', 'enum': ['Supported author cues are listed from the adjacent state outputs; they remain local signals only.']}
    else:
        author_node['analysis'] = {'type': 'string', 'enum': ['No supported author cue is present in either adjacent state output.']}
    update = node['recommendation_update']['properties']
    enum_items(update['long_term_positive_anchors'], positive['retained'], pos_cap)
    enum_items(update['recent_positive_focus'], positive['emerging'], pos_cap)
    enum_items(update['soft_caution_signals'], negative['retained'] + negative['emerging'], neg_cap)
    enum_items(update['do_not_over_penalize'], conflicts + negative['retained'] + negative['emerging'] + positive['fading'], max(pos_cap, neg_cap))
    return schema
PROSE_KEYS = {'analysis', 'why', 'conflict', 'transition_summary', 'transition_label'}
HARD = re.compile('\\b(?:avoid\\w*|reject\\w*|dislik\\w*|hat(?:e|ed|es|ing)|downweight|penalize|exclude|do not recommend|strong (?:preference|interest)|consistent preference|preferences? (?:strengthened|weakened|increased|decreased|changed)|interests? (?:strengthened|weakened|increased|decreased|changed)|reduced interest|increased interest|have weakened|has weakened)\\b', re.I)

def prose_errors(output):
    bad = []

    def walk(value, key=''):
        if isinstance(value, dict):
            for k, v in value.items():
                walk(v, k)
        elif isinstance(value, list):
            for v in value:
                walk(v, key)
        elif isinstance(value, str) and key in PROSE_KEYS and HARD.search(value):
            bad.append(f'book_overstated_feedback:{key}:{HARD.search(value).group(0)}')
    walk(output)
    return bad

def run_one(pack, prompt, base_schema, args):
    schema = dynamic_schema(base_schema, pack, args.generation_variant)
    messages = [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': pack['pack_text']}]
    attempts = []
    for attempt in range(1, args.max_attempts + 1):
        try:
            raw, usage, lp = base.request_completion(args.endpoint, args.model_name, messages, schema, args.max_tokens, args.temperature, args.timeout, args.seed + attempt - 1, True, args.top_logprobs)
            parsed = base.parse_json(raw)
            se, re_errors, warnings = edgev3.validate_edge_node(parsed, pack, schema)
            errors = se + re_errors + prose_errors(parsed)
        except Exception as exc:
            raw, usage, lp, parsed, warnings = ('', {}, {}, None, [])
            errors = [repr(exc)]
        attempts.append({'attempt': attempt, 'errors': errors, 'raw_output': raw, 'usage': usage, 'logprob_summary': lp})
        if not errors:
            return {'ok': True, 'row': {**base.identity(pack), 'generation_variant': args.generation_variant, 'edge_candidate_id': args.generation_variant, 'seed': args.seed + attempt - 1, 'model_name': args.model_name, 'model_output': parsed, 'logprob_summary': lp, 'usage': usage, 'validator_errors': [], 'validator_warnings': warnings, 'attempts_used': attempt}, 'attempts': attempts}
        messages = [{'role': 'system', 'content': prompt + '\nRetry: regenerate the complete JSON. Use only schema-enumerated source signals and neutral local-caution wording.'}, {'role': 'user', 'content': pack['pack_text']}]
    return {'ok': False, 'failure': {**base.identity(pack), 'generation_variant': args.generation_variant, 'errors': attempts[-1]['errors']}, 'attempts': attempts}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--packs', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--prompt', type=Path, required=True)
    ap.add_argument('--schema', type=Path, required=True)
    ap.add_argument('--endpoint', default='http://127.0.0.1:8010/v1')
    ap.add_argument('--model-name', default='Qwen2.5-72B-Instruct-AWQ')
    ap.add_argument('--generation-variant', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--parallel', type=int, default=10)
    ap.add_argument('--temperature', type=float, default=0.2)
    ap.add_argument('--max-tokens', type=int, default=1536)
    ap.add_argument('--timeout', type=int, default=300)
    ap.add_argument('--max-attempts', type=int, default=3)
    ap.add_argument('--top-logprobs', type=int, default=5)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    packs = read(args.packs)
    prompt = args.prompt.read_text(encoding='utf-8')
    schema = json.loads(args.schema.read_text())
    nodes_path = args.output_dir / 'transition_candidates.jsonl'
    attempts_path = args.output_dir / 'edge_generation_attempts_v3.jsonl'
    failures_path = args.output_dir / 'edge_generation_failures_v3.jsonl'
    report_path = args.output_dir / 'edge_generation_v3_run_report.json'
    existing = completed_keys(nodes_path)
    todo = [pack for pack in packs if record_key(pack, args.generation_variant) not in existing]
    started = time.time()
    success = failures = attempts = 0

    def report(status):
        return {'status': status, 'packs_total': len(packs), 'done_existing': len(packs) - len(todo), 'todo_at_start': len(todo), 'processed_this_run': success + failures, 'success_this_run': success, 'failures_this_run': failures, 'attempts_this_run': attempts, 'completed_total': len(existing) + success, 'remaining': len(todo) - success - failures, 'variant': args.generation_variant, 'seed': args.seed, 'parallel': args.parallel, 'elapsed_sec': time.time() - started, 'dynamic_source_term_enums': True, 'top_logprobs': args.top_logprobs, 'updated_at': datetime.now(timezone.utc).isoformat()}
    write_report(report_path, report('running'))
    with nodes_path.open('a', encoding='utf-8') as fn, attempts_path.open('a', encoding='utf-8') as fa, failures_path.open('a', encoding='utf-8') as ff, cf.ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futures = [ex.submit(run_one, pack, prompt, schema, args) for pack in todo]
        for future in cf.as_completed(futures):
            result = future.result()
            identity = result.get('row') or result.get('failure')
            for attempt in result['attempts']:
                append_jsonl(fa, {**{k: identity.get(k) for k in ('sample_id', 'from_segment', 'to_segment', 'generation_variant', 'pack_hash')}, **attempt})
                attempts += 1
            if result['ok']:
                append_jsonl(fn, result['row'])
                success += 1
            else:
                append_jsonl(ff, result['failure'])
                failures += 1
            if (success + failures) % 25 == 0:
                write_report(report_path, report('running'))
    final_status = 'complete' if failures == 0 else 'complete_with_failures'
    report = report(final_status)
    write_report(report_path, report)
    print(json.dumps(report, indent=2))
