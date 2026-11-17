from __future__ import annotations

def main():
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', required=True, type=Path)
    parser.add_argument('--judge-dir', required=True, type=Path)
    parser.add_argument('--transition-repair-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--prompt', required=True, type=Path)
    parser.add_argument('--schema', required=True, type=Path)
    parser.add_argument('--tokenizer', required=True, type=Path)
    args = parser.parse_args()
    import json
    import os
    import re
    import math
    import hashlib
    import sys
    import time
    from pathlib import Path
    from lifecycle_rec.teacher import transition_packs as edgev3
    from lifecycle_rec.teacher import io as stategen
    from transformers import AutoTokenizer
    SRC = args.source_dir
    BASE_JUDGE = args.judge_dir
    EDGE_ONLY = args.transition_repair_dir
    OUT = args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    ANALYSIS = SRC / 'analysis_inputs'
    PROMPT = args.prompt
    SCHEMA = args.schema
    TOKENIZER = str(args.tokenizer)
    MAX_STATE_ATTEMPTS = 2

    def read_jsonl(p):
        if not Path(p).exists():
            return []
        return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]

    def keyseg(x):
        return int(re.sub('\\D', '', str(x)) or 0)

    def norm_value(vals, v):
        xs = [x for x in vals if isinstance(x, (int, float)) and math.isfinite(x)]
        if not xs or not isinstance(v, (int, float)) or (not math.isfinite(v)):
            return 0.0
        lo, hi = (min(xs), max(xs))
        return 0.0 if hi <= lo else max(0.0, min(1.0, (v - lo) / (hi - lo)))

    def uncertainty_from_summaries(summaries, s):
        return 0.5 * norm_value([x.get('avg_token_nll') for x in summaries], s.get('avg_token_nll')) + 0.3 * norm_value([x.get('approx_topk_entropy_mean') for x in summaries], s.get('approx_topk_entropy_mean')) + 0.2 * norm_value([x.get('low_conf_token_rate') for x in summaries], s.get('low_conf_token_rate'))
    base_judge = {d['sample_id']: d for d in read_jsonl(BASE_JUDGE / 'state_edge_results.jsonl')}
    edge_final = {d['sample_id']: d for d in read_jsonl(EDGE_ONLY / 'repair_final_results.jsonl')}
    remaining = [sid for sid, d in base_judge.items() if d.get('prediction') != d.get('label') and (not edge_final.get(sid, {}).get('final_correct'))]
    states = {}
    segs = {}
    for d in read_jsonl(SRC / 'state_candidates.jsonl'):
        sid = 'u%s_target_%s' % (d['user_id'], d['item_id'])
        states[sid, d['segment_id'], d['candidate_id']] = d
        segs.setdefault(sid, set()).add(d['segment_id'])
    state_lps = {}
    summaries = []
    for d in read_jsonl(SRC / 'state_logprobs.jsonl'):
        sid = 'u%s_target_%s' % (d['user_id'], d['item_id'])
        k = (sid, d['segment_id'], d['candidate_id'])
        s = d.get('logprob_summary') or {}
        state_lps[k] = s
        summaries.append(s)
    state_unc = {k: uncertainty_from_summaries(summaries, s) for k, s in state_lps.items()}

    def cand_status(row):
        c = (row.get('model_output') or {}).get('candidate') or {}
        return c.get('candidate_status')

    def selected_row(sid, seg, cid):
        row = states[sid, seg, cid]
        cand = (row.get('model_output') or {}).get('candidate') or {}
        return {'sample_id': sid, 'user_id': row.get('user_id'), 'item_id': row.get('item_id'), 'segment_id': seg, 'state_id': seg, 'selected_candidate': cand, 'candidate_id': cid, 'repair_selected_candidate_id': cid}
    transitions = []
    for d in read_jsonl(ANALYSIS / 'transition_packages.jsonl'):
        if d.get('sample_id') in remaining:
            transitions.append(d)
    trans_by_sample = {}
    for t in transitions:
        trans_by_sample.setdefault(t['sample_id'], []).append(t)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
    prompt_hash = stategen.sha256_file(PROMPT)
    schema_hash = stategen.sha256_file(SCHEMA)
    trans_hash = stategen.sha256_file(ANALYSIS / 'transition_packages.jsonl')
    attempts = []
    packs = []
    failures = []
    for sid in remaining:
        sample_segs = sorted(segs.get(sid, []), key=keyseg)
        q = []
        n = len(sample_segs)
        for i, seg in enumerate(sample_segs):
            cur = (sid, seg, 'C1')
            curu = state_unc.get(cur, 0.0)
            repl = []
            for cid in ['C2', 'C3']:
                k = (sid, seg, cid)
                if k not in states:
                    continue
                if cand_status(states[k]) != 'proposed':
                    continue
                ru = state_unc.get(k, 1.0)
                if ru <= curu + 0.05:
                    repl.append((cid, ru))
            if repl:
                repl = sorted(repl, key=lambda x: x[1])
                rec = 1.0 + (0.4 * i / max(1, n - 1) if n > 1 else 0.2)
                q.append({'seg': seg, 'old_uncertainty': curu, 'priority': curu * rec, 'replacement': repl[0]})
        q = sorted(q, key=lambda x: -x['priority'])[:MAX_STATE_ATTEMPTS]
        for aid, a in enumerate(q, 1):
            seg = a['seg']
            newcid, newu = a['replacement']
            attempt_id = f'{sid}::state_{seg}_{newcid}::attempt{aid}'
            attempts.append({'repair_attempt_key': attempt_id, 'sample_id': sid, 'attempt_id': aid, 'repair_action': 'state_replacement_with_edge_regeneration', 'target_node': seg, 'old_candidate_id': 'C1', 'new_candidate_id': newcid, 'old_uncertainty': a['old_uncertainty'], 'new_uncertainty': newu, 'priority': a['priority']})
            selected = {s: selected_row(sid, s, 'C1' if s != seg else newcid) for s in sample_segs}
            for tr in trans_by_sample.get(sid, []):
                fs = str(tr.get('from_segment'))
                ts = str(tr.get('to_segment'))
                if fs != seg and ts != seg:
                    continue
                ident = {'repair_attempt_key': attempt_id, 'sample_id': sid, 'user_id': tr.get('user_id'), 'item_id': tr.get('item_id'), 'from_segment': fs, 'to_segment': ts}
                try:
                    from_state = edgev3.state_core_payload(selected[fs])
                    to_state = edgev3.state_core_payload(selected[ts])
                    analysis = edgev3.build_transition_analysis(tr, from_state, to_state)
                    if not any(analysis['positive_signal_comparison'].values()) and (not any(analysis['negative_caution_comparison'].values())):
                        failures.append({**ident, 'failure_type': 'insufficient_transition_analysis'})
                        continue
                    row = {**ident, 'edge_task': {'edge_id': f'{fs}->{ts}', 'from_state_id': fs, 'to_state_id': ts, 'expected_output': 'Analyze positive and negative transition updates between adjacent states.'}, 'from_state_node': from_state, 'to_state_node': to_state, 'transition_analysis_data': analysis, 'source_file_hashes': {'transition_packages': trans_hash, 'selected_state_nodes': 'state_repair_dynamic', 'prompt': prompt_hash, 'schema': schema_hash}}
                    text = edgev3.build_pack_text(row)
                    row['pack_text'] = text
                    row['pack_token_count'] = len(tokenizer.encode(text))
                    row['pack_hash'] = stategen.sha256_bytes(text.encode('utf-8'))
                    packs.append(row)
                except Exception as e:
                    failures.append({**ident, 'failure_type': 'pack_build_exception', 'error': repr(e)})
    stategen.atomic_write_jsonl(OUT / 'state_repair_action_queue.jsonl', attempts)
    stategen.atomic_write_jsonl(OUT / 'repair_transition_packs.jsonl', packs)
    stategen.atomic_write_jsonl(OUT / 'repair_transition_failures.jsonl', failures)
    report = {'remaining_after_edge_only': len(remaining), 'state_repair_attempts': len(attempts), 'edge_regeneration_packs': len(packs), 'edge_regeneration_pack_failures': len(failures), 'max_state_attempts_per_sample': MAX_STATE_ATTEMPTS, 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S')}
    stategen.atomic_write_json(OUT / 'state_repair_pack_build_report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
