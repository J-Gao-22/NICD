from __future__ import annotations

def main():
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', required=True, type=Path)
    parser.add_argument('--judge-dir', required=True, type=Path)
    parser.add_argument('--transition-repair-dir', required=True, type=Path)
    parser.add_argument('--state-repair-dir', required=True, type=Path)
    parser.add_argument('--state-judgment-dir', required=True, type=Path)
    parser.add_argument('--state-output-dir', required=True, type=Path)
    parser.add_argument('--transition-output-dir', required=True, type=Path)
    parser.add_argument('--state-prompt', required=True, type=Path)
    parser.add_argument('--state-packs', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    import json
    import os
    import re
    import random
    import hashlib
    import time
    from pathlib import Path
    from collections import Counter
    SRC = args.source_dir
    JUDGE = args.judge_dir
    EDGE_ONLY = args.transition_repair_dir
    STATE_REPAIR = args.state_repair_dir
    STATE_EVAL = args.state_judgment_dir
    STATE_OUT = args.state_output_dir
    EDGE_OUT = args.transition_output_dir
    STATE_PROMPT = args.state_prompt
    EDGE_PROMPT_INSTRUCTION = 'Generate the compact edge node from this input pack.\nKeep from_state_id and to_state_id exactly.\nUse concise transition analysis.\nretained/emerging/fading arrays should contain semantic signal strings, not objects.\nOnly include polarity_conflicts when explicitly supported; otherwise use an empty list.'
    SCHEMA_EDGE = 'lifecycle'

    def read_jsonl(p: Path):
        if not p.exists():
            return []
        return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]

    def write_jsonl(p: Path, rows):
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(p) + '.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n')
        tmp.replace(p)

    def write_json(p: Path, obj):
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(p) + '.tmp')
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        tmp.replace(p)

    def sha(x):
        text = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(text.encode()).hexdigest()

    def canon(x):
        return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

    def segnum(s):
        return int(re.sub('\\D', '', str(s)) or 0)

    def short(x, n):
        return ' '.join(str(x or '').strip().split()[:n])

    def unique_strings(xs, limit):
        out = []
        seen = set()
        for x in xs or []:
            if isinstance(x, dict):
                s = x.get('signal') or x.get('term') or x.get('value') or x.get('pattern')
            else:
                s = x
            s = short(s, 8)
            if not s:
                continue
            k = s.lower()
            if k in seen:
                continue
            out.append(s)
            seen.add(k)
            if len(out) >= limit:
                break
        return out

    def mintrim_output(cand):
        src = (cand.get('model_output') or {}).get('edge_node') or cand.get('edge_node') or {}
        pos = src.get('positive_transition') or {}
        neg = src.get('negative_caution_transition') or {}
        return {'edge_node': {'from_state_id': str(src.get('from_state_id') or cand.get('from_segment') or ''), 'to_state_id': str(src.get('to_state_id') or cand.get('to_segment') or ''), 'transition_label': short(src.get('transition_label'), 12), 'transition_type': src.get('transition_type') or 'mixed_update', 'positive_transition': {'analysis': short(pos.get('analysis'), 45), 'retained': unique_strings(pos.get('retained'), 8), 'emerging': unique_strings(pos.get('emerging'), 8), 'fading': unique_strings(pos.get('fading'), 8)}, 'negative_caution_transition': {'analysis': short(neg.get('analysis'), 45), 'retained': unique_strings(neg.get('retained'), 8), 'emerging': unique_strings(neg.get('emerging'), 8), 'fading': unique_strings(neg.get('fading'), 8)}, 'polarity_conflicts': unique_strings(src.get('polarity_conflicts'), 8), 'transition_summary': short(src.get('transition_summary'), 60)}}

    def split_records(records, key_field='split_group_key', seed=2026):
        keys = sorted({r[key_field] for r in records})
        random.Random(seed).shuffle(keys)
        n = len(keys)
        ntr = int(n * 0.8)
        nv = int(n * 0.1)
        split = {k: 'train' if i < ntr else 'valid' if i < ntr + nv else 'test' for i, k in enumerate(keys)}
        out = {'train': [], 'valid': [], 'test': []}
        for r in records:
            rr = dict(r)
            rr['split'] = split[r[key_field]]
            out[rr['split']].append(rr)
        return out

    def file_sha(p):
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            for b in iter(lambda: f.read(1024 * 1024), b''):
                h.update(b)
        return h.hexdigest()

    def sums(dirp):
        with open(dirp / 'SHA256SUMS', 'w', encoding='utf-8') as f:
            for p in sorted(dirp.iterdir()):
                if p.is_file() and p.name != 'SHA256SUMS':
                    f.write(f'{file_sha(p)}  {p.name}\n')
    base = {d['sample_id']: d for d in read_jsonl(JUDGE / 'state_edge_results.jsonl')}
    edge_final = {d['sample_id']: d for d in read_jsonl(EDGE_ONLY / 'repair_final_results.jsonl')}
    edge_success = {d['sample_id']: d for d in read_jsonl(EDGE_ONLY / 'repair_attempts.jsonl') if d.get('success')}
    state_final = {d['sample_id']: d for d in read_jsonl(STATE_EVAL / 'state_repair_final_results.jsonl')}
    state_success = {d['sample_id']: d for d in read_jsonl(STATE_EVAL / 'state_repair_attempts_14b.jsonl') if d.get('success')}
    final_samples = {}
    for sid, d in base.items():
        if d.get('correct') is True:
            final_samples[sid] = {'status': 'base_correct', 'label': d.get('label'), 'pred': d.get('prediction')}
    for sid, d in edge_final.items():
        if d.get('final_correct') is True:
            final_samples[sid] = {'status': 'edge_repaired_success', 'label': d.get('label'), 'pred': d.get('final_prediction')}
    for sid, d in state_final.items():
        if d.get('final_correct') is True:
            final_samples[sid] = {'status': 'state_repaired_success', 'label': d.get('label'), 'pred': d.get('final_prediction')}
    state_rows = {}
    segs = {}
    pack_files = sorted(set(args.state_packs))
    for d in read_jsonl(SRC / 'state_candidates.jsonl'):
        sid = f"u{d['user_id']}_target_{d['item_id']}"
        state_rows[sid, d['segment_id'], d['candidate_id']] = d
        segs.setdefault(sid, set()).add(d['segment_id'])
    pack_index = {}
    pack_by_simple = {}
    for pf in pack_files:
        for p in read_jsonl(Path(pf)):
            sid = f"u{p['user_id']}_target_{p['item_id']}"
            key = (sid, p['segment_id'], p['candidate_id'], p['pack_hash'])
            pack_index[key] = p
            key_simple = (sid, p['segment_id'], p['candidate_id'])
            if sha(p['pack_text']) != p['pack_hash']:
                raise ValueError(('State pack hash mismatch', key_simple))
            previous_pack = pack_by_simple.get(key_simple)
            if previous_pack is not None and previous_pack['pack_hash'] != p['pack_hash']:
                raise ValueError(('Ambiguous state pack', key_simple))
            pack_by_simple[key_simple] = p
    state_system = STATE_PROMPT.read_text(encoding='utf-8')

    def selected_state_cid(sid, seg):
        fs = final_samples[sid]
        if fs['status'] == 'state_repaired_success':
            a = state_success.get(sid)
            if a and a.get('target_node') == seg:
                return a.get('new_candidate_id')
        return 'C1'

    def interest_node_from_candidate(cand):
        c = (cand.get('model_output') or {}).get('candidate') or {}
        node = {k: v for k, v in c.items() if k not in {'candidate_id', 'analysis_lens', 'candidate_status', 'insufficient_reason'}}
        return {'interest_state_node': node}
    state_records = []
    state_fail = []
    for sid, fs in sorted(final_samples.items()):
        for seg in sorted(segs.get(sid, []), key=segnum):
            cid = selected_state_cid(sid, seg)
            cand = state_rows.get((sid, seg, cid))
            if not cand:
                state_fail.append({'sample_id': sid, 'segment_id': seg, 'candidate_id': cid, 'reason': 'missing_candidate'})
                continue
            c = (cand.get('model_output') or {}).get('candidate') or {}
            if c.get('candidate_status') != 'proposed':
                state_fail.append({'sample_id': sid, 'segment_id': seg, 'candidate_id': cid, 'reason': 'candidate_not_proposed'})
                continue
            pack = pack_index.get((sid, seg, cid, cand['pack_hash'])) if cand.get('pack_hash') else pack_by_simple.get((sid, seg, cid))
            if not pack or not pack.get('pack_text'):
                state_fail.append({'sample_id': sid, 'segment_id': seg, 'candidate_id': cid, 'reason': 'missing_pack'})
                continue
            out = interest_node_from_candidate(cand)
            ih = sha(pack['pack_text'])
            oh = sha(out)
            state_records.append({'record_id': f'{sid}::{seg}::{cid}::{ih[:12]}', 'sample_id': sid, 'user_id': cand.get('user_id'), 'segment_id': seg, 'split_group_key': pack.get('pack_hash') or ih, 'source_label': fs['label'], 'teacher_recommendation_pred': fs['pred'], 'pack_hash': pack.get('pack_hash'), 'input_hash': ih, 'output_hash': oh, 'messages': [{'role': 'system', 'content': state_system}, {'role': 'user', 'content': pack['pack_text']}, {'role': 'assistant', 'content': canon(out)}], 'quality': {'final_source_status': fs['status'], 'selected_candidate_id': cid, 'sample_weight': 1.0}, 'provenance': {'teacher_model': 'Qwen2.5-72B-Instruct-AWQ', 'source_candidate_file': cand.get('complete_source_file') or cand.get('source_candidate_file'), 'source_pack_file': cand.get('source_pack_file'), 'repair_source': fs['status']}})
    splits = split_records(state_records)
    STATE_OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(STATE_OUT / 'segment_interest_node_sft_all.jsonl', state_records)
    for sp, rows in splits.items():
        write_jsonl(STATE_OUT / f'segment_interest_node_sft_{sp}.jsonl', rows)
    write_jsonl(STATE_OUT / 'segment_interest_node_sft_conflicts.jsonl', state_fail)
    write_json(STATE_OUT / 'segment_interest_node_sft_report.json', {'records': len(state_records), 'failures': len(state_fail), 'sample_count': len({r['sample_id'] for r in state_records}), 'split_counts': {k: len(v) for k, v in splits.items()}, 'final_sample_status_counts': dict(Counter((x['status'] for x in final_samples.values()))), 'candidate_counts': dict(Counter((r['quality']['selected_candidate_id'] for r in state_records))), 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S')})
    write_json(STATE_OUT / 'segment_interest_node_sft_config.json', {'source': 'selected_teacher_samples', 'system_prompt_source': str(STATE_PROMPT), 'teacher_model': 'Qwen2.5-72B-Instruct-AWQ'})
    sums(STATE_OUT)
    old_edges = {}
    old_packs = {}
    for d in read_jsonl(SRC / 'transition_candidates.jsonl'):
        eid = d.get('edge_candidate_id') or d.get('generation_variant')
        old_edges[d['sample_id'], d['from_segment'], d['to_segment'], eid] = d
    for p in read_jsonl(SRC / 'transition_packs.jsonl'):
        old_packs[p['sample_id'], p['from_segment'], p['to_segment']] = p
    regen_pack_by_attempt = {}
    for p in read_jsonl(STATE_REPAIR / 'repair_transition_packs.jsonl'):
        regen_pack_by_attempt[p.get('repair_attempt_key'), p.get('from_segment'), p.get('to_segment')] = p
    regen_edge_by_hash = {}
    for root in [STATE_REPAIR / 'transitions']:
        for d in read_jsonl(root / 'transition_candidates.jsonl'):
            regen_edge_by_hash[d.get('pack_hash')] = d

    def final_edges_for(sid):
        sample_segs = sorted(segs.get(sid, []), key=segnum)
        fs = final_samples[sid]
        edge_repl = {}
        if fs['status'] == 'edge_repaired_success':
            a = edge_success.get(sid)
            if a and a.get('target_node'):
                parts = str(a['target_node']).split('->')
                if len(parts) == 2:
                    edge_repl[parts[0], parts[1]] = a.get('new_candidate_id')
        state_action = state_success.get(sid) if fs['status'] == 'state_repaired_success' else None
        rows = []
        for i in range(len(sample_segs) - 1):
            fr, to = (sample_segs[i], sample_segs[i + 1])
            if state_action and (fr == state_action.get('target_node') or to == state_action.get('target_node')):
                attempt_key = state_action.get('repair_attempt_key')
                pack = regen_pack_by_attempt.get((attempt_key, fr, to))
                edge = regen_edge_by_hash.get(pack.get('pack_hash')) if pack else None
                slot = 'E1_regen'
            else:
                slot = edge_repl.get((fr, to), 'E1')
                edge = old_edges.get((sid, fr, to, slot))
                pack = old_packs.get((sid, fr, to))
            rows.append((fr, to, slot, pack, edge))
        return rows
    edge_records = []
    edge_fail = []
    for sid, fs in sorted(final_samples.items()):
        for fr, to, slot, pack, edge in final_edges_for(sid):
            if not pack or not pack.get('pack_text'):
                edge_fail.append({'sample_id': sid, 'from_segment': fr, 'to_segment': to, 'slot': slot, 'reason': 'missing_pack'})
                continue
            if not edge:
                edge_fail.append({'sample_id': sid, 'from_segment': fr, 'to_segment': to, 'slot': slot, 'reason': 'missing_edge'})
                continue
            out = mintrim_output(edge)
            node = out.get('edge_node') or {}
            if node.get('from_state_id') != fr or node.get('to_state_id') != to:
                edge_fail.append({'sample_id': sid, 'from_segment': fr, 'to_segment': to, 'slot': slot, 'reason': 'state_id_mismatch'})
                continue
            inp = {'instruction': EDGE_PROMPT_INSTRUCTION, 'pack_text': pack['pack_text']}
            ih = sha(inp)
            oh = sha(out)
            edge_records.append({'record_id': f'{sid}::{fr}_to_{to}::{slot}::{ih[:12]}', 'sample_id': sid, 'user_id': edge.get('user_id'), 'item_id': edge.get('item_id'), 'from_segment': fr, 'to_segment': to, 'edge_candidate_slot': slot, 'split_group_key': f'{sid}:{fr}:{to}:{ih}', 'edge_instance_key': f'{sid}:{fr}:{to}:{ih}:{oh}', 'input_hash': ih, 'output_hash': oh, 'pack_hash': pack.get('pack_hash'), 'pack_source_file': pack.get('_pack_source_file') or edge.get('complete_source_file') or edge.get('raw_output_path'), 'teacher_model': edge.get('model_name'), 'sample_weight': 1.0, 'input': inp, 'output': out, 'quality': {'final_source_status': fs['status'], 'edge_candidate_slot': slot}})
    edge_splits = split_records(edge_records)
    EDGE_OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(EDGE_OUT / 'edge_transition_sft_all.jsonl', edge_records)
    for sp, rows in edge_splits.items():
        write_jsonl(EDGE_OUT / f'edge_transition_sft_{sp}.jsonl', rows)
    write_jsonl(EDGE_OUT / 'edge_transition_sft_valid_dedup.jsonl', edge_splits['valid'])
    write_jsonl(EDGE_OUT / 'edge_transition_sft_test_dedup.jsonl', edge_splits['test'])
    write_jsonl(EDGE_OUT / 'edge_transition_sft_failures.jsonl', edge_fail)
    write_json(EDGE_OUT / 'edge_transition_sft_report.json', {'records': len(edge_records), 'failures': len(edge_fail), 'sample_count': len({r['sample_id'] for r in edge_records}), 'split_counts': {k: len(v) for k, v in edge_splits.items()}, 'edge_slot_counts': dict(Counter((r['edge_candidate_slot'] for r in edge_records))), 'final_sample_status_counts': dict(Counter((x['status'] for x in final_samples.values()))), 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S')})
    write_json(EDGE_OUT / 'edge_transition_sft_config.json', {'source': 'selected_teacher_samples', 'user_instruction': EDGE_PROMPT_INSTRUCTION, 'teacher_model': 'Qwen2.5-72B-Instruct-AWQ'})
    sums(EDGE_OUT)
    print(json.dumps({'final_samples': len(final_samples), 'state_records': len(state_records), 'state_failures': len(state_fail), 'edge_records': len(edge_records), 'edge_failures': len(edge_fail), 'state_dir': str(STATE_OUT), 'edge_dir': str(EDGE_OUT)}, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
