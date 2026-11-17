from __future__ import annotations

def main():
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', required=True, type=Path)
    parser.add_argument('--judge-dir', required=True, type=Path)
    parser.add_argument('--transition-repair-dir', required=True, type=Path)
    parser.add_argument('--state-repair-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--endpoint', default='http://127.0.0.1:8010/v1')
    parser.add_argument('--model', default='Qwen2.5-14B-Instruct')
    args = parser.parse_args()
    import json
    import os
    import re
    import time
    import urllib.request
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor, as_completed
    SRC = args.source_dir
    JUDGE = args.judge_dir
    EDGE_ONLY = args.transition_repair_dir
    STATE = args.state_repair_dir
    MAIN = STATE / 'transitions'
    MISS = STATE / 'transitions'
    OUT = args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    ENDPOINT = args.endpoint.rstrip('/') + '/chat/completions'
    MODEL = args.model
    WORKERS = int(os.environ.get('WORKERS', '32'))
    MAX_ATTEMPTS_PER_SAMPLE = 2

    def read_jsonl(p):
        if not Path(p).exists():
            return []
        return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]

    def clean(x):
        return '' if x is None else str(x).strip()

    def segnum(x):
        return int(re.sub('\\D', '', str(x)) or 0)
    NEG = {'none', 'unknown', 'n/a', ''}

    def terms(arr, maxn=8):
        out = []
        if not isinstance(arr, list):
            return 'none'
        for x in arr:
            if isinstance(x, dict):
                x = x.get('signal') or x.get('term') or x.get('pattern') or x.get('name')
            s = clean(x)
            if s and s.lower() not in NEG and (s not in out):
                out.append(s)
        return ', '.join(out[:maxn]) if out else 'none'

    def items(arr, maxn=5):
        out = []
        if not isinstance(arr, list):
            return 'none'
        for x in arr:
            if isinstance(x, dict):
                sig = clean(x.get('signal') or x.get('term') or x.get('pattern'))
                why = clean(x.get('why') or x.get('reason'))
                s = sig + (' (' + why + ')' if why else '')
            else:
                s = clean(x)
            if s and s.lower() not in NEG:
                out.append(s)
        return '; '.join(out[:maxn]) if out else 'none'

    def pattern(p):
        if isinstance(p, dict):
            pat = clean(p.get('pattern') or p.get('factor') or p.get('signal'))
            ts = terms(p.get('support_terms') or [], 8)
            ev = terms(p.get('evidence_titles') or [], 5)
            extra = []
            if ts != 'none':
                extra.append('terms: ' + ts)
            if ev != 'none':
                extra.append('evidence: ' + ev)
            return pat + (' (' + '; '.join(extra) + ')' if extra else '')
        return clean(p)

    def render_state(seg, row):
        c = (row.get('model_output') or {}).get('candidate') or row.get('selected_candidate') or {}
        lines = ['[STATE {}]'.format(seg), 'State label: ' + clean(c.get('state_label')), 'State summary: ' + clean(c.get('state_summary'))]
        pa = c.get('positive_analysis') or {}
        lines.append('Positive analysis:')
        if pa.get('reasoning'):
            lines.append('- reasoning: ' + clean(pa.get('reasoning')))
        pats = pa.get('liked_patterns') or []
        if pats:
            for p in pats[:4]:
                lines.append('- ' + pattern(p))
        else:
            lines.append('- none')
        na = c.get('negative_analysis') or {}
        lines.append('Negative / caution analysis:')
        if na.get('reasoning'):
            lines.append('- reasoning: ' + clean(na.get('reasoning')))
        rps = na.get('rejected_or_risky_patterns') or []
        if rps:
            for p in rps[:3]:
                lines.append('- ' + pattern(p))
        else:
            lines.append('- none')
        dc = c.get('decision_cues') or {}
        lines.append('Decision cues:')
        lines.append('Positive cues: ' + terms(dc.get('positive_match_cues') or [], 10))
        lines.append('Negative cues: ' + terms(dc.get('negative_match_cues') or [], 8))
        dac = dc.get('director_actor_cues') or {}
        pp = []
        for k in ['liked_directors', 'liked_actors', 'rejected_directors', 'rejected_actors']:
            v = terms(dac.get(k) or [], 4)
            if v != 'none':
                pp.append(k + ': ' + v)
        lines.append('Director / actor cues: ' + ('; '.join(pp) if pp else 'none'))
        return '\n'.join(lines)

    def render_edge(row):
        e = (row.get('model_output') or {}).get('edge_node') or {}
        fs = row.get('from_segment')
        ts = row.get('to_segment')
        pos = e.get('positive_transition') or {}
        neg = e.get('negative_caution_transition') or {}
        return '\n'.join(['[EDGE {}->{}]'.format(fs, ts), 'Transition label: ' + clean(e.get('transition_label')), 'Transition type: ' + clean(e.get('transition_type')), 'Positive transition:', '- analysis: ' + clean(pos.get('analysis')), '- retained: ' + items(pos.get('retained')), '- emerging: ' + items(pos.get('emerging')), '- fading context: ' + items(pos.get('fading')), 'Negative caution transition:', '- analysis: ' + clean(neg.get('analysis')), '- retained: ' + items(neg.get('retained')), '- emerging: ' + items(neg.get('emerging')), '- fading: ' + items(neg.get('fading')), 'Polarity conflicts: ' + items(e.get('polarity_conflicts')), 'Transition summary: ' + clean(e.get('transition_summary'))])

    def parse_pred(txt):
        t = (txt or '').strip().lower()
        if t.startswith('yes'):
            return 1
        if t.startswith('no'):
            return 0
        m = re.search('\\b(yes|no)\\b', t[:100])
        return 1 if m and m.group(1) == 'yes' else 0 if m else None

    def call(messages):
        body = {'model': MODEL, 'messages': messages, 'temperature': 0.0, 'top_p': 1.0, 'max_tokens': 128}
        req = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=240) as r:
                d = json.loads(r.read().decode())
            txt = d['choices'][0]['message']['content']
            return {'ok': True, 'response': txt, 'prediction': parse_pred(txt), 'usage': d.get('usage'), 'finish_reason': d['choices'][0].get('finish_reason'), 'latency_sec': round(time.time() - t0, 3)}
        except Exception as e:
            return {'ok': False, 'error': repr(e), 'latency_sec': round(time.time() - t0, 3)}
    prompt_meta = {}
    for d in read_jsonl(JUDGE / 'state_edge_prompts.jsonl'):
        u = d['messages'][1]['content']
        idx = u.find('Target movie:')
        prompt_meta[d['sample_id']] = {'system': d['messages'][0]['content'], 'tail': u[idx:] if idx >= 0 else u, 'label': d['label']}
    base_judge = {d['sample_id']: d for d in read_jsonl(JUDGE / 'state_edge_results.jsonl')}
    edge_final = {d['sample_id']: d for d in read_jsonl(EDGE_ONLY / 'repair_final_results.jsonl')}
    state_queue = read_jsonl(STATE / 'state_repair_action_queue.jsonl')
    q_by_sample = {}
    for a in state_queue:
        q_by_sample.setdefault(a['sample_id'], []).append(a)
    for sid in q_by_sample:
        q_by_sample[sid] = sorted(q_by_sample[sid], key=lambda x: x['attempt_id'])[:MAX_ATTEMPTS_PER_SAMPLE]
    regen_pack_hash = {}
    for p in read_jsonl(STATE / 'repair_transition_packs.jsonl'):
        regen_pack_hash[p.get('repair_attempt_key'), p.get('from_segment'), p.get('to_segment')] = p.get('pack_hash')
    states = {}
    segs = {}
    for d in read_jsonl(SRC / 'state_candidates.jsonl'):
        sid = 'u{}_target_{}'.format(d['user_id'], d['item_id'])
        states[sid, d['segment_id'], d['candidate_id']] = d
        segs.setdefault(sid, set()).add(d['segment_id'])
    old_edges = {}
    for d in read_jsonl(SRC / 'transition_candidates.jsonl'):
        eid = d.get('edge_candidate_id') or d.get('generation_variant')
        if eid == 'E1':
            old_edges[d['sample_id'], d['from_segment'], d['to_segment']] = d
    regen_edges_by_hash = {}
    regen_fail_hashes = set()
    for root in [MAIN]:
        for d in read_jsonl(root / 'transition_candidates.jsonl'):
            regen_edges_by_hash[d.get('pack_hash')] = d
        for d in read_jsonl(root / 'transition_failures.jsonl'):
            regen_fail_hashes.add(d.get('pack_hash'))

    def build_messages(sid, action):
        replace_seg = action['target_node']
        newcid = action['new_candidate_id']
        attempt_key = action['repair_attempt_key']
        sample_segs = sorted(segs.get(sid, []), key=segnum)
        parts = ['Lifecycle interest chain before the target:\n']
        for i, seg in enumerate(sample_segs):
            cid = newcid if seg == replace_seg else 'C1'
            parts.append(render_state(seg, states[sid, seg, cid]))
            if i < len(sample_segs) - 1:
                fs, ts = (seg, sample_segs[i + 1])
                if fs == replace_seg or ts == replace_seg:
                    ph = regen_pack_hash.get((attempt_key, fs, ts))
                    ed = regen_edges_by_hash.get(ph)
                    if not ed:
                        return (None, 'missing_regenerated_edge_{}->{} pack_hash={}'.format(fs, ts, ph))
                else:
                    ed = old_edges.get((sid, fs, ts))
                    if not ed:
                        return (None, 'missing_original_edge_{}->{}'.format(fs, ts))
                parts.append(render_edge(ed))
        user = '\n\n'.join(parts) + '\n\n' + prompt_meta[sid]['tail']
        return ([{'role': 'system', 'content': prompt_meta[sid]['system']}, {'role': 'user', 'content': user}], None)

    def run_sample(sid):
        label = prompt_meta[sid]['label']
        rows = []
        for action in q_by_sample.get(sid, [])[:MAX_ATTEMPTS_PER_SAMPLE]:
            messages, err = build_messages(sid, action)
            if err:
                row = {**action, 'label': label, 'repair_action': 'state_replacement_with_edge_regeneration', 'skipped': True, 'skip_reason': err, 'success': False, 'judge_after_prediction': None}
            else:
                res = call(messages)
                pred = res.get('prediction')
                row = {**action, 'label': label, 'repair_action': 'state_replacement_with_edge_regeneration', 'skipped': False, 'judge_before_prediction': base_judge[sid].get('prediction'), 'judge_after_prediction': pred, 'success': pred == label, 'valid_prediction': pred is not None, 'response': res.get('response'), 'error': res.get('error'), 'latency_sec': res.get('latency_sec'), 'usage': res.get('usage')}
            rows.append(row)
            if row.get('success'):
                break
        final = rows[-1] if rows else None
        return (sid, rows, {'sample_id': sid, 'label': label, 'original_prediction': base_judge[sid].get('prediction'), 'final_prediction': final.get('judge_after_prediction') if final and final.get('success') else base_judge[sid].get('prediction'), 'final_correct': bool(final and final.get('success')), 'final_status': 'repaired_success_state_edge_regen' if final and final.get('success') else 'state_repair_failed_or_no_eligible_action', 'num_attempts': len(rows), 'successful_attempt_id': final.get('attempt_id') if final and final.get('success') else None, 'repair_action': 'state_replacement_with_edge_regeneration' if final and final.get('success') else None, 'target_node': final.get('target_node') if final and final.get('success') else None, 'new_candidate_id': final.get('new_candidate_id') if final and final.get('success') else None})
    remaining = [sid for sid, d in base_judge.items() if d.get('prediction') != d.get('label') and (not edge_final.get(sid, {}).get('final_correct'))]
    final_path = OUT / 'state_repair_final_results.jsonl'
    att_path = OUT / 'state_repair_attempts_14b.jsonl'
    done = set((x['sample_id'] for x in read_jsonl(final_path)))
    todo = [sid for sid in remaining if sid not in done]
    print(json.dumps({'remaining_after_edge_only': len(remaining), 'todo': len(todo), 'workers': WORKERS, 'regen_edges': len(regen_edges_by_hash), 'regen_failures': len(regen_fail_hashes)}, ensure_ascii=False))
    fa = open(att_path, 'a', encoding='utf-8')
    ff = open(final_path, 'a', encoding='utf-8')
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(run_sample, sid): sid for sid in todo}
        n = 0
        for fut in as_completed(futs):
            sid, rows, final = fut.result()
            n += 1
            for r in rows:
                fa.write(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n')
            ff.write(json.dumps(final, ensure_ascii=False, separators=(',', ':')) + '\n')
            if n % 25 == 0:
                fa.flush()
                ff.flush()
                print('done', n, '/', len(todo), flush=True)
    fa.close()
    ff.close()
    edge_finals = {d['sample_id']: d for d in read_jsonl(EDGE_ONLY / 'repair_final_results.jsonl')}
    state_finals = {d['sample_id']: d for d in read_jsonl(final_path)}
    y = []
    s = []
    correct = 0
    edge_succ = 0
    state_succ = 0
    for sid, d in base_judge.items():
        label = d['label']
        pred = d.get('prediction')
        if edge_finals.get(sid, {}).get('final_correct'):
            pred = edge_finals[sid]['final_prediction']
            edge_succ += 1
        elif state_finals.get(sid, {}).get('final_correct'):
            pred = state_finals[sid]['final_prediction']
            state_succ += 1
        y.append(label)
        s.append(float(pred))
        correct += int(pred == label)

    def auc(y, s):
        pos = [b for a, b in zip(y, s) if a == 1]
        neg = [b for a, b in zip(y, s) if a == 0]
        wins = ties = 0
        for p in pos:
            for n in neg:
                if p > n:
                    wins += 1
                elif p == n:
                    ties += 1
        return (wins + 0.5 * ties) / (len(pos) * len(neg))
    report = {'base_correct': sum((1 for d in base_judge.values() if d.get('correct'))), 'base_accuracy': sum((1 for d in base_judge.values() if d.get('correct'))) / len(base_judge), 'edge_repaired_success': edge_succ, 'state_repaired_success': state_succ, 'total_repaired_success': edge_succ + state_succ, 'correct_after_repair': correct, 'accuracy_after_repair': correct / len(base_judge), 'auc_after_repair': auc(y, s), 'total_samples': len(base_judge), 'edge_regen_success_edges': len(regen_edges_by_hash), 'edge_regen_failed_edges': len(regen_fail_hashes), 'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S')}
    json.dump(report, open(OUT / 'state_repair_eval_report.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
