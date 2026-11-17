from __future__ import annotations

def main():
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', required=True, type=str)
    parser.add_argument('--judge-dir', required=True, type=str)
    parser.add_argument('--output-dir', required=True, type=str)
    parser.add_argument('--endpoint', default='http://127.0.0.1:8010/v1')
    parser.add_argument('--model', default='Qwen2.5-14B-Instruct')
    args = parser.parse_args()
    import json
    import os
    import re
    import math
    import time
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor, as_completed
    SRC = args.source_dir
    JUDGE = args.judge_dir
    OUT = args.output_dir
    os.makedirs(OUT, exist_ok=True)
    ENDPOINT = args.endpoint.rstrip('/') + '/chat/completions'
    MODEL = args.model
    WORKERS = int(os.environ.get('WORKERS', '32'))
    MAX_ATTEMPTS = 2

    def read_jsonl(p):
        with open(p, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    def norm_value(vals, v):
        xs = [x for x in vals if isinstance(x, (int, float)) and math.isfinite(x)]
        if not xs or not isinstance(v, (int, float)) or (not math.isfinite(v)):
            return 0.0
        lo, hi = (min(xs), max(xs))
        return 0.0 if hi <= lo else max(0.0, min(1.0, (v - lo) / (hi - lo)))
    prompts = {}
    for d in read_jsonl(JUDGE + '/state_edge_prompts.jsonl'):
        u = d['messages'][1]['content']
        idx = u.find('Target movie:')
        prompts[d['sample_id']] = {'meta': {k: d.get(k) for k in ['sample_id', 'user_id', 'target_item_id', 'label', 'rating', 'num_segments', 'num_edges']}, 'system': d['messages'][0]['content'], 'tail': u[idx:] if idx >= 0 else u}
    judge = {}
    failed = []
    for d in read_jsonl(JUDGE + '/state_edge_results.jsonl'):
        judge[d['sample_id']] = d
        if d.get('prediction') != d.get('label'):
            failed.append(d['sample_id'])
    states = {}
    segs_by_sample = {}
    for d in read_jsonl(SRC + '/state_candidates.jsonl'):
        sid = 'u%s_target_%s' % (d['user_id'], d['item_id'])
        if d['candidate_id'] == 'C1':
            states[sid, d['segment_id']] = d
            segs_by_sample.setdefault(sid, set()).add(d['segment_id'])
    edges = {}
    for d in read_jsonl(SRC + '/transition_candidates.jsonl'):
        sid = d['sample_id']
        eid = d.get('edge_candidate_id') or d.get('generation_variant')
        edges[sid, d['from_segment'], d['to_segment'], eid] = d
    edge_lps = {}
    raw = []
    for d in read_jsonl(SRC + '/transition_logprobs.jsonl'):
        sid = d['sample_id']
        eid = d.get('edge_candidate_id') or d.get('generation_variant')
        k = (sid, d['from_segment'], d['to_segment'], eid)
        s = d.get('logprob_summary') or {}
        edge_lps[k] = s
        raw.append(s)
    fields = {f: [x.get(f) for x in raw] for f in ['avg_token_nll', 'approx_topk_entropy_mean', 'low_conf_token_rate']}

    def uncertainty(s):
        return 0.5 * norm_value(fields['avg_token_nll'], s.get('avg_token_nll')) + 0.3 * norm_value(fields['approx_topk_entropy_mean'], s.get('approx_topk_entropy_mean')) + 0.2 * norm_value(fields['low_conf_token_rate'], s.get('low_conf_token_rate'))
    edge_unc = {k: uncertainty(s) for k, s in edge_lps.items()}
    NEG = {'none', 'unknown', 'n/a', ''}

    def clean(x):
        return '' if x is None else str(x).strip()

    def items(arr, maxn=5):
        out = []
        if not isinstance(arr, list):
            return 'none'
        for x in arr:
            if isinstance(x, dict):
                sig = clean(x.get('signal') or x.get('term') or x.get('pattern'))
                why = clean(x.get('why') or x.get('reason'))
                txt = sig + (' (' + why + ')' if why else '')
            else:
                txt = clean(x)
            if txt and txt.lower() not in NEG:
                out.append(txt)
        return '; '.join(out[:maxn]) if out else 'none'

    def terms(arr, maxn=8):
        out = []
        if not isinstance(arr, list):
            return 'none'
        for x in arr:
            if isinstance(x, dict):
                x = x.get('signal') or x.get('term') or x.get('pattern') or x.get('name')
            x = clean(x)
            if x and x.lower() not in NEG and (x not in out):
                out.append(x)
        return ', '.join(out[:maxn]) if out else 'none'

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
        c = (row.get('model_output') or {}).get('candidate') or {}
        lines = ['[STATE %s]' % seg, 'State label: ' + clean(c.get('state_label')), 'State summary: ' + clean(c.get('state_summary'))]
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
        fs, ts = (row.get('from_segment'), row.get('to_segment'))
        pos, neg = (e.get('positive_transition') or {}, e.get('negative_caution_transition') or {})
        return '\n'.join(['[EDGE %s->%s]' % (fs, ts), 'Transition label: ' + clean(e.get('transition_label')), 'Transition type: ' + clean(e.get('transition_type')), 'Positive transition:', '- analysis: ' + clean(pos.get('analysis')), '- retained: ' + items(pos.get('retained')), '- emerging: ' + items(pos.get('emerging')), '- fading context: ' + items(pos.get('fading')), 'Negative caution transition:', '- analysis: ' + clean(neg.get('analysis')), '- retained: ' + items(neg.get('retained')), '- emerging: ' + items(neg.get('emerging')), '- fading: ' + items(neg.get('fading')), 'Polarity conflicts: ' + items(e.get('polarity_conflicts')), 'Transition summary: ' + clean(e.get('transition_summary'))])

    def seg_sort(x):
        return int(re.sub('\\D', '', x) or 0)

    def make_messages(sid, edge_sel):
        segs = sorted(segs_by_sample.get(sid, []), key=seg_sort)
        parts = ['Lifecycle interest chain before the target:\n']
        for i, seg in enumerate(segs):
            parts.append(render_state(seg, states[sid, seg]))
            if i < len(segs) - 1:
                fs, ts = (seg, segs[i + 1])
                eid = edge_sel.get((fs, ts), 'E1')
                row = edges.get((sid, fs, ts, eid)) or edges.get((sid, fs, ts, 'E1'))
                if row:
                    parts.append(render_edge(row))
        user = '\n\n'.join(parts) + '\n\n' + prompts[sid]['tail']
        return [{'role': 'system', 'content': prompts[sid]['system']}, {'role': 'user', 'content': user}]

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

    def build_q(sid):
        q = []
        segs = sorted(segs_by_sample.get(sid, []), key=seg_sort)
        for i in range(len(segs) - 1):
            fs, ts = (segs[i], segs[i + 1])
            cur = (sid, fs, ts, 'E1')
            curu = edge_unc.get(cur, 0.0)
            repl = []
            for eid in ['E2', 'E3']:
                k = (sid, fs, ts, eid)
                if k in edges:
                    repl.append({'candidate_id': eid, 'uncertainty': edge_unc.get(k, 1.0)})
            repl = sorted(repl, key=lambda x: x['uncertainty'])
            if repl:
                q.append({'sample_id': sid, 'from_segment': fs, 'to_segment': ts, 'old_candidate_id': 'E1', 'old_uncertainty': curu, 'replacement_candidates': repl, 'priority': curu * (1.0 + 0.2 * i)})
        return sorted(q, key=lambda x: -x['priority'])
    queues = {sid: build_q(sid) for sid in failed}
    with open(OUT + '/repair_action_queue.jsonl', 'w', encoding='utf-8') as f:
        for sid, q in queues.items():
            for r, a in enumerate(q, 1):
                b = dict(a)
                b['queue_rank'] = r
                f.write(json.dumps(b, ensure_ascii=False) + '\n')

    def run_sid(sid):
        label = judge[sid]['label']
        attempts = []
        aid = 0
        for action in queues.get(sid, []):
            for repl in action['replacement_candidates']:
                if aid >= MAX_ATTEMPTS:
                    break
                aid += 1
                edge_sel = {(action['from_segment'], action['to_segment']): repl['candidate_id']}
                res = call(make_messages(sid, edge_sel))
                pred = res.get('prediction')
                success = pred == label
                row = {'sample_id': sid, 'attempt_id': aid, 'label': label, 'repair_action': 'edge_replacement', 'target_node': action['from_segment'] + '->' + action['to_segment'], 'old_candidate_id': 'E1', 'new_candidate_id': repl['candidate_id'], 'old_uncertainty': action['old_uncertainty'], 'new_uncertainty': repl['uncertainty'], 'judge_before_prediction': judge[sid].get('prediction'), 'judge_after_prediction': pred, 'success': success, 'valid_prediction': pred is not None, 'response': res.get('response'), 'error': res.get('error'), 'latency_sec': res.get('latency_sec'), 'usage': res.get('usage')}
                attempts.append(row)
                if success:
                    return {'sample_id': sid, 'label': label, 'original_prediction': judge[sid].get('prediction'), 'final_prediction': pred, 'final_correct': True, 'final_status': 'repaired_success_edge_replacement', 'num_attempts': aid, 'successful_attempt_id': aid, 'repair_action': 'edge_replacement', 'target_node': row['target_node'], 'new_candidate_id': repl['candidate_id'], 'attempts': attempts}
            if aid >= MAX_ATTEMPTS:
                break
        return {'sample_id': sid, 'label': label, 'original_prediction': judge[sid].get('prediction'), 'final_prediction': judge[sid].get('prediction'), 'final_correct': False, 'final_status': 'edge_repair_failed_or_no_eligible_action', 'num_attempts': aid, 'successful_attempt_id': None, 'attempts': attempts}
    final_path = OUT + '/repair_final_results.jsonl'
    attempts_path = OUT + '/repair_attempts.jsonl'
    done = set()
    if os.path.exists(final_path):
        for d in read_jsonl(final_path):
            done.add(d['sample_id'])
    todo = [sid for sid in failed if sid not in done]
    print('edge_only failed_total', len(failed), 'todo', len(todo), 'workers', WORKERS)
    fa = open(attempts_path, 'a', encoding='utf-8')
    ff = open(final_path, 'a', encoding='utf-8')
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(run_sid, sid): sid for sid in todo}
        n = 0
        for fut in as_completed(futs):
            res = fut.result()
            n += 1
            attempts = res.pop('attempts')
            for a in attempts:
                fa.write(json.dumps(a, ensure_ascii=False) + '\n')
            ff.write(json.dumps(res, ensure_ascii=False) + '\n')
            if n % 25 == 0:
                fa.flush()
                ff.flush()
                print('done', n, '/', len(todo), flush=True)
    fa.close()
    ff.close()
    finals = {d['sample_id']: d for d in read_jsonl(final_path)}
    y = []
    s = []
    correct = 0
    repaired = 0
    for sid, d in judge.items():
        pred = d.get('prediction')
        if sid in finals:
            pred = finals[sid]['final_prediction']
            if finals[sid]['final_correct']:
                repaired += 1
        lab = d['label']
        y.append(lab)
        s.append(float(pred))
        correct += int(pred == lab)

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
    report = {'base_correct': sum((1 for d in judge.values() if d.get('correct'))), 'base_accuracy': sum((1 for d in judge.values() if d.get('correct'))) / len(judge), 'failed_total': len(failed), 'repair_final_rows': len(finals), 'repaired_success': repaired, 'correct_after_repair': correct, 'accuracy_after_repair': correct / len(judge), 'auc_after_repair': auc(y, s), 'workers': WORKERS, 'max_attempts_per_failed_sample': MAX_ATTEMPTS, 'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S')}
    json.dump(report, open(OUT + '/repair_report.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
