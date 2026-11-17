from __future__ import annotations
import argparse, concurrent.futures, json, math, re, time, urllib.request
from pathlib import Path
PROMPT_VERSION = 'lifecycle'
PACK_VERSION = 'lifecycle'
MODEL = 'Qwen2.5-72B-Instruct-AWQ'
LENSES = {'C1': ('dominant_feedback_pattern', 'Focus on the strongest overall feedback pattern in this segment. If the segment is mixed, identify the main positive pattern and the main negative contrast. If the segment is mostly positive, emphasize positive preference. If mostly negative, emphasize rejection pattern.'), 'C2': ('positive_negative_contrast', 'Focus on what separates positive interactions from negative interactions. Identify semantic patterns that are liked versus patterns that are rejected or risky. Use this lens to make the state useful for binary Yes/No recommendation.'), 'C3': ('fine_grained_decision_signal', 'Focus on finer-grained signals not captured by the dominant pattern. Look for specific themes, tones, story styles, director/actor signals, or semantic terms that distinguish positive from negative interactions. If no distinct signal exists, return insufficient_distinct_evidence.')}

def jdump(x):
    return json.dumps(x, ensure_ascii=False, separators=(',', ':'))

def read_jsonl(p: Path):
    with p.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def parse_output(raw: str):
    m = re.search('\\{.*\\}', raw, flags=re.S)
    if not m:
        raise ValueError('no_json_object')
    return json.loads(m.group(0))

def validate(obj, cid, lens):
    errs = []
    cand = (obj or {}).get('candidate') if isinstance(obj, dict) else None
    if not isinstance(cand, dict):
        return ['missing_candidate']
    if cand.get('candidate_id') != cid:
        errs.append('candidate_id_mismatch')
    if cand.get('analysis_lens') != lens:
        errs.append('analysis_lens_mismatch')
    status = cand.get('candidate_status')
    if status not in ['proposed', 'insufficient_distinct_evidence']:
        errs.append('bad_candidate_status')
    if cid != 'C3' and status != 'proposed':
        errs.append('only_c3_can_be_insufficient')
    if status == 'proposed':
        for k in ['state_type', 'state_label', 'state_summary', 'positive_analysis', 'negative_analysis', 'decision_cues']:
            if k not in cand:
                errs.append('missing_' + k)
        cues = cand.get('decision_cues', {}) if isinstance(cand.get('decision_cues'), dict) else {}
        if len(cues.get('positive_match_cues', []) or []) > 5:
            errs.append('too_many_positive_match_cues')
        if len(cues.get('negative_match_cues', []) or []) > 5:
            errs.append('too_many_negative_match_cues')
        da = cues.get('director_actor_cues', {}) if isinstance(cues.get('director_actor_cues'), dict) else {}
        for key in ['liked_directors', 'liked_actors', 'rejected_directors', 'rejected_actors']:
            if len(da.get(key, []) or []) > 2:
                errs.append('too_many_' + key)
        for section, arrkey in [('positive_analysis', 'liked_patterns'), ('negative_analysis', 'rejected_or_risky_patterns')]:
            sec = cand.get(section, {}) if isinstance(cand.get(section), dict) else {}
            for pat in sec.get(arrkey, []) or []:
                if len(pat.get('evidence_titles', []) or []) > 3:
                    errs.append('too_many_evidence_titles')
                if len(pat.get('support_terms', []) or []) > 6:
                    errs.append('too_many_support_terms')
    return errs

def retry_instruction(errs):
    return 'Regenerate the full JSON only. Fix these validation errors: ' + ', '.join(errs) + '. Keep arrays within limits: evidence_titles <=3 per pattern; support_terms <=6 per pattern; positive_match_cues and negative_match_cues <=5; each director/actor cue list <=2.'

def logprob_summary(choice):
    lp = choice.get('logprobs') or {}
    toks = lp.get('content') or []
    vals = []
    ents = []
    for tok in toks:
        if not isinstance(tok, dict):
            continue
        if tok.get('logprob') is not None:
            vals.append(float(tok['logprob']))
        top = tok.get('top_logprobs') or []
        probs = []
        for x in top:
            if isinstance(x, dict) and x.get('logprob') is not None:
                probs.append(math.exp(float(x['logprob'])))
        z = sum(probs)
        if z > 0:
            ps = [p / z for p in probs]
            ents.append(-sum((p * math.log(max(p, 1e-30)) for p in ps)))
    return {'token_count_for_scoring': len(toks), 'sampled_logprob_count': len(vals), 'mean_sampled_token_logprob': sum(vals) / len(vals) if vals else None, 'avg_token_nll': -sum(vals) / len(vals) if vals else None, 'approx_topk_entropy_mean': sum(ents) / len(ents) if ents else None, 'topk_entropy_count': len(ents), 'low_conf_token_rate': sum((1 for v in vals if v < -2.0)) / len(vals) if vals else None}

def call_chat(endpoint, model, messages, max_tokens, timeout, logprobs, top_logprobs):
    payload = {'model': model, 'messages': messages, 'temperature': 0.0, 'max_tokens': max_tokens}
    if logprobs:
        payload['logprobs'] = True
        payload['top_logprobs'] = int(top_logprobs)
    req = urllib.request.Request(endpoint.rstrip() + '/chat/completions', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        res = json.loads(resp.read().decode('utf-8'))
    choice = res['choices'][0]
    return (choice['message']['content'], logprob_summary(choice) if logprobs else None, res.get('usage'))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompt', required=True)
    ap.add_argument('--pack-tasks', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--endpoint', default='http://127.0.0.1:8010/v1')
    ap.add_argument('--model', default=MODEL)
    ap.add_argument('--max-output-tokens', type=int, default=768)
    ap.add_argument('--timeout', type=int, default=300)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--max-attempts', type=int, default=2)
    ap.add_argument('--logprobs', action='store_true')
    ap.add_argument('--top-logprobs', type=int, default=5)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'raw_outputs').mkdir(exist_ok=True)
    prompt = Path(args.prompt).read_text(encoding='utf-8')
    tasks = list(read_jsonl(Path(args.pack_tasks)))
    with (out / 'state_packs.jsonl').open('w', encoding='utf-8') as f:
        for r in tasks:
            f.write(jdump({k: r.get(k) for k in ['user_id', 'item_id', 'segment_id', 'candidate_id', 'analysis_lens', 'pack_hash', 'source_pack_file', 'pack_text']}) + '\n')

    def run_one(r):
        cid = str(r['candidate_id'])
        lens, instr = LENSES[cid]
        pp = prompt.replace('{{candidate_id}}', cid).replace('{{analysis_lens}}', lens).replace('{{lens_instruction}}', instr)
        base_messages = [{'role': 'system', 'content': pp}, {'role': 'user', 'content': r['pack_text']}]
        messages = base_messages
        attempts = []
        final_raw = ''
        obj = None
        final_status = 'success'
        final_err = []
        final_lp = None
        final_usage = None
        total = 0
        for ai in range(1, max(1, args.max_attempts) + 1):
            st = time.time()
            raw = ''
            status = 'success'
            err = []
            parsed = None
            lp = None
            usage = None
            try:
                raw, lp, usage = call_chat(args.endpoint, args.model, messages, args.max_output_tokens, args.timeout, args.logprobs, args.top_logprobs)
                parsed = parse_output(raw)
                err = validate(parsed, cid, lens)
                if err:
                    status = 'validator_error'
            except Exception as e:
                status = 'request_or_parse_error'
                err = [repr(e)]
            dt = time.time() - st
            total += dt
            raw_path = out / 'raw_outputs' / f"u{r['user_id']}_i{r['item_id']}_{r['segment_id']}_{cid}_a{ai}.txt"
            raw_path.write_text(raw, encoding='utf-8')
            attempts.append({'user_id': r['user_id'], 'item_id': r['item_id'], 'segment_id': r['segment_id'], 'candidate_id': cid, 'analysis_lens': lens, 'attempt_index': ai, 'status': status, 'errors': err, 'latency_sec': round(dt, 3), 'pack_hash': r.get('pack_hash'), 'raw_output_path': str(raw_path)})
            final_raw = raw
            obj = parsed
            final_status = status
            final_err = err
            final_lp = lp
            final_usage = usage
            if status == 'success':
                break
            if ai < args.max_attempts:
                messages = base_messages + [{'role': 'user', 'content': retry_instruction(err)}]
        row = {'user_id': r['user_id'], 'item_id': r['item_id'], 'segment_id': r['segment_id'], 'candidate_id': cid, 'analysis_lens': lens, 'pack_hash': r.get('pack_hash'), 'source_pack_file': r.get('source_pack_file'), 'source_candidate_file': r.get('source_candidate_file'), 'model_output': obj, 'logprob_summary': final_lp, 'usage': final_usage}
        fail = {'user_id': r['user_id'], 'item_id': r['item_id'], 'segment_id': r['segment_id'], 'candidate_id': cid, 'status': final_status, 'errors': final_err, 'pack_hash': r.get('pack_hash')} if final_status != 'success' else None
        return (attempts, row if final_status == 'success' else None, fail, final_status, total, final_lp, final_usage)
    success = fail = done = 0
    with (out / 'state_generation_attempts.jsonl').open('w', encoding='utf-8', buffering=1) as fa, (out / 'state_candidates.jsonl').open('w', encoding='utf-8', buffering=1) as fc, (out / 'state_generation_failures.jsonl').open('w', encoding='utf-8', buffering=1) as ff, (out / 'state_logprobs.jsonl').open('w', encoding='utf-8', buffering=1) as fl:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_one, r) for r in tasks]
            for fut in concurrent.futures.as_completed(futs):
                attempts, row, fr, status, total, lp, usage = fut.result()
                done += 1
                for a in attempts:
                    fa.write(jdump(a) + '\n')
                if row:
                    success += 1
                    fc.write(jdump(row) + '\n')
                else:
                    fail += 1
                    ff.write(jdump(fr) + '\n')
                fl.write(jdump({'user_id': (row or fr)['user_id'], 'item_id': (row or fr)['item_id'], 'segment_id': (row or fr)['segment_id'], 'candidate_id': (row or fr)['candidate_id'], 'pack_hash': (row or fr).get('pack_hash'), 'status': status, 'logprob_summary': lp, 'usage': usage}) + '\n')
                print('done', done, '/', len(tasks), status, round(total, 3), 'success', success, 'fail', fail, flush=True)
    report = {'prompt_version': PROMPT_VERSION, 'pack_version': PACK_VERSION, 'tasks': len(tasks), 'success': success, 'failures': fail, 'workers': args.workers, 'max_attempts': args.max_attempts, 'logprobs': bool(args.logprobs), 'top_logprobs': int(args.top_logprobs), 'source': 'prepared_pack'}
    (out / 'state_generation_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
