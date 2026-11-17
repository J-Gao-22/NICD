from __future__ import annotations

def main():
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--endpoint', default='http://127.0.0.1:8010/v1')
    parser.add_argument('--model', default='Qwen2.5-14B-Instruct')
    args = parser.parse_args()
    import json
    import pathlib
    import time
    import urllib.request
    import re
    import collections
    import datetime
    from concurrent.futures import ThreadPoolExecutor, as_completed
    OUT = args.output_dir
    PROMPTS = OUT / 'state_edge_prompts.jsonl'
    RESULTS = OUT / 'state_edge_results.jsonl'
    REPORT = OUT / 'state_edge_report.json'
    ENDPOINT = args.endpoint
    MODEL = args.model

    def read_jsonl(p):
        return [json.loads(x) for x in p.open(encoding='utf-8') if x.strip()]

    def append(p, r):
        with p.open('a', encoding='utf-8') as f:
            f.write(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n')

    def call(messages, timeout=120):
        payload = {'model': MODEL, 'messages': messages, 'temperature': 0.0, 'top_p': 1.0, 'max_tokens': 128}
        req = urllib.request.Request(ENDPOINT.rstrip('/') + '/chat/completions', data=json.dumps(payload, ensure_ascii=False).encode(), headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
        choice = body['choices'][0]
        return (choice['message']['content'], choice.get('finish_reason'), body.get('usage') or {})

    def parse_pred(text):
        m = re.match('^\\s*(Yes|No)\\b', str(text or ''), re.I)
        if not m:
            return None
        return 1 if m.group(1).lower() == 'yes' else 0

    def auc(labels, scores):
        pairs = sorted(zip(scores, labels), key=lambda x: x[0])
        npos = sum(labels)
        nneg = len(labels) - npos
        if npos == 0 or nneg == 0:
            return None
        rank_sum = 0.0
        i = 0
        rank = 1
        while i < len(pairs):
            j = i
            while j < len(pairs) and pairs[j][0] == pairs[i][0]:
                j += 1
            avg = (rank + rank + (j - i) - 1) / 2
            for k in range(i, j):
                if pairs[k][1] == 1:
                    rank_sum += avg
            rank += j - i
            i = j
        return (rank_sum - npos * (npos + 1) / 2) / (npos * nneg)

    def summarize(rows, start, final=False):
        valid = [x for x in rows if x.get('valid_prediction')]
        labels = [int(x['label']) for x in valid]
        scores = [float(x['score']) for x in valid]
        rep = {'requested': len(prompt_rows), 'completed': len(rows), 'valid_predictions': len(valid), 'errors': sum((1 for x in rows if x.get('error'))), 'auc': auc(labels, scores) if valid else None, 'accuracy': sum((1 for x in valid if x.get('correct'))) / len(valid) if valid else None, 'correct': sum((1 for x in valid if x.get('correct'))), 'wrong': sum((1 for x in valid if not x.get('correct'))), 'pred_counts': dict(collections.Counter((str(x.get('prediction')) for x in valid))), 'label_counts': dict(collections.Counter((str(x.get('label')) for x in rows))), 'elapsed_sec': round(time.time() - start, 3), 'updated_at' if not final else 'finished_at': datetime.datetime.now().isoformat()}
        REPORT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
        return rep

    def run_one(r):
        st = time.time()
        meta = {k: r.get(k) for k in ['sample_id', 'user_id', 'target_item_id', 'label', 'rating', 'num_segments', 'num_edges', 'prompt_variant', 'edge_variant']}
        try:
            text, finish, usage = call(r['messages'])
            pred = parse_pred(text)
            return {**meta, 'response': text, 'prediction': pred, 'score': 1.0 if pred == 1 else 0.0 if pred == 0 else None, 'valid_prediction': pred is not None, 'correct': pred == r.get('label') if pred is not None else False, 'finish_reason': finish, 'usage': usage, 'latency_sec': round(time.time() - st, 4)}
        except Exception as e:
            return {**meta, 'error': repr(e), 'valid_prediction': False, 'prediction': None, 'score': None, 'latency_sec': round(time.time() - st, 4)}
    prompt_rows = read_jsonl(PROMPTS)
    done = set()
    if RESULTS.exists():
        for r in read_jsonl(RESULTS):
            done.add(r.get('sample_id'))
    todo = [r for r in prompt_rows if r['sample_id'] not in done]
    RESULTS.touch(exist_ok=True)
    start = time.time()
    with ThreadPoolExecutor(max_workers=32) as ex:
        futs = [ex.submit(run_one, r) for r in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            append(RESULTS, res)
            if i == 1 or i % 100 == 0:
                all_rows = read_jsonl(RESULTS)
                rep = summarize(all_rows, start)
                print(json.dumps(rep, ensure_ascii=False), flush=True)
    all_rows = read_jsonl(RESULTS)
    rep = summarize(all_rows, start, final=True)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
