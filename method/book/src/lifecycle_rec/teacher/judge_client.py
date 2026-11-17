import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib import request, error
ENDPOINT = 'http://127.0.0.1:8010/v1/chat/completions'
MODEL = 'Qwen2.5-14B-Instruct'

def get_messages(row):
    if 'messages' in row and row['messages']:
        return row['messages']
    if 'system_prompt' in row and 'user_prompt' in row:
        return [{'role': 'system', 'content': row['system_prompt']}, {'role': 'user', 'content': row['user_prompt']}]
    if 'prompt' in row:
        return [{'role': 'user', 'content': row['prompt']}]
    raise KeyError('cannot find prompt/messages fields')

def parse_pred(text):
    s = (text or '').strip()
    if re.match('^yes\\b', s, flags=re.I):
        return (1, 'yes')
    if re.match('^no\\b', s, flags=re.I):
        return (0, 'no')
    return (None, 'invalid')

def extract_score(resp, pred):
    try:
        lp = resp['choices'][0].get('logprobs') or {}
        content = lp.get('content') or []
        if not content:
            return (float(pred) if pred is not None else 0.5, 'pred_fallback')
        first = content[0]
        tops = first.get('top_logprobs') or []
        yes_lp = None
        no_lp = None
        for t in tops:
            tok = (t.get('token') or '').strip().lower()
            val = t.get('logprob')
            if tok == 'yes':
                yes_lp = val if yes_lp is None else max(yes_lp, val)
            if tok == 'no':
                no_lp = val if no_lp is None else max(no_lp, val)
        first_tok = (first.get('token') or '').strip().lower()
        first_lp = first.get('logprob')
        if first_tok == 'yes':
            yes_lp = first_lp if yes_lp is None else max(yes_lp, first_lp)
        if first_tok == 'no':
            no_lp = first_lp if no_lp is None else max(no_lp, first_lp)
        floor = -30.0
        if yes_lp is not None or no_lp is not None:
            return ((yes_lp if yes_lp is not None else floor) - (no_lp if no_lp is not None else floor), 'logprob_diff')
    except Exception:
        pass
    return (float(pred) if pred is not None else 0.5, 'pred_fallback')

def call_model(row, idx, max_retries=3):
    payload = {'model': MODEL, 'messages': get_messages(row), 'temperature': 0.0, 'max_tokens': 192, 'logprobs': True, 'top_logprobs': 20}
    data = json.dumps(payload).encode('utf-8')
    headers = {'Content-Type': 'application/json'}
    last_err = None
    for attempt in range(1, max_retries + 1):
        t0 = time.time()
        try:
            req = request.Request(ENDPOINT, data=data, headers=headers, method='POST')
            with request.urlopen(req, timeout=180) as r:
                resp = json.loads(r.read().decode('utf-8'))
            text = resp['choices'][0]['message']['content']
            pred, pred_text = parse_pred(text)
            score, score_source = extract_score(resp, pred)
            return {'idx': idx, 'sample_id': row.get('sample_id'), 'user_id': row.get('user_id') or row.get('uid'), 'item_id': row.get('item_id') or row.get('iid') or row.get('target_item_id'), 'label': int(row.get('label')), 'prediction': pred, 'prediction_text': pred_text, 'score': score, 'score_source': score_source, 'response_text': text, 'latency_sec': time.time() - t0, 'attempt': attempt, 'error': None}
        except Exception as e:
            last_err = repr(e)
            time.sleep(1.5 * attempt)
    return {'idx': idx, 'sample_id': row.get('sample_id'), 'label': int(row.get('label')), 'prediction': None, 'prediction_text': 'request_failed', 'score': 0.5, 'score_source': 'request_failed', 'response_text': '', 'latency_sec': None, 'attempt': max_retries, 'error': last_err}

def auc(labels, scores):
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    n = len(pairs)
    npos = sum(labels)
    nneg = n - npos
    if npos == 0 or nneg == 0:
        return None
    rank_sum = 0.0
    i = 0
    while i < n:
        j = i + 1
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            if pairs[k][1] == 1:
                rank_sum += avg_rank
        i = j
    return (rank_sum - npos * (npos + 1) / 2.0) / (npos * nneg)
