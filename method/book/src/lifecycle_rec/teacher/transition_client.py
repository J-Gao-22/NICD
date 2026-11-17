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
from lifecycle_rec.teacher import transition_support as edgev3
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
    return {'sample_id': pack.get('sample_id'), 'user_id': pack.get('user_id'), 'item_id': pack.get('item_id'), 'from_segment': pack.get('from_segment'), 'to_segment': pack.get('to_segment'), 'prompt_version': pack.get('prompt_version'), 'pack_hash': pack.get('pack_hash')}
