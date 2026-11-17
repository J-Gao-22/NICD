from __future__ import annotations
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any
import numpy as np

def jdump(x):
    return json.dumps(x, ensure_ascii=False, separators=(',', ':'))

def sha(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def read_jsonl(p: Path):
    with p.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def parse_terms(text: str) -> list[str]:
    m = re.search('Semantic interest terms:\\s*(.*?)(?:\\n|$)', text or '')
    if m:
        return [x.strip() for x in re.split('[,;]', m.group(1)) if x.strip()][:10]
    return []

def parse_genres(text: str) -> list[str]:
    m = re.search('Genres:\\s*(.*?)(?:\\n|$)', text or '')
    return [x.strip() for x in re.split('[,;]', m.group(1)) if x.strip()][:5] if m else []

def parse_people(text: str):
    d = 'none'
    actors = []
    m = re.search('directed by ([^;\\n\\.]+)', text or '')
    if m:
        d = m.group(1).strip()
    m = re.search('starring ([^\\.\\n]+)', text or '')
    if m:
        actors = [x.strip() for x in m.group(1).split(',')[:2] if x.strip()]
    return (d, actors[:2])

def load_items(root: Path):
    compact = root / 'item_text_semantic_compact.jsonl'
    balanced = root / 'item_text_balanced_semantic.jsonl'
    items = {}
    for r in read_jsonl(compact):
        iid = int(r['item_id'])
        items[iid] = {'title': r.get('title', ''), 'semantic_terms': parse_terms(r.get('text', ''))}
    for r in read_jsonl(balanced):
        iid = int(r['item_id'])
        it = items.setdefault(iid, {'title': r.get('title', ''), 'semantic_terms': []})
        it['genres'] = parse_genres(r.get('text', ''))
        it['director'], it['actors'] = parse_people(r.get('text', ''))
    return items

def load_embeddings(root: Path):
    emb_dir = root
    emb_path = emb_dir / 'item_embeddings.npy'
    ids_path = emb_dir / 'item_ids.json'
    if not emb_path.exists() or not ids_path.exists():
        return {}
    arr = np.load(emb_path, mmap_mode='r')
    ids = json.loads(ids_path.read_text())
    return {int(iid): arr[idx] for idx, iid in enumerate(ids)}

def _mean_vec(ids, embeddings):
    vecs = [embeddings.get(int(i)) for i in ids if int(i) in embeddings]
    if not vecs:
        return None
    v = np.mean(np.stack(vecs, axis=0), axis=0)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v

def _cos_dist(a, b):
    if a is None or b is None:
        return 0.0
    return max(0.0, min(1.0, 1.0 - float(np.dot(a, b))))

def _boundary_scores(triples, embeddings):
    n = len(triples)
    ids = [int(x[0]) for x in triples]
    labs = [int(x[1]) for x in triples]
    ts = [float(x[2] or 0) for x in triples]
    raw_gaps = [max(0.0, ts[i] - ts[i - 1]) for i in range(1, n)]
    log_gaps = [math.log1p(x) for x in raw_gaps]
    max_gap = max(log_gaps) if log_gaps else 0.0
    scores = {}
    for b in range(1, n):
        lo = max(0, b - 4)
        hi = min(n, b + 4)
        left_ids = ids[lo:b]
        right_ids = ids[b:hi]
        sem = _cos_dist(_mean_vec(left_ids, embeddings), _mean_vec(right_ids, embeddings))
        left_pol = sum(labs[lo:b]) / max(1, b - lo)
        right_pol = sum(labs[b:hi]) / max(1, hi - b)
        polarity = abs(left_pol - right_pol)
        time_score = log_gaps[b - 1] / max_gap if max_gap > 0 else 0.0
        score = 0.4 * sem + 0.35 * polarity + 0.25 * time_score
        scores[b] = {'score': score, 'semantic': sem, 'polarity': polarity, 'time': time_score}
    return scores

def _valid_cuts(n, k):
    if k == 1:
        return [()] if n <= 15 else []
    out = []
    if k == 2:
        for c1 in range(4, n - 3):
            lens = [c1, n - c1]
            if all((4 <= x <= 15 for x in lens)):
                out.append((c1,))
    elif k == 3:
        for c1 in range(4, n - 7):
            for c2 in range(c1 + 4, n - 3):
                lens = [c1, c2 - c1, n - c2]
                if all((4 <= x <= 15 for x in lens)):
                    out.append((c1, c2))
    return out

def build_segments_recent30(triples, embeddings, t1=0.36, t2=0.66):
    n = len(triples)
    if n == 0:
        return []
    if n <= 8:
        cuts = [n]
    else:
        scores = _boundary_scores(triples, embeddings)
        best = {}
        for k in (1, 2, 3):
            candidates = _valid_cuts(n, k)
            if not candidates:
                continue
            if k == 1:
                best[k] = ((), 0.0)
            else:
                best[k] = max(((cs, sum((scores[c]['score'] for c in cs)) / len(cs)) for cs in candidates), key=lambda x: x[1])
        min_k = 1 if n <= 15 else 2
        if 3 in best and best[3][1] >= t2:
            chosen = best[3][0]
        elif 2 in best and (min_k == 2 or best[2][1] >= t1):
            chosen = best[2][0]
        elif min_k == 1:
            chosen = ()
        else:
            chosen = best[2][0]
        cuts = list(chosen) + [n]
    segs = []
    prev = 0
    for si, c in enumerate(cuts, 1):
        if len(cuts) == 1:
            pos = 'only'
        elif si == 1:
            pos = 'first'
        elif si == len(cuts):
            pos = 'last'
        else:
            pos = 'middle'
        segs.append({'segment_id': f'S{si}', 'start': prev, 'end': c, 'position': pos})
        prev = c
    return segs

def term_norm(t):
    t = str(t).lower().strip()
    t = re.sub('[^a-z0-9 -]+', ' ', t)
    return re.sub('\\s+', ' ', t).strip()

def item_terms(it):
    return [term_norm(x) for x in it.get('genres', [])[:2] + it.get('semantic_terms', [])[:10] if term_norm(x)]

def common_terms(ids, labs, items, pos=True):
    own = [iid for iid, l in zip(ids, labs) if (l == 1) == pos]
    other = [iid for iid, l in zip(ids, labs) if (l == 1) != pos]
    c = Counter()
    oc = Counter()
    for iid in own:
        c.update(set(item_terms(items.get(int(iid), {}))))
    for iid in other:
        oc.update(set(item_terms(items.get(int(iid), {}))))
    out = []
    for term, n in c.most_common(8):
        if n < 2 and len(own) > 3:
            continue
        role = 'positive_discriminative' if pos and oc[term] == 0 else 'negative_rejection_signal' if not pos and oc[term] == 0 else 'shared_non_discriminative'
        out.append((term, n, len(own), oc[term], len(other), role))
    return out[:6]

def render_item(iid: int, label: int, order: int, rating: int, items: dict[int, dict[str, Any]]):
    it = items.get(int(iid), {})
    return '\n'.join([f"- title: {it.get('title', 'unknown')}", f'  rating: {rating}', f'  order: {order}', f"  genres: {', '.join(it.get('genres', [])[:5]) or 'none'}", f"  semantic_terms: {', '.join(it.get('semantic_terms', [])[:10]) or 'none'}", f"  director: {it.get('director', 'none')}", f"  actors: {', '.join(it.get('actors', [])[:2]) or 'none'}"])

def block_hint(labs):
    p = sum(labs)
    n = len(labs) - p
    if p >= 2 and n <= 1:
        return 'positive_preference_state'
    if n >= 2 and p <= 1:
        return 'negative_rejection_state'
    return 'mixed_feedback_state'

def render_pack(row, segment, items):
    a, b = (segment['start'], segment['end'])
    ids = row['history_item_ids'][a:b]
    labs = row['his_label'][a:b]
    pos = [(i, l) for i, l in zip(ids, labs) if int(i) != 0 and int(l) == 1]
    neg = [(i, l) for i, l in zip(ids, labs) if int(i) != 0 and int(l) == 0]
    lines = ['[SEGMENT CONTEXT]', f"segment_id: {segment['segment_id']}", f"segment_position: {segment['position']}", f'segment_length: {len(ids)}', f'positive_count: {len(pos)}', f'negative_count: {len(neg)}', f'feedback_state_hint: {block_hint([int(x) for x in labs])}', '', '[POSITIVE INTERACTIONS]']
    if pos:
        for j, (iid, _) in enumerate(pos, 1):
            lines.append(render_item(iid, 1, j, 4, items))
    else:
        lines.append('none')
    lines += ['', '[NEGATIVE INTERACTIONS]']
    if neg:
        for j, (iid, _) in enumerate(neg, 1):
            lines.append(render_item(iid, 0, j, 3, items))
    else:
        lines.append('none')
    lines += ['', '[POSITIVE COMMON TERMS]']
    pct = common_terms(ids, [int(x) for x in labs], items, True)
    lines += [f'- {t} | support={n}/{den} | negative_overlap={on}/{oden} | role={role}' for t, n, den, on, oden, role in pct] or ['none']
    lines += ['', '[NEGATIVE COMMON TERMS]']
    nct = common_terms(ids, [int(x) for x in labs], items, False)
    lines += [f'- {t} | support={n}/{den} | positive_overlap={on}/{oden} | role={role}' for t, n, den, on, oden, role in nct] or ['none']
    return '\n'.join(lines)
