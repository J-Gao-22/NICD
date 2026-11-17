TARGET_N = 1000
from lifecycle_rec.teacher.states import build_segments_recent30, item_terms, common_terms, read_jsonl
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

def key(row):
    uid = int(row.get('user_id', row.get('uid')))
    iid = int(row.get('target_item_id', row.get('gold_item_id', row.get('item_id', row.get('iid')))))
    return (uid, iid)

def sample_id(uid, iid):
    return f'u{uid}_target_{iid}'

def stable_hash(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()[:16]

def bucket(n, edges):
    for i, e in enumerate(edges):
        if n <= e:
            return str(i)
    return str(len(edges))

def row_triples(row):
    hist_items = row.get('history_item_ids', row.get('hist_item_ids', []))
    hist_ratings = row.get('history_ratings', row.get('hist_ratings', row.get('his_label', [])))
    hist_ts = row.get('history_timestamps', row.get('hist_timestamps', []))
    triples = []
    for idx, iid in enumerate(hist_items):
        try:
            iid_i = int(iid)
        except Exception:
            continue
        if iid_i == 0:
            continue
        rating = hist_ratings[idx] if idx < len(hist_ratings) else None
        ts = hist_ts[idx] if idx < len(hist_ts) else idx
        triples.append((iid_i, rating, ts))
    return triples[-30:]

def term_signature(seg_item_ids, items, top_k=3):
    counts = Counter()
    for iid in seg_item_ids:
        for t in item_terms(items.get(int(iid), {})):
            t = str(t).lower().strip()
            if t:
                counts[t] += 1
    return [t for t, _ in counts.most_common(top_k)]

def summarize_candidate(row, items, embeddings):
    uid, iid = key(row)
    triples = row_triples(row)
    if not triples:
        return None
    segs = build_segments_recent30(triples, embeddings)
    if not segs:
        return None
    blocks = []
    sig_parts = []
    seg_ranges = []
    for si, seg_obj in enumerate(segs, start=1):
        if isinstance(seg_obj, dict):
            a, b = (int(seg_obj['start']), int(seg_obj['end']))
        else:
            a, b = seg_obj
        seg_ranges.append([a, b])
        seg = triples[a:b]
        ids = [x[0] for x in seg]
        pos = sum((1 for _, r, _ in seg if r is not None and float(r) >= 1))
        neg = sum((1 for _, r, _ in seg if r is not None and float(r) < 1))
        if pos and neg:
            typ = 'M'
        elif pos:
            typ = 'P'
        elif neg:
            typ = 'N'
        else:
            typ = 'W'
        blocks.append(typ)
        terms = term_signature(ids, items)
        sig_parts.append(f'{si}:{typ}:len{bucket(len(seg), [4, 8, 12, 15])}:p{bucket(pos, [1, 3, 6, 10])}:n{bucket(neg, [1, 3, 6, 10])}:' + ','.join(terms))
    sig = stable_hash('|'.join(sig_parts))
    coarse = f"{len(segs)}seg_{''.join(blocks)}"
    return {'sample_id': sample_id(uid, iid), 'user_id': uid, 'item_id': iid, 'target_item_id': iid, 'split': row.get('split', 'train'), 'label': row.get('label'), 'rating': row.get('rating'), 'history_len_recent30': len(triples), 'n_segments': len(segs), 'segment_ranges_recent30': seg_ranges, 'segment_block_types': blocks, 'coarse_type': coarse, 'chain_signature_hash': sig, 'chain_signature_text': '|'.join(sig_parts)}

def choose_global1000(candidates):
    by_user = defaultdict(list)
    for c in candidates:
        by_user[c['user_id']].append(c)
    selected = []
    selected_keys = set()
    selected_sigs = set()
    for uid in sorted(by_user):
        rows = sorted(by_user[uid], key=lambda r: (-int(r['history_len_recent30']), -int(r['n_segments']), r['coarse_type'], int(r['item_id'])))
        r = dict(rows[0])
        r['selection_source'] = 'per_user_longest'
        selected.append(r)
        selected_keys.add((r['user_id'], r['item_id']))
        selected_sigs.add(r['chain_signature_hash'])
    remaining = [c for c in candidates if (c['user_id'], c['item_id']) not in selected_keys]
    by_type = defaultdict(list)
    for r in remaining:
        by_type[r['coarse_type']].append(r)
    for t in by_type:
        by_type[t].sort(key=lambda r: (r['chain_signature_hash'] in selected_sigs, -int(r['n_segments']), -int(r['history_len_recent30']), int(r['user_id']), int(r['item_id'])))
    type_counts = Counter((r['coarse_type'] for r in selected))
    while len(selected) < TARGET_N and by_type:
        types = sorted([t for t, rows in by_type.items() if rows], key=lambda t: (type_counts[t], -len(by_type[t]), t))
        if not types:
            break
        picked_any = False
        for t in types:
            if len(selected) >= TARGET_N:
                break
            rows = by_type[t]
            pick_idx = 0
            for i, r in enumerate(rows):
                if r['chain_signature_hash'] not in selected_sigs:
                    pick_idx = i
                    break
            r = dict(rows.pop(pick_idx))
            if (r['user_id'], r['item_id']) in selected_keys:
                continue
            r['selection_source'] = 'type_diversity_supplement'
            selected.append(r)
            selected_keys.add((r['user_id'], r['item_id']))
            selected_sigs.add(r['chain_signature_hash'])
            type_counts[t] += 1
            picked_any = True
        if not picked_any:
            break
    return selected[:TARGET_N]
