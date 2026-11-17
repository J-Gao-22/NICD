from lifecycle_rec.teacher.states import build_segments_recent30, item_terms, common_terms, read_jsonl
import collections
import hashlib
import json
import sys
from pathlib import Path

def sample_id(row):
    return row.get('sample_id') or f"u{row['user_id']}_target_{row.get('target_item_id', row.get('item_id'))}"

def block_type(labels):
    p = sum((1 for x in labels if int(x) == 1))
    n = sum((1 for x in labels if int(x) == 0))
    if p == 0 and n == 0:
        return 'W'
    if p >= 2 and n <= 1:
        return 'P'
    if n >= 2 and p <= 1:
        return 'N'
    return 'M'

def segment_signature(row, segment, items):
    a, b = (segment['start'], segment['end'])
    ids = [int(x) for x in row['history_item_ids'][a:b] if int(x) != 0]
    labels = [int(x) for x in row['his_label'][a:b]][:len(ids)]
    typ = block_type(labels)
    pos = sum(labels)
    neg = len(labels) - pos
    terms = []
    for term, *_ in common_terms(ids, labels, items, True):
        terms.append(term)
    for term, *_ in common_terms(ids, labels, items, False):
        if term not in terms:
            terms.append(term)
    terms = terms[:3]
    return f"{segment['segment_id']}:{typ}:len{len(ids)}:p{pos}:n{neg}:{','.join(terms)}"

def make_candidate(row, items, embeddings):
    triples = [(int(iid), int(label), float(ts or 0)) for iid, label, ts in zip(row.get('history_item_ids', []), row.get('his_label', []), row.get('history_timestamps', [])) if int(iid) != 0][-30:]
    if not triples:
        return None
    local = dict(row)
    local['history_item_ids'] = [x[0] for x in triples]
    local['his_label'] = [x[1] for x in triples]
    local['history_timestamps'] = [x[2] for x in triples]
    segments = build_segments_recent30(triples, embeddings)
    if not segments:
        return None
    sig_parts = [segment_signature(local, seg, items) for seg in segments]
    coarse = f'{len(segments)}seg_' + ''.join((part.split(':')[1] for part in sig_parts))
    sig_text = '|'.join(sig_parts)
    sig_hash = hashlib.sha256(sig_text.encode()).hexdigest()[:16]
    sid = sample_id(row)
    return {'sample_id': sid, 'user_id': int(row['user_id']), 'item_id': int(row['item_id']), 'target_item_id': int(row['item_id']), 'label': int(row['label']), 'rating': row.get('rating'), 'split': row.get('split'), 'source_index': row.get('source_index'), 'history_len_recent30': len(triples), 'n_segments': len(segments), 'segment_ranges_recent30': [[int(seg['start']), int(seg['end'])] for seg in segments], 'segment_block_types': [part.split(':')[1] for part in sig_parts], 'coarse_type': coarse, 'chain_signature_text': sig_text, 'chain_signature_hash': sig_hash}

def within_type_key(cand, selected_sig_counts, selected_user_counts):
    transitions = sum((1 for a, b in zip(cand['segment_block_types'], cand['segment_block_types'][1:]) if a != b))
    return (selected_sig_counts.get(cand['chain_signature_hash'], 0), selected_user_counts.get(cand['user_id'], 0), -transitions, -cand['n_segments'], -cand['history_len_recent30'], cand['user_id'], cand['item_id'])
