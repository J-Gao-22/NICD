import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from lifecycle_rec.prepare.sampling_extension import make_candidate, within_type_key
from lifecycle_rec.prepare.sampling_initial import choose_global1000, summarize_candidate
from lifecycle_rec.teacher.states import load_embeddings, load_items, read_jsonl

def extend(candidates, previous, size, prioritize_segments=False):
    existing_samples = {r['sample_id'] for r in previous}
    signatures = Counter((r['chain_signature_hash'] for r in previous))
    users = Counter((int(r['user_id']) for r in previous))
    previous_types = Counter((r['coarse_type'] for r in previous))
    pool = [r for r in candidates if r['sample_id'] not in existing_samples]
    type_counts = Counter((r['coarse_type'] for r in pool))
    groups = defaultdict(list)
    for r in pool:
        groups[r['coarse_type']].append(r)
    order = sorted(groups, key=lambda t: (previous_types[t], type_counts[t], t))
    for group in groups.values():
        group.sort(key=lambda r: within_type_key(r, signatures, users))
    selected, selected_ids = ([], set())
    changed = True
    while len(selected) < size and changed:
        changed = False
        for typ in order:
            if len(selected) >= size:
                break
            chosen = next((r for r in groups[typ] if r['sample_id'] not in selected_ids and signatures[r['chain_signature_hash']] == 0), None)
            if chosen is None:
                continue
            selected.append(chosen)
            selected_ids.add(chosen['sample_id'])
            signatures[chosen['chain_signature_hash']] += 1
            users[chosen['user_id']] += 1
            changed = True
    while len(selected) < size:

        def key(r):
            value = (previous_types[r['coarse_type']], signatures[r['chain_signature_hash']], users[r['user_id']])
            if prioritize_segments:
                value += (-r['n_segments'],)
            return value + (r['coarse_type'], r['user_id'], r['item_id'])
        chosen = min((r for r in pool if r['sample_id'] not in selected_ids), key=key, default=None)
        if chosen is None:
            break
        selected.append(chosen)
        selected_ids.add(chosen['sample_id'])
        signatures[chosen['chain_signature_hash']] += 1
        users[chosen['user_id']] += 1
    if len(selected) != size:
        raise ValueError(('Insufficient candidates', len(selected), size))
    return selected

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--all-interactions', type=Path, required=True)
    p.add_argument('--item-text-dir', type=Path, required=True)
    p.add_argument('--embedding-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    items = load_items(args.item_text_dir)
    embeddings = load_embeddings(args.embedding_dir)
    if not items or not embeddings:
        raise ValueError('Missing item inputs')
    rows = [r for r in read_jsonl(args.all_interactions) if r.get('split') == 'train']
    initial = [r for row in rows if (r := summarize_candidate(row, items, embeddings)) is not None]
    first = choose_global1000(initial)
    candidates = [r for row in rows if (r := make_candidate(row, items, embeddings)) is not None]
    second = extend(candidates, first, 300)
    third = extend(candidates, first + second, 700, prioritize_segments=True)
    selected = first + second + third
    assert len(selected) == len({(r['user_id'], r['item_id']) for r in selected}) == 2000
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as f:
        for r in selected:
            f.write(json.dumps({k: v for k, v in r.items() if k != 'selection_source'}, ensure_ascii=False) + '\n')
if __name__ == '__main__':
    main()
