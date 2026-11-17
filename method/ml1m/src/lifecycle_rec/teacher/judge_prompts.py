import argparse
from collections import defaultdict
from pathlib import Path
from lifecycle_rec.paths import PROMPTS
from lifecycle_rec.teacher.artifacts import read, write
from lifecycle_rec.teacher.render import render_edge, render_state, seg_ord
from lifecycle_rec.teacher.targets import render_item

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sample-list', type=Path, required=True)
    p.add_argument('--source-dir', type=Path, required=True)
    p.add_argument('--item-text-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--prompt', type=Path, default=PROMPTS/'judge.txt')
    args = p.parse_args()
    items = {}
    for name in ('item_text_semantic_compact.jsonl', 'item_text_balanced_semantic.jsonl', 'item_text_full.jsonl'):
        for r in read(args.item_text_dir / name):
            iid = int(r.get('item_id') or r.get('iid') or r.get('movie_id'))
            items.setdefault(iid, {}).update({k: v for k, v in r.items() if v not in (None, '', [])})
    states = defaultdict(dict)
    for r in read(args.source_dir / 'state_candidates.jsonl'):
        if r['candidate_id'] == 'C1':
            sid = f"u{r['user_id']}_target_{r['item_id']}"
            states[sid][r['segment_id']] = dict(segment_id=r['segment_id'], state_node=r['model_output']['candidate'])
    edges = {}
    for r in read(args.source_dir / 'transition_candidates.jsonl'):
        if (r.get('edge_candidate_id') or r.get('generation_variant')) == 'E1':
            edges[r['sample_id'], r['from_segment'], r['to_segment']] = r
    system = args.prompt.read_text(encoding='utf-8')
    prompts = []
    for sample in read(args.sample_list):
        uid = int(sample['user_id'])
        iid = int(sample.get('target_item_id', sample.get('item_id')))
        sid = f'u{uid}_target_{iid}'
        segments = sorted(states[sid], key=seg_ord)
        if not segments:
            raise ValueError(('Missing states', sid))
        parts = ['Lifecycle interest chain before the target:\n']
        for i, segment in enumerate(segments):
            parts.append(render_state(states[sid][segment]))
            if i + 1 < len(segments):
                parts.append(render_edge(edges[sid, segment, segments[i + 1]]))
        target = render_item(iid, items[iid], dict(sample, target_item_id=iid))
        user = '\n\n'.join(parts) + '\n\nTarget movie:\n' + target + "\n\nQuestion:\nBased on the ordered lifecycle interest chain and the target movie's semantic profile, would the user have a positive interaction with the target movie?\n\nAnswer:"
        prompts.append(dict(sample_id=sid, user_id=uid, target_item_id=iid, label=sample['label'], rating=sample.get('rating'), num_segments=len(segments), num_edges=len(segments) - 1, messages=[dict(role='system', content=system), dict(role='user', content=user)]))
    write(args.output_dir / 'state_edge_prompts.jsonl', prompts)
if __name__ == '__main__':
    main()
