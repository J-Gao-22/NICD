import argparse
import hashlib
import json
from pathlib import Path
from lifecycle_rec.paths import PROMPTS
from lifecycle_rec.teacher.states import build_segments_recent30, load_embeddings, load_items, read_jsonl, render_pack

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--all-interactions', type=Path, required=True)
    p.add_argument('--item-text-dir', type=Path, required=True)
    p.add_argument('--embedding-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--prompt', type=Path, default=PROMPTS/'states.txt')
    args = p.parse_args()
    items = load_items(args.item_text_dir)
    embeddings = load_embeddings(args.embedding_dir)
    system = args.prompt.read_text(encoding='utf-8')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    handles = {s: (args.output_dir / f'{s}_state_segment_messages.jsonl').open('w', encoding='utf-8') for s in ('train', 'valid', 'test')}
    try:
        for r in read_jsonl(args.all_interactions):
            if r.get('split') not in handles:
                continue
            triples = [(i, int(y), t) for i, y, t in zip(r['history_item_ids'], r['his_label'], r['history_timestamps']) if int(i) != 0][-30:]
            if not triples:
                continue
            row = dict(r, history_item_ids=[x[0] for x in triples], his_label=[x[1] for x in triples])
            segments = build_segments_recent30(triples, embeddings)
            for segment in segments:
                pack = render_pack(row, segment, items)
                entry = dict(split=r['split'], sample_id=f"u{r['user_id']}_target_{r['item_id']}", user_id=r['user_id'], item_id=r['item_id'], segment_id=segment['segment_id'], segment_position=segment.get('position'), source_label=r['label'], rating=r.get('rating'), history_recent30_len=len(triples), num_segments=len(segments), pack_hash=hashlib.sha256(pack.encode()).hexdigest(), messages=[dict(role='system', content=system), dict(role='user', content=pack)])
                handles[r['split']].write(json.dumps(entry, ensure_ascii=False) + '\n')
    finally:
        for f in handles.values():
            f.close()
if __name__ == '__main__':
    main()
