import argparse
from pathlib import Path
import pandas as pd
from lifecycle_rec.recommendation.plain_inputs import ItemRenderer, build_plain_records, load_id2title, load_semantic_cache, load_text_variant, write_jsonl

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--compact', type=Path, required=True)
    p.add_argument('--balanced', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    id2title = load_id2title(args.data_dir / 'id2title.json')
    renderer = ItemRenderer(id2title=id2title, cache=load_semantic_cache(args.cache), compact=load_text_variant(args.compact), balanced=load_text_variant(args.balanced))
    for split in ('train', 'valid', 'test'):
        frame = pd.read_pickle(args.data_dir / f'{split}_ood2.pkl')
        records, _ = build_plain_records(frame, split, renderer)
        write_jsonl(args.output_dir / f'{split}_plain_yesno.jsonl', records)
if __name__ == '__main__':
    main()
