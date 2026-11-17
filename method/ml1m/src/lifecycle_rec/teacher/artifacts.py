import argparse
import json
from pathlib import Path

def read(path):
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

def unique(rows, keys):
    index = {}
    for r in rows:
        key = tuple((r.get(k) for k in keys))
        if key in index and index[key] != r:
            raise ValueError(('Conflicting duplicate', key))
        index[key] = r
    return list(index.values())

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--state-dir', type=Path, required=True)
    p.add_argument('--transition-dirs', type=Path, nargs='+', required=True)
    p.add_argument('--transition-packs', type=Path, required=True)
    p.add_argument('--analysis-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    pack_file = (args.state_dir / 'state_packs.jsonl').resolve()
    states = [dict(r, source_pack_file=str(pack_file)) for r in read(args.state_dir / 'state_candidates.jsonl')]
    state_keys = ('user_id', 'item_id', 'segment_id', 'candidate_id')
    edge_keys = ('sample_id', 'from_segment', 'to_segment', 'edge_candidate_id')
    write(args.output_dir / 'state_candidates.jsonl', unique(states, state_keys))
    write(args.output_dir / 'state_logprobs.jsonl', unique(read(args.state_dir / 'state_logprobs.jsonl'), state_keys))
    write(args.output_dir / 'transition_packs.jsonl', read(args.transition_packs))
    write(args.output_dir / 'analysis_inputs/transition_packages.jsonl', read(args.analysis_dir / 'transition_packages.jsonl'))
    for name in ('transition_candidates.jsonl', 'transition_logprobs.jsonl'):
        rows = [r for directory in args.transition_dirs for r in read(directory / name)]
        write(args.output_dir / name, unique(rows, edge_keys))
if __name__ == '__main__':
    main()
