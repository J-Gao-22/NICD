import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--packs', type=Path, nargs='+', required=True)
    parser.add_argument('--sample-list', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    samples = {(int(row['user_id']), int(row['item_id'])) for row in read(args.sample_list)}
    index = {}
    for path in args.packs:
        for row in read(path):
            key = (int(row['user_id']), int(row['item_id']), str(row['segment_id']), str(row['candidate_id']))
            if key[:2] not in samples:
                continue
            digest = hashlib.sha256(row['pack_text'].encode('utf-8')).hexdigest()
            if digest != (row.get('pack_hash') or row.get('source_pack_hash')):
                raise ValueError(('Pack hash mismatch', key))
            if key in index and index[key]['pack_text'] != row['pack_text']:
                raise ValueError(('Conflicting pack', key))
            index[key] = dict(user_id=key[0], item_id=key[1], segment_id=key[2], candidate_id=key[3], pack_hash=digest, pack_text=row['pack_text'])
    if {key[:2] for key in index} != samples:
        raise ValueError('Teacher sample population mismatch')
    segments = {}
    for key in index:
        segments.setdefault(key[:3], set()).add(key[3])
    if any(candidates != {'C1', 'C2', 'C3'} for candidates in segments.values()):
        raise ValueError('Incomplete candidate packs')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        for row in index.values():
            stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')


if __name__ == '__main__':
    main()
