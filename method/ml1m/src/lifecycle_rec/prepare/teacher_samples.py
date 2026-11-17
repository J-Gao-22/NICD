import argparse
import hashlib
import json
from pathlib import Path

def read(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]

def select(interactions, manifest, expected_count, expected_sha256):
    if len(manifest) != expected_count:
        raise ValueError(('Sample count mismatch', len(manifest), expected_count))
    keys = []
    for row in manifest:
        key = tuple((row[name] for name in ('user_id', 'item_id', 'label')))
        if any((type(value) is not int for value in key)) or key[2] not in (0, 1):
            raise ValueError(('Invalid sample key', key))
        keys.append(key)
    if len({key[:2] for key in keys}) != expected_count:
        raise ValueError('Duplicate sample')
    digest = hashlib.sha256(json.dumps(keys, separators=(',', ':')).encode()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(('Sample order or identity mismatch', digest))
    wanted = {key[:2] for key in keys}
    index = {}
    for row in interactions:
        key = (int(row['user_id']), int(row['item_id']))
        if row.get('split') != 'train' or key not in wanted:
            continue
        if key in index:
            raise ValueError(('Ambiguous interaction', key))
        index[key] = row
    selected = []
    for entry, key in zip(manifest, keys):
        row = index.get(key[:2])
        if row is None or int(row['label']) != key[2]:
            raise ValueError(('Sample not found in training split', key))
        if 'rating' in entry and row.get('rating') != entry['rating']:
            raise ValueError(('Sample rating mismatch', key))
        selected.append(dict(row))
    return selected

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all-interactions', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--expected-count', type=int, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    selected = select(read(args.all_interactions), read(args.manifest), args.expected_count, args.expected_sha256)
    if args.output.resolve() in (args.manifest.resolve(), args.all_interactions.resolve()):
        raise ValueError('Output overwrites an input')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8', newline='\n') as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
if __name__ == '__main__':
    main()
