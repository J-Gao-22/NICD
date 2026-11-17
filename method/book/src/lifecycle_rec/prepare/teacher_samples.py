import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--all-interactions',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--expected-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    payload=args.manifest.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=args.expected_sha256:raise ValueError('Manifest checksum mismatch')
    entries=[json.loads(line) for line in payload.decode('utf-8').splitlines() if line.strip()]
    if len(entries)!=6000 or len({r['sample_id'] for r in entries})!=6000:raise ValueError('Invalid sample population')
    wanted={r['sample_id'] for r in entries}
    index={}
    with args.all_interactions.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():continue
            row=json.loads(line)
            if row['sample_id'] not in wanted:continue
            if row['sample_id'] in index:raise ValueError('Duplicate event')
            index[row['sample_id']]=row
    selected=[]
    for entry in entries:
        row=index.get(entry['sample_id'])
        if row is None or row['split']!='train' or any(row.get(key)!=value for key,value in entry.items()):raise ValueError(('Sample mismatch',entry['sample_id']))
        selected.append(row)
    if args.output.resolve() in (args.all_interactions.resolve(),args.manifest.resolve()):raise ValueError('Output overwrites input')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',encoding='utf-8',newline='\n') as stream:
        for row in selected:stream.write(json.dumps(row,ensure_ascii=False)+'\n')


if __name__=='__main__':main()
