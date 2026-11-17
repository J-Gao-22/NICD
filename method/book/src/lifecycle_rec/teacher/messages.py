import argparse
import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer
from lifecycle_rec.prepare import evidence
from lifecycle_rec.prepare.message_format import revise_row


def render_records(row, items, embeddings, template, tokenizer):
    base=evidence.base
    records=[]
    target=items[int(row['item_id'])]
    segments,_=base.build_segments(row,embeddings)
    for segment in segments:
        pack=base.render_pack(row,segment,items,base.clean(target.get('title')))
        for candidate_id,lens,instruction in base.LENSES:
            system=template.replace('{{candidate_id}}',candidate_id).replace('{{analysis_lens}}',lens).replace('{{lens_instruction}}',instruction)
            record=dict(record_id=f"{row['sample_id']}::{segment['segment_id']}::{candidate_id}",sample_id=row['sample_id'],split='train',user_id=int(row['user_id']),item_id=int(row['item_id']),segment_id=segment['segment_id'],candidate_id=candidate_id,analysis_lens=lens,messages=[dict(role='system',content=system),dict(role='user',content=pack)])
            records.append(revise_row(record,tokenizer))
    return records


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--sample-list',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--expected-sha256',required=True)
    p.add_argument('--canonical-profile',type=Path,required=True)
    p.add_argument('--embedding-dir',type=Path,required=True)
    p.add_argument('--prompt',type=Path,required=True)
    p.add_argument('--tokenizer',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    payload=args.manifest.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=args.expected_sha256:raise ValueError('Manifest checksum mismatch')
    manifest=[json.loads(line) for line in payload.decode().splitlines() if line.strip()]
    rows=list(evidence.base.read_jsonl(args.sample_list))
    if len(rows)!=len(manifest) or len(rows)!=6000:raise ValueError('Teacher population mismatch')
    for row,entry in zip(rows,manifest):
        if row.get('split')!='train' or any(row.get(key)!=value for key,value in entry.items()):raise ValueError(('Teacher sample mismatch',entry['sample_id']))
    items={int(row['item_id']):row for row in evidence.base.read_jsonl(args.canonical_profile)}
    embeddings=evidence.base.load_embeddings(args.embedding_dir)
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,trust_remote_code=True)
    template=args.prompt.read_text(encoding='utf-8')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with (args.output_dir/'state_messages.jsonl').open('w',encoding='utf-8',newline='\n') as stream:
        for row in rows:
            for record in render_records(row,items,embeddings,template,tokenizer):stream.write(json.dumps(record,ensure_ascii=False)+'\n')


if __name__=='__main__':main()
