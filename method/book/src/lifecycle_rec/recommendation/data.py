import argparse
import random
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer
from lifecycle_rec.recommendation.model import read_jsonl,encode_records
from lifecycle_rec.recommendation.identity import hybrid_explicit


def schedule(chain_count,plain_count):
    rng=random.Random(20260710)
    chain=list(range(chain_count));plain=list(range(plain_count))
    rng.shuffle(chain);rng.shuffle(plain)
    result=[]
    for position,index in enumerate(chain):
        result.append(('chain',index))
        if (position+1)%4==0 or position==chain_count-1:
            result.append(('plain',plain[position//4%plain_count]))
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--frame-dir',type=Path,required=True)
    p.add_argument('--base-model',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    tokenizer=AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token=tokenizer.pad_token or tokenizer.eos_token
    raw=read_jsonl(args.data_dir/'train_plain_yesno.jsonl')
    chains=read_jsonl(args.data_dir/'train_chain_yesno.jsonl')
    random.Random(20260709+911).shuffle(raw)
    raw=raw[:(len(chains)+3)//4]
    datasets=dict(train_plain=encode_records(tokenizer,raw,2048),train_chain=encode_records(tokenizer,chains,2048))
    assert len(datasets['train_chain'])==643308 and len(datasets['train_plain'])==160827
    for split,count in [('valid',56402),('test',56403)]:
        chain=encode_records(tokenizer,read_jsonl(args.data_dir/(split+'_chain_yesno.jsonl')),2048)
        plain=encode_records(tokenizer,read_jsonl(args.data_dir/(split+'_plain_yesno.jsonl')),2048)
        frame=pd.read_pickle(args.frame_dir/(split+'_ood2.pkl')).reset_index(drop=True)
        datasets[split]=hybrid_explicit(chain,plain,frame)
        assert len(datasets[split])==count
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(datasets=datasets,schedule=schedule(len(datasets['train_chain']),len(datasets['train_plain'])),pad_token_id=tokenizer.pad_token_id),args.output)


if __name__=='__main__':main()
