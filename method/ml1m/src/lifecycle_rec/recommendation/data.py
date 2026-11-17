import argparse
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from lifecycle_rec.recommendation.schedule import mainline_schedule

def encode(tokenizer, path):
    rows = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            text = tokenizer.apply_chat_template([dict(role='system', content=r['system']), dict(role='user', content=r['user'])], tokenize=False, add_generation_prompt=True)
            ids = tokenizer(text, add_special_tokens=False, truncation=False).input_ids
            if len(ids) <= 2048:
                rows.append({k: r[k] for k in ('uid', 'iid', 'label', 'row_index', 'record_id', 'input_type')} | dict(_ids=ids, _length=len(ids)))
    return rows

def combine(chain, plain):
    index = {int(r['row_index']): r for r in plain}
    if len(index) != len(plain):
        raise ValueError('Duplicate row_index')
    for r in chain:
        old = index[int(r['row_index'])]
        if (old['uid'], old['iid'], old['label']) != (r['uid'], r['iid'], r['label']):
            raise ValueError('Target identity mismatch')
        index[int(r['row_index'])] = r
    return [index[k] for k in sorted(index)]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--transition-dir', type=Path, required=True)
    p.add_argument('--base-model', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    if tokenizer.encode('Yes', add_special_tokens=False) != [9454]:
        raise ValueError('Yes token mismatch')
    datasets = {}
    for split in ('train', 'valid', 'test'):
        for kind in ('chain', 'plain'):
            root = args.transition_dir if split == 'test' else args.data_dir
            datasets[f'{split}_{kind}'] = encode(tokenizer, root / f'{split}_{kind}_yesno.jsonl')
    for split, size in (('valid', 10401), ('test', 7331)):
        datasets[split] = combine(datasets[f'{split}_chain'], datasets[f'{split}_plain'])
        if len(datasets[split]) != size:
            raise ValueError((split, len(datasets[split]), size))
    if (len(datasets['train_chain']), len(datasets['train_plain'])) != (32864, 33891):
        raise ValueError('Training record count mismatch')
    plan = mainline_schedule(len(datasets['train_chain']), len(datasets['train_plain']))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(datasets=datasets, schedule=plan, pad_token_id=tokenizer.pad_token_id), args.output)
if __name__ == '__main__':
    main()
