import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

def read_jsonl(path: Path):
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def last_token_pool(last_hidden_states, attention_mask):
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def mean_pool(last_hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden_states.dtype)
    summed = (last_hidden_states * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-06)
    return summed / denom

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-length', type=int, default=1024)
    parser.add_argument('--pooling', choices=['last', 'mean'], default='last')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', choices=['float16', 'bfloat16', 'float32'], default='float16')
    parser.add_argument('--trust-remote-code', action='store_true')
    args = parser.parse_args()
    input_path = Path(args.input)
    model_path = Path(args.model)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = list(read_jsonl(input_path))
    rows.sort(key=lambda r: int(r['item_id']))
    if args.limit and args.limit > 0:
        rows = rows[:args.limit]
    item_ids = [int(r['item_id']) for r in rows]
    titles = [r.get('title') for r in rows]
    texts = [r.get('text', '') for r in rows]
    dtype_map = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=args.trust_remote_code, padding_side='left')
    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=args.trust_remote_code)
    raw_config_path = model_path / 'config.json'
    if raw_config_path.exists():
        with raw_config_path.open('r', encoding='utf-8') as f:
            raw_config = json.load(f)
        for key in ('rope_theta', 'sliding_window', 'max_window_layers', 'use_sliding_window', 'is_causal'):
            if key in raw_config and (not hasattr(config, key)):
                setattr(config, key, raw_config[key])
    model = AutoModel.from_pretrained(str(model_path), config=config, trust_remote_code=args.trust_remote_code, torch_dtype=dtype_map[args.dtype])
    model.to(device)
    model.eval()
    all_embeddings = []
    started = time.time()
    with torch.inference_mode():
        for start in range(0, len(texts), args.batch_size):
            batch_texts = texts[start:start + args.batch_size]
            encoded = tokenizer(batch_texts, max_length=args.max_length, padding=True, truncation=True, return_tensors='pt')
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            if args.pooling == 'last':
                emb = last_token_pool(outputs.last_hidden_state, encoded['attention_mask'])
            else:
                emb = mean_pool(outputs.last_hidden_state, encoded['attention_mask'])
            emb = F.normalize(emb, p=2, dim=1)
            all_embeddings.append(emb.detach().cpu().to(torch.float32).numpy())
            if (start // args.batch_size + 1) % 20 == 0:
                print(f'encoded={min(start + args.batch_size, len(texts))}/{len(texts)} elapsed={time.time() - started:.1f}s', flush=True)
    embeddings = np.concatenate(all_embeddings, axis=0) if all_embeddings else np.zeros((0, 0), dtype=np.float32)
    np.save(outdir / 'item_embeddings.npy', embeddings.astype(np.float32))
    with (outdir / 'item_ids.json').open('w', encoding='utf-8') as f:
        json.dump(item_ids, f, ensure_ascii=False)
        f.write('\n')
    with (outdir / 'item_titles.json').open('w', encoding='utf-8') as f:
        json.dump(titles, f, ensure_ascii=False)
        f.write('\n')
    meta = {'input': str(input_path), 'model': str(model_path), 'rows': len(rows), 'embedding_shape': list(embeddings.shape), 'dtype_saved': 'float32', 'normalized_l2': True, 'pooling': args.pooling, 'max_length': args.max_length, 'batch_size': args.batch_size, 'device': str(device), 'torch_dtype': args.dtype, 'elapsed_seconds': time.time() - started, 'outputs': {'item_embeddings': 'item_embeddings.npy', 'item_ids': 'item_ids.json', 'item_titles': 'item_titles.json', 'embedding_meta': 'embedding_meta.json'}}
    with (outdir / 'embedding_meta.json').open('w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write('\n')
    print(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True))
if __name__ == '__main__':
    main()
