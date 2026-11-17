from __future__ import annotations
import argparse
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from lifecycle_rec.generators.transition_format import SYSTEM_PROMPT, USER_INSTRUCTION, validate_output

def read_jsonl(path: Path, limit: int | None=None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    tmp.replace(path)

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n')
    tmp.replace(path)

def extract_json(text: str) -> dict[str, Any] | None:
    s = text.strip()
    s = re.sub('^```(?:json)?\\s*', '', s, flags=re.I)
    s = re.sub('\\s*```$', '', s)
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else None
    except Exception:
        pass
    start = s.find('{')
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                try:
                    v = json.loads(s[start:i + 1])
                    return v if isinstance(v, dict) else None
                except Exception:
                    return None
    return None

def prompt_text(tokenizer: Any, row: dict[str, Any]) -> str:
    inp = row['input']
    user = USER_INSTRUCTION + '\n\n' + str(inp.get('pack_text') or '')
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def answer_text(tokenizer: Any, row: dict[str, Any]) -> str:
    return json.dumps(row['output'], ensure_ascii=False, sort_keys=True, separators=(',', ':')) + (tokenizer.eos_token or '')

class SftDataset(Dataset):

    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_seq_len: int) -> None:
        self.items: list[dict[str, Any]] = []
        self.skipped_too_long = 0
        for row in rows:
            p = prompt_text(tokenizer, row)
            a = answer_text(tokenizer, row)
            p_ids = tokenizer(p, add_special_tokens=False).input_ids
            ids = tokenizer(p + a, add_special_tokens=False).input_ids
            if len(ids) > max_seq_len:
                self.skipped_too_long += 1
                continue
            labels = [-100] * len(p_ids) + ids[len(p_ids):]
            self.items.append({'input_ids': torch.tensor(ids, dtype=torch.long), 'labels': torch.tensor(labels, dtype=torch.long), 'weight': float(row.get('sample_weight', 1.0)), 'record_id': row.get('record_id'), 'sample_id': row.get('sample_id'), 'from_segment': row.get('from_segment'), 'to_segment': row.get('to_segment')})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.items[idx]

def collate(batch: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    n = max((len(x['input_ids']) for x in batch))
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), n), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    weights = torch.tensor([x['weight'] for x in batch], dtype=torch.float32)
    meta = {k: [x.get(k) for x in batch] for k in ('record_id', 'sample_id', 'from_segment', 'to_segment')}
    for i, x in enumerate(batch):
        m = len(x['input_ids'])
        ids[i, :m] = x['input_ids']
        labels[i, :m] = x['labels']
        mask[i, :m] = 1
    return {'input_ids': ids, 'labels': labels, 'attention_mask': mask, 'weights': weights, **meta}

def weighted_loss(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    tok = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(shift_labels.size())
    m = shift_labels.ne(-100)
    per = (tok * m).sum(1) / m.sum(1).clamp_min(1)
    return (per * weights.to(per.device)).mean()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-model', default='Qwen/Qwen2.5-7B-Instruct')
    ap.add_argument('--corpus-dir', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--mode', choices=['smoke', 'train'], default='train')
    ap.add_argument('--max-seq-len', type=int, default=4096)
    ap.add_argument('--lr', type=float, default=0.0001)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--grad-accum', type=int, default=8)
    ap.add_argument('--lora-rank', type=int, default=16)
    ap.add_argument('--lora-alpha', type=int, default=32)
    ap.add_argument('--lora-dropout', type=float, default=0.05)
    ap.add_argument('--max-steps', type=int, default=None)
    ap.add_argument('--eval-max-new-tokens', type=int, default=768)
    ap.add_argument('--smoke-eval-limit', type=int, default=5)
    ap.add_argument('--skip-generation-eval', action='store_true')
    args = ap.parse_args()
    random.seed(42)
    torch.manual_seed(42)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    train_rows = read_jsonl(args.corpus_dir / 'edge_transition_sft_train.jsonl', 20 if args.mode == 'smoke' else None)
    valid_rows = read_jsonl(args.corpus_dir / 'edge_transition_sft_valid.jsonl', 10 if args.mode == 'smoke' else None)
    test_rows = read_jsonl(args.corpus_dir / 'edge_transition_sft_test_dedup.jsonl', args.smoke_eval_limit if args.mode == 'smoke' else None)
    train_ds = SftDataset(train_rows, tokenizer, args.max_seq_len)
    valid_ds = SftDataset(valid_rows, tokenizer, args.max_seq_len)
    if not len(train_ds):
        raise RuntimeError('no usable train rows')
    coll = lambda b: collate(b, tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=coll)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=coll) if len(valid_ds) else None
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype, device_map='auto', trust_remote_code=True)
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    model = get_peft_model(model, LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], bias='none', task_type='CAUSAL_LM'))
    model.train()
    model.print_trainable_parameters()
    opt = AdamW(model.parameters(), lr=args.lr)
    device = next(model.parameters()).device
    max_steps = 20 if args.mode == 'smoke' else args.max_steps
    total_steps = 0
    losses: list[float] = []
    events: list[dict[str, Any]] = []
    start = time.time()
    opt.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        for bi, batch in enumerate(train_loader):
            out = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))
            loss = weighted_loss(out.logits, batch['labels'].to(device), batch['weights'].to(device))
            (loss / args.grad_accum).backward()
            losses.append(float(loss.detach().cpu()))
            if (bi + 1) % args.grad_accum == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
                total_steps += 1
                if total_steps == 1 or total_steps % 5 == 0:
                    ev = {'step': total_steps, 'epoch': epoch, 'train_loss_recent': sum(losses[-20:]) / min(len(losses), 20)}
                    events.append(ev)
                    print(json.dumps(ev, ensure_ascii=False), flush=True)
                if max_steps and total_steps >= max_steps:
                    break
        if max_steps and total_steps >= max_steps:
            break
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    train_metrics = {'mode': args.mode, 'train_rows_loaded': len(train_rows), 'train_rows_used': len(train_ds), 'train_rows_skipped_too_long': train_ds.skipped_too_long, 'valid_rows_loaded': len(valid_rows), 'valid_rows_used': len(valid_ds), 'steps': total_steps, 'elapsed_sec': time.time() - start, 'train_loss_final_recent': sum(losses[-20:]) / min(len(losses), 20), 'events': events}
    write_json(args.output_dir / 'train_metrics.json', train_metrics)
    print(json.dumps({'train_metrics': train_metrics}, ensure_ascii=False, sort_keys=True))
if __name__ == '__main__':
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    main()
