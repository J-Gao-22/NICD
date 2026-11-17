from __future__ import annotations
import argparse
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
SCHEMA_VERSION = 'lifecycle'

def read_jsonl(path: Path, limit: int | None=None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open('r', encoding='utf-8') as fh:
        for line in fh:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='\n') as fh:
        json.dump(value, fh, ensure_ascii=False, indent=2, sort_keys=True)
    tmp.replace(path)

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='\n') as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n')
    tmp.replace(path)
    return len(rows)

def extract_first_json(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith('```'):
        stripped = re.sub('^```(?:json)?\\s*', '', stripped, flags=re.I)
        stripped = re.sub('\\s*```$', '', stripped)
    try:
        value = json.loads(stripped)
        return value if isinstance(value, dict) else None
    except Exception:
        pass
    start = stripped.find('{')
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for idx in range(start, len(stripped)):
        ch = stripped[idx]
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
                    value = json.loads(stripped[start:idx + 1])
                    return value if isinstance(value, dict) else None
                except Exception:
                    return None
    return None

def validate_interest_node(value: dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {'json_valid': False, 'schema_valid': False, 'errors': ['json_parse_failed']}
    errors: list[str] = []
    node = value.get('interest_state_node')
    if not isinstance(node, dict):
        errors.append('missing_interest_state_node')
        node = {}
    for key in ['state_type', 'state_label', 'state_summary', 'positive_analysis', 'negative_analysis', 'decision_cues']:
        if key not in node:
            errors.append('missing_' + key)
    if node.get('state_type') not in ['positive_preference_state', 'negative_rejection_state', 'mixed_feedback_state']:
        errors.append('bad_state_type')
    for sec, arr_key in [('positive_analysis', 'liked_patterns'), ('negative_analysis', 'rejected_or_risky_patterns')]:
        sec_val = node.get(sec)
        if not isinstance(sec_val, dict):
            errors.append('bad_' + sec)
            continue
        if not isinstance(sec_val.get('reasoning'), str) or not sec_val.get('reasoning'):
            errors.append('missing_' + sec + '_reasoning')
        arr = sec_val.get(arr_key)
        if not isinstance(arr, list):
            errors.append('bad_' + arr_key)
            continue
        for item in arr:
            if not isinstance(item, dict):
                errors.append('bad_pattern_item')
                continue
            if len(item.get('evidence_titles', []) or []) > 3:
                errors.append('too_many_evidence_titles')
            if len(item.get('support_terms', []) or []) > 6:
                errors.append('too_many_support_terms')
    cues = node.get('decision_cues')
    if not isinstance(cues, dict):
        errors.append('bad_decision_cues')
    else:
        if len(cues.get('positive_match_cues', []) or []) > 5:
            errors.append('too_many_positive_match_cues')
        if len(cues.get('negative_match_cues', []) or []) > 5:
            errors.append('too_many_negative_match_cues')
    forbidden = ['candidate_id', 'analysis_lens', 'candidate_status']
    raw = json.dumps(value, ensure_ascii=False)
    for key in forbidden:
        if f'"{key}"' in raw:
            errors.append('candidate_mechanism_field_' + key)
    return {'json_valid': True, 'schema_valid': not errors, 'errors': sorted(set(errors))}

def load_tokenizer(model_path: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    return tokenizer

def row_prompt_answer(tokenizer: Any, row: dict[str, Any]) -> tuple[str, str, str]:
    messages = row['messages']
    prompt_messages = messages[:-1]
    assistant = messages[-1]['content']
    prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    eos = tokenizer.eos_token or ''
    return (prompt, assistant + eos, assistant)

class MessagesSftDataset(Dataset):

    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_seq_len: int) -> None:
        self.items: list[dict[str, Any]] = []
        self.skipped_too_long = 0
        for row in rows:
            prompt_text, answer_text, _ = row_prompt_answer(tokenizer, row)
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
            full_ids = tokenizer(prompt_text + answer_text, add_special_tokens=False).input_ids
            if len(full_ids) > max_seq_len:
                self.skipped_too_long += 1
                continue
            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
            if len(labels) != len(full_ids):
                self.skipped_too_long += 1
                continue
            self.items.append({'input_ids': torch.tensor(full_ids, dtype=torch.long), 'labels': torch.tensor(labels, dtype=torch.long), 'weight': float(row.get('sample_weight', 1.0)), 'record_id': row.get('record_id'), 'split': row.get('split')})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.items[idx]

def collate_batch(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    max_len = max((len(x['input_ids']) for x in batch))
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    weights = torch.tensor([float(x['weight']) for x in batch], dtype=torch.float32)
    record_ids = [x['record_id'] for x in batch]
    for i, item in enumerate(batch):
        n = len(item['input_ids'])
        input_ids[i, :n] = item['input_ids']
        labels[i, :n] = item['labels']
        attention_mask[i, :n] = 1
    return {'input_ids': input_ids, 'labels': labels, 'attention_mask': attention_mask, 'weights': weights, 'record_ids': record_ids}

def weighted_assistant_loss(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    token_loss = loss_fct(shift_logits.float().view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    token_loss = token_loss.view(shift_labels.size())
    mask = shift_labels.ne(-100)
    denom = mask.sum(dim=1).clamp_min(1)
    per_sample_loss = (token_loss * mask).sum(dim=1) / denom
    weighted = per_sample_loss * weights.to(per_sample_loss.device)
    return (weighted.mean(), {'assistant_token_count_mean': float(denom.float().mean().detach().cpu()), 'sample_loss_mean': float(per_sample_loss.mean().detach().cpu()), 'weighted_sample_loss_mean': float(weighted.mean().detach().cpu())})

@dataclass
class TrainConfig:
    base_model: str
    output_dir: str
    corpus_dir: str
    mode: str
    max_seq_len: int = 4096
    lr: float = 0.0001
    epochs: int = 2
    batch_size: int = 1
    grad_accum: int = 8
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
    dtype: str = 'bf16'
    seed: int = 42
    max_steps: int | None = None
    eval_max_new_tokens: int = 768
    smoke_eval_limit: int = 5
    grad_clip: float = 1.0
    warmup_steps: int = 20
    eval_every: int = 20
    checkpoint_every: int = 100
    generation_valid_limit: int | None = None
    generation_test_limit: int | None = None
    max_skipped_nonfinite_windows: int = 3

def load_lora_model(cfg: TrainConfig) -> Any:
    dtype = torch.bfloat16 if cfg.dtype == 'bf16' else torch.float16
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=dtype, device_map='auto', trust_remote_code=True)
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    peft_config = LoraConfig(r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout, target_modules=list(cfg.target_modules), bias='none', task_type='CAUSAL_LM')
    return get_peft_model(model, peft_config)

def train(cfg: TrainConfig) -> dict[str, Any]:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(cfg.base_model)
    train_limit = 20 if cfg.mode == 'smoke' else None
    valid_limit = 10 if cfg.mode == 'smoke' else None
    train_rows = read_jsonl(Path(cfg.corpus_dir) / 'segment_interest_node_sft_train.jsonl', train_limit)
    valid_rows = read_jsonl(Path(cfg.corpus_dir) / 'segment_interest_node_sft_valid.jsonl', valid_limit)
    test_rows = read_jsonl(Path(cfg.corpus_dir) / 'segment_interest_node_sft_test.jsonl', cfg.smoke_eval_limit if cfg.mode == 'smoke' else None)
    if not train_rows:
        raise RuntimeError('No train rows found.')
    train_ds = MessagesSftDataset(train_rows, tokenizer, cfg.max_seq_len)
    valid_ds = MessagesSftDataset(valid_rows, tokenizer, cfg.max_seq_len)
    if len(train_ds) == 0:
        raise RuntimeError('No usable train rows after max_seq_len filtering.')
    collate = lambda batch: collate_batch(batch, tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate) if len(valid_ds) else None
    model = load_lora_model(cfg)
    model.train()
    model.print_trainable_parameters()
    optimizer = AdamW(model.parameters(), lr=cfg.lr)
    device = next(model.parameters()).device
    max_steps = 20 if cfg.mode == 'smoke' else cfg.max_steps
    steps_per_epoch = len(train_loader) // cfg.grad_accum
    planned_steps = cfg.epochs * steps_per_epoch
    if max_steps is not None:
        planned_steps = min(planned_steps, max_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=min(cfg.warmup_steps, max(0, planned_steps - 1)), num_training_steps=max(1, planned_steps))
    total_steps, running, events = (0, [], [])
    accumulation_record_ids: list[str] = []
    skipped_nonfinite_windows: list[dict[str, Any]] = []
    start = time.time()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(cfg.epochs):
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            weights = batch['weights'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss, stats = weighted_assistant_loss(outputs.logits, labels, weights)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite_loss epoch={epoch} batch={batch_idx} step={total_steps} records={batch['record_ids']}")
            (loss / cfg.grad_accum).backward()
            running.append(float(loss.detach().cpu()))
            accumulation_record_ids.extend(batch['record_ids'])
            if (batch_idx + 1) % cfg.grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                if not torch.isfinite(grad_norm):
                    skipped = {'epoch': epoch, 'batch_index': batch_idx, 'step_before_skip': total_steps, 'record_ids': list(accumulation_record_ids), 'reason': 'nonfinite_accumulated_gradient'}
                    skipped_nonfinite_windows.append(skipped)
                    print(json.dumps({'skipped_window': skipped}, ensure_ascii=False), flush=True)
                    optimizer.zero_grad(set_to_none=True)
                    accumulation_record_ids.clear()
                    if len(skipped_nonfinite_windows) > cfg.max_skipped_nonfinite_windows:
                        raise RuntimeError(f'too_many_nonfinite_gradient_windows count={len(skipped_nonfinite_windows)}')
                    continue
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accumulation_record_ids.clear()
                total_steps += 1
                if total_steps == 1 or total_steps % 5 == 0:
                    bad_params = [name for name, param in model.named_parameters() if param.requires_grad and (not torch.isfinite(param).all())]
                    if bad_params:
                        raise RuntimeError(f'nonfinite_parameters step={total_steps} names={bad_params[:8]}')
                    event = {'step': total_steps, 'epoch': epoch, 'lr': scheduler.get_last_lr()[0], 'grad_norm_before_clip': float(grad_norm.detach().cpu()), 'train_loss_recent': sum(running[-20:]) / min(len(running), 20), **stats}
                    events.append(event)
                    print(json.dumps(event, ensure_ascii=False), flush=True)
                if cfg.checkpoint_every > 0 and total_steps % cfg.checkpoint_every == 0:
                    latest = output_dir / 'checkpoint_latest'
                    model.save_pretrained(latest)
                    tokenizer.save_pretrained(latest)
                    write_json(latest / 'checkpoint_state.json', {'step': total_steps, 'epoch': epoch})
                if max_steps is not None and total_steps >= max_steps:
                    break
        if max_steps is not None and total_steps >= max_steps:
            break
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    train_metrics = {'config': asdict(cfg), 'train_rows_loaded': len(train_rows), 'train_rows_used': len(train_ds), 'train_rows_skipped_too_long': train_ds.skipped_too_long, 'valid_rows_loaded': len(valid_rows), 'valid_rows_used': len(valid_ds), 'valid_rows_skipped_too_long': valid_ds.skipped_too_long, 'steps': total_steps, 'train_loss_final_recent': sum(running[-20:]) / min(len(running), 20) if running else math.nan, 'elapsed_sec': time.time() - start, 'events': events, 'skipped_nonfinite_windows': skipped_nonfinite_windows, 'skipped_nonfinite_window_count': len(skipped_nonfinite_windows)}
    write_json(output_dir / 'train_metrics.json', train_metrics)
    valid_generation_limit = cfg.smoke_eval_limit if cfg.mode == 'smoke' else cfg.generation_valid_limit
    test_generation_limit = cfg.smoke_eval_limit if cfg.mode == 'smoke' else cfg.generation_test_limit
    if valid_generation_limit is None and cfg.mode != 'smoke':
        valid_generation_limit = min(30, len(valid_rows))
    if test_generation_limit is None and cfg.mode != 'smoke':
        test_generation_limit = len(test_rows)
    write_json(output_dir / 'trainer_state.json', {'mode': cfg.mode, 'completed_steps': total_steps, 'adapter_dir': str(output_dir)})
    return {'train_metrics': train_metrics}

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-model', default='Qwen/Qwen2.5-7B-Instruct')
    ap.add_argument('--corpus-dir', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--mode', choices=['smoke', 'train'], default='train')
    ap.add_argument('--max-seq-len', type=int, default=4096)
    ap.add_argument('--lr', type=float, default=0.0001)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--grad-accum', type=int, default=8)
    ap.add_argument('--lora-rank', type=int, default=16)
    ap.add_argument('--lora-alpha', type=int, default=32)
    ap.add_argument('--lora-dropout', type=float, default=0.05)
    ap.add_argument('--target-modules', default='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj')
    ap.add_argument('--dtype', choices=['bf16', 'fp16'], default='bf16')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-steps', type=int, default=None)
    ap.add_argument('--eval-max-new-tokens', type=int, default=768)
    ap.add_argument('--smoke-eval-limit', type=int, default=5)
    ap.add_argument('--grad-clip', type=float, default=1.0)
    ap.add_argument('--warmup-steps', type=int, default=20)
    ap.add_argument('--eval-every', type=int, default=20)
    ap.add_argument('--checkpoint-every', type=int, default=100)
    ap.add_argument('--generation-valid-limit', type=int, default=None)
    ap.add_argument('--generation-test-limit', type=int, default=None)
    ap.add_argument('--max-skipped-nonfinite-windows', type=int, default=3)
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    cfg = TrainConfig(base_model=args.base_model, corpus_dir=args.corpus_dir, output_dir=args.output_dir, mode=args.mode, max_seq_len=args.max_seq_len, lr=args.lr, epochs=args.epochs, batch_size=args.batch_size, grad_accum=args.grad_accum, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, target_modules=tuple((x.strip() for x in args.target_modules.split(',') if x.strip())), dtype=args.dtype, seed=args.seed, max_steps=args.max_steps, eval_max_new_tokens=args.eval_max_new_tokens, smoke_eval_limit=args.smoke_eval_limit, grad_clip=args.grad_clip, warmup_steps=args.warmup_steps, eval_every=args.eval_every, checkpoint_every=args.checkpoint_every, generation_valid_limit=args.generation_valid_limit, generation_test_limit=args.generation_test_limit, max_skipped_nonfinite_windows=args.max_skipped_nonfinite_windows)
    print(json.dumps(train(cfg), ensure_ascii=False, sort_keys=True))
if __name__ == '__main__':
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    main()
