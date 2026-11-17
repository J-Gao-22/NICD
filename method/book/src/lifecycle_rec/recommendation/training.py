from __future__ import annotations
import argparse
import json
import random
import time
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from lifecycle_rec.recommendation.model import RecommendationModel, collate, write_json

def set_trainable(model: RecommendationModel, stage: int) -> list[torch.nn.Parameter]:
    for param in model.parameters():
        param.requires_grad = False
    if stage == 1:
        for name, param in model.llm.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
    elif stage == 2:
        for param in model.proj.parameters():
            param.requires_grad = True
    elif stage == 3:
        for name, param in model.llm.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
    else:
        raise ValueError(stage)
    return [param for param in model.parameters() if param.requires_grad]

def train_stage(stage: int, model: RecommendationModel, plain_records: list[dict[str, Any]], chain_records: list[dict[str, Any]], tokenizer, yes_id: int, output_dir: Path, batch_size: int, seed: int, lr: float, sampling_mode: str, length_bucketing: bool, max_train_steps: int | None=None, log_every: int=100, pad_avoid_64: bool=False, checkpoint_every: int=0, resume_checkpoint: Path | None=None) -> dict[str, Any]:
    params = set_trainable(model, stage)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.001)
    use_mf_prefix = stage != 1
    rng = random.Random(seed + stage)
    chain_order = list(range(len(chain_records)))
    plain_order = list(range(len(plain_records)))
    rng.shuffle(chain_order)
    rng.shuffle(plain_order)
    schedule: list[tuple[str, int]] = []
    if sampling_mode == 'plain100':
        schedule = [('plain', idx) for idx in plain_order]
    elif sampling_mode == 'chain100':
        schedule = [('chain', idx) for idx in chain_order]
    elif sampling_mode == 'chain80_plain20':
        plain_pos = 0
        for chain_pos, chain_idx in enumerate(chain_order, start=1):
            schedule.append(('chain', chain_idx))
            if chain_pos % 4 == 0:
                if plain_pos >= len(plain_order):
                    plain_pos = 0
                    rng.shuffle(plain_order)
                schedule.append(('plain', plain_order[plain_pos]))
                plain_pos += 1
    else:
        raise ValueError(f'unknown sampling_mode={sampling_mode}')
    if length_bucketing:
        schedule.sort(key=lambda value: (chain_records[value[1]] if value[0] == 'chain' else plain_records[value[1]])['_length'])
        buckets = [schedule[start:start + batch_size] for start in range(0, len(schedule), batch_size)]
        rng.shuffle(buckets)
        schedule = [value for bucket in buckets for value in bucket]
    if stage == 2:
        model.eval()
        model.proj.train()
    else:
        model.train()
    loss_sum = 0.0
    steps_done = 0
    plain_seen = 0
    chain_seen = 0
    start_step = 0
    checkpoint_path = output_dir / f'stage{stage}_resume.pt'
    if resume_checkpoint is not None:
        saved = torch.load(resume_checkpoint, map_location='cpu', weights_only=False)
        expected = (stage, len(schedule), batch_size, len(chain_records), len(plain_records))
        actual = tuple((saved[key] for key in ('stage', 'schedule_length', 'batch_size', 'chain_count', 'plain_count')))
        if actual != expected:
            raise ValueError(f'checkpoint training geometry mismatch: {actual} != {expected}')
        for name, param in model.named_parameters():
            if name in saved['trainable_parameters']:
                param.data.copy_(saved['trainable_parameters'][name].to(param.device))
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        start_step = int(saved['step'])
        loss_sum = float(saved['loss_sum'])
        steps_done = int(saved['steps_done'])
        plain_seen = int(saved['plain_seen'])
        chain_seen = int(saved['chain_seen'])
        print(json.dumps({'event': 'resumed', 'stage': stage, 'step': start_step, 'checkpoint': str(resume_checkpoint)}), flush=True)
    started = time.time()
    for step, offset in enumerate(range(0, len(schedule), batch_size), start=1):
        if step <= start_step:
            continue
        if max_train_steps is not None and step > max_train_steps:
            break
        chosen = [chain_records[idx] if kind == 'chain' else plain_records[idx] for kind, idx in schedule[offset:offset + batch_size]]
        batch = collate(chosen, tokenizer.pad_token_id)
        if pad_avoid_64 and batch['input_ids'].shape[1] % 64 == 0:
            pad_ids = torch.full((len(chosen), 1), tokenizer.pad_token_id, dtype=batch['input_ids'].dtype, device=batch['input_ids'].device)
            pad_mask = torch.zeros((len(chosen), 1), dtype=batch['attention_mask'].dtype, device=batch['attention_mask'].device)
            batch['input_ids'] = torch.cat([batch['input_ids'], pad_ids], dim=1)
            batch['attention_mask'] = torch.cat([batch['attention_mask'], pad_mask], dim=1)
        optimizer.zero_grad(set_to_none=True)
        score = model(batch['input_ids'], batch['attention_mask'], batch['lengths'], batch['uid'], batch['iid'], yes_id, use_mf_prefix=use_mf_prefix)
        loss = F.binary_cross_entropy_with_logits(score.float(), batch['label'])
        if not torch.isfinite(score).all() or not torch.isfinite(loss):
            raise RuntimeError(json.dumps({'stage': stage, 'step': step, 'failure': 'nonfinite_forward', 'record_ids': [r['record_id'] for r in chosen], 'lengths': batch['lengths'].tolist(), 'scores': score.float().detach().cpu().tolist(), 'labels': batch['label'].tolist()}))
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        optimizer.step()
        if not all((torch.isfinite(p).all() for p in params)):
            raise RuntimeError(json.dumps({'stage': stage, 'step': step, 'failure': 'nonfinite_parameter', 'record_ids': [r['record_id'] for r in chosen], 'grad_norm': float(grad_norm)}))
        step_loss = float(loss.detach().cpu())
        loss_sum += step_loss
        steps_done += 1
        cur_kinds = [kind for kind, _ in schedule[offset:offset + batch_size]]
        chain_seen += sum((kind == 'chain' for kind in cur_kinds))
        plain_seen += sum((kind == 'plain' for kind in cur_kinds))
        if step == 1 or step % log_every == 0:
            print(json.dumps({'stage': stage, 'step': step, 'steps_total': (len(schedule) + batch_size - 1) // batch_size, 'loss': step_loss, 'grad_norm': float(grad_norm), 'plain_seen': plain_seen, 'chain_seen': chain_seen, 'elapsed_sec': round(time.time() - started, 2)}, ensure_ascii=False), flush=True)
        if checkpoint_every and step % checkpoint_every == 0:
            snapshot = {'stage': stage, 'step': step, 'schedule_length': len(schedule), 'batch_size': batch_size, 'chain_count': len(chain_records), 'plain_count': len(plain_records), 'trainable_parameters': {name: param.detach().cpu().clone() for name, param in model.named_parameters() if param.requires_grad}, 'optimizer': optimizer.state_dict(), 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(), 'loss_sum': loss_sum, 'steps_done': steps_done, 'plain_seen': plain_seen, 'chain_seen': chain_seen}
            tmp = checkpoint_path.with_suffix('.pt.tmp')
            torch.save(snapshot, tmp)
            tmp.replace(checkpoint_path)
    stage_dir = output_dir / f'stage{stage}'
    stage_dir.mkdir(parents=True, exist_ok=True)
    model.llm.save_pretrained(str(stage_dir / 'lora'))
    torch.save(model.proj.state_dict(), stage_dir / 'projection.pt')
    result = {'stage': stage, 'steps': steps_done, 'train_loss': loss_sum / steps_done, 'plain_seen': plain_seen, 'chain_seen': chain_seen, 'actual_chain_fraction': chain_seen / (plain_seen + chain_seen), 'elapsed_sec': time.time() - started, 'use_mf_prefix': use_mf_prefix, 'lr': lr, 'sampling_mode': sampling_mode, 'length_bucketing': length_bucketing}
    write_json(stage_dir / 'train_report.json', result)
    return result
