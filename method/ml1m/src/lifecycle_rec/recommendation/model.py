from __future__ import annotations
import argparse
import gzip
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from sklearn.metrics import roc_auc_score
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

def read_jsonl(path: Path, limit: int | None=None) -> list[dict[str, Any]]:
    rows = []
    source = path
    if not source.exists() and path.with_suffix(path.suffix + '.gz').exists():
        source = path.with_suffix(path.suffix + '.gz')
    opener = gzip.open if source.suffix == '.gz' else open
    with opener(source, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')
    tmp.replace(path)

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n')
    tmp.replace(path)

def render_chat(tokenizer, system: str, user: str) -> str:
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def detect_token_policy(tokenizer) -> dict[str, Any]:
    probes = {'Yes': tokenizer.encode('Yes', add_special_tokens=False), ' Yes': tokenizer.encode(' Yes', add_special_tokens=False), 'No': tokenizer.encode('No', add_special_tokens=False), ' No': tokenizer.encode(' No', add_special_tokens=False)}
    return {'probes': probes, 'yes_token_id': int(probes['Yes'][0]), 'no_token_id': int(probes['No'][0])}

def metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    labels = np.array([r['label'] for r in rows], dtype=np.float32)
    scores = np.array([r['score'] for r in rows], dtype=np.float32)
    out = {'auc': float(roc_auc_score(labels, scores)) if len(set(labels.tolist())) > 1 else 0.0}
    per_user: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        per_user[int(r['uid'])].append(r)
    uauc, hr5, ndcg5, map5 = ([], [], [], [])
    for rs in per_user.values():
        y = np.array([r['label'] for r in rs], dtype=np.float32)
        s = np.array([r['score'] for r in rs], dtype=np.float32)
        if len(set(y.tolist())) > 1:
            uauc.append(float(roc_auc_score(y, s)))
        order = np.argsort(-s)
        ranked = y[order]
        k = min(5, len(ranked))
        gains = ranked[:k]
        hr5.append(float(np.any(gains == 1)))
        dcg = float(sum((float(rel) / math.log2(i + 2) for i, rel in enumerate(gains))))
        ideal = np.sort(y)[::-1][:k]
        idcg = float(sum((float(rel) / math.log2(i + 2) for i, rel in enumerate(ideal))))
        ndcg5.append(dcg / idcg if idcg > 0 else 0.0)
        hits = 0.0
        prec_sum = 0.0
        for i, rel in enumerate(gains, 1):
            if rel == 1:
                hits += 1.0
                prec_sum += hits / i
        denom = float(np.sum(y))
        map5.append(prec_sum / denom if denom > 0 else 0.0)
    out.update({'uauc': float(np.mean(uauc)) if uauc else 0.0, 'uauc_user_count': len(uauc), 'hr_at_5': float(np.mean(hr5)) if hr5 else 0.0, 'ndcg_at_5': float(np.mean(ndcg5)) if ndcg5 else 0.0, 'map_at_5': float(np.mean(map5)) if map5 else 0.0, 'rows_scored': len(rows)})
    return out

class FrozenMF(nn.Module):

    def __init__(self, checkpoint: Path):
        super().__init__()
        obj = torch.load(checkpoint, map_location='cpu')
        self.user_num = int(obj['user_num'])
        self.item_num = int(obj['item_num'])
        self.embedding_size = int(obj['embedding_size'])
        self.user_embedding = nn.Embedding(self.user_num, self.embedding_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.item_num, self.embedding_size, padding_idx=0)
        sd = obj['model_state_dict']
        self.user_embedding.weight.data.copy_(sd['user_embedding.weight'])
        self.item_embedding.weight.data.copy_(sd['item_embedding.weight'])
        for p in self.parameters():
            p.requires_grad = False

    def user_item(self, uid: torch.Tensor, iid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        uid = torch.clamp(uid, 0, self.user_num - 1)
        iid = torch.clamp(iid, 0, self.item_num - 1)
        return (self.user_embedding(uid), self.item_embedding(iid))

class RecommendationModel(nn.Module):

    def __init__(self, base_model: str, mf_checkpoint: Path, lora_rank: int, lora_alpha: int, lora_dropout: float, target_modules: list[str], proj_mid_times: int=10, gradient_checkpointing: bool=True, last_token_only: bool=False, attn_implementation: str | None=None):
        super().__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True, **{'attn_implementation': attn_implementation} if attn_implementation else {})
        self.llm.config.use_cache = False
        if gradient_checkpointing:
            self.llm.gradient_checkpointing_enable()
        cfg = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, target_modules=target_modules, bias='none', task_type='CAUSAL_LM')
        self.llm = get_peft_model(self.llm, cfg)
        self.last_token_only = last_token_only
        self.llm.print_trainable_parameters()
        self.mf = FrozenMF(mf_checkpoint)
        hidden = int(self.llm.config.hidden_size)
        emb = self.mf.embedding_size
        self.proj = nn.Sequential(nn.Linear(emb, emb * proj_mid_times), nn.ReLU(), nn.Linear(emb * proj_mid_times, hidden))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, lengths: torch.Tensor, uid: torch.Tensor, iid: torch.Tensor, yes_id: int, use_mf_prefix: bool=True) -> torch.Tensor:
        uid = uid.to(input_ids.device)
        iid = iid.to(input_ids.device)
        tok_emb = self.llm.get_input_embeddings()(input_ids)
        prefix_len = 0
        if use_mf_prefix:
            self.mf.to(input_ids.device)
            user_e, item_e = self.mf.user_item(uid, iid)
            llm_dtype = next(self.llm.parameters()).dtype
            user_p = self.proj(user_e.float()).to(dtype=llm_dtype)
            item_p = self.proj(item_e.float()).to(dtype=llm_dtype)
            prefix = torch.stack([user_p, item_p], dim=1)
            inputs_embeds = torch.cat([prefix, tok_emb], dim=1)
            prefix_mask = torch.ones((input_ids.shape[0], 2), dtype=attention_mask.dtype, device=attention_mask.device)
            attn = torch.cat([prefix_mask, attention_mask], dim=1)
            prefix_len = 2
        else:
            inputs_embeds = tok_emb
            attn = attention_mask
        batch_idx = torch.arange(input_ids.shape[0], device=input_ids.device)
        answer_pos = lengths + prefix_len - 1
        if self.last_token_only:
            causal_lm = self.llm.get_base_model()
            hidden = causal_lm.model(inputs_embeds=inputs_embeds, attention_mask=attn, use_cache=False, return_dict=True).last_hidden_state
            answer_hidden = hidden[batch_idx, answer_pos, :]
            logits = causal_lm.lm_head(answer_hidden)
        else:
            out = self.llm(inputs_embeds=inputs_embeds, attention_mask=attn, use_cache=False)
            logits = out.logits[batch_idx, answer_pos, :]
        return logits[:, yes_id]

def encode_records(tokenizer, records: list[dict[str, Any]], max_seq_len: int, tokenizer_batch_size: int=256) -> list[dict[str, Any]]:
    out = []
    started = time.time()
    for start in range(0, len(records), tokenizer_batch_size):
        batch = records[start:start + tokenizer_batch_size]
        texts = [render_chat(tokenizer, r['system'], r['user']) for r in batch]
        encoded = tokenizer(texts, add_special_tokens=False, truncation=False)
        for r, input_ids in zip(batch, encoded.input_ids):
            length = len(input_ids)
            if length <= max_seq_len:
                out.append({**r, '_ids': input_ids, '_length': length})
        if start == 0 or (start + len(batch)) % 32768 < tokenizer_batch_size:
            print(json.dumps({'event': 'tokenize_progress', 'seen': start + len(batch), 'total': len(records), 'kept': len(out), 'rows_per_sec': round((start + len(batch)) / max(time.time() - started, 1e-06), 2)}), flush=True)
    return out

def collate(batch: list[dict[str, Any]], pad_id: int, device='cuda') -> dict[str, torch.Tensor]:
    max_len = max((r['_length'] for r in batch))
    ids, masks = ([], [])
    for r in batch:
        cur = torch.tensor(r['_ids'], dtype=torch.long)
        pad = max_len - cur.numel()
        if pad:
            cur = torch.cat([cur, torch.full((pad,), pad_id, dtype=torch.long)])
        ids.append(cur)
        masks.append(torch.tensor([1] * r['_length'] + [0] * pad, dtype=torch.long))
    return {'input_ids': torch.stack(ids).to(device), 'attention_mask': torch.stack(masks).to(device), 'lengths': torch.tensor([r['_length'] for r in batch], dtype=torch.long, device=device), 'uid': torch.tensor([int(r['uid']) for r in batch], dtype=torch.long, device=device), 'iid': torch.tensor([int(r['iid']) for r in batch], dtype=torch.long, device=device), 'label': torch.tensor([float(r['label']) for r in batch], dtype=torch.float32, device=device), 'records': batch}
