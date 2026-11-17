from __future__ import annotations
import argparse
import gc
import json
import random
import time
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from lifecycle_rec.recommendation.model import RecommendationModel, collate, detect_token_policy, encode_records, metrics, read_jsonl, write_json, write_jsonl

def make_schedule(n_chain: int, n_plain: int, max_steps: int, batch_size: int, rng: random.Random) -> list[tuple[str, int]]:
    chain_order = list(range(n_chain))
    plain_order = list(range(n_plain))
    rng.shuffle(chain_order)
    rng.shuffle(plain_order)
    out: list[tuple[str, int]] = []
    cp = pp = 0
    while len(out) < max_steps * batch_size:
        for _ in range(4):
            if cp >= len(chain_order):
                cp = 0
                rng.shuffle(chain_order)
            out.append(('chain', chain_order[cp]))
            cp += 1
            if len(out) >= max_steps * batch_size:
                break
        if len(out) >= max_steps * batch_size:
            break
        if pp >= len(plain_order):
            pp = 0
            rng.shuffle(plain_order)
        out.append(('plain', plain_order[pp]))
        pp += 1
    return out[:max_steps * batch_size]

def build_model(args, adapter_name: str, device='cuda'):
    model = RecommendationModel(args.base_model, args.mf_checkpoint, 8, 16, 0.05, ['q_proj', 'v_proj']).to(device)
    model.llm.load_adapter(str(args.source_model_dir / 'stage1' / 'lora'), adapter_name=adapter_name, is_trainable=True)
    model.llm.set_adapter(adapter_name)
    proj = args.source_model_dir / 'stage1' / 'projection.pt'
    if proj.exists():
        model.proj.load_state_dict(torch.load(proj, map_location='cpu'))
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.llm.named_parameters():
        if 'lora_' in name and adapter_name in name:
            p.requires_grad = True
    return model
