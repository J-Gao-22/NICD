import argparse
import math
import random
from pathlib import Path
import numpy as np
import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoTokenizer
from lifecycle_rec.recommendation.metrics import map_at_k, ndcg_at_k, reorganize_by_user, roc_auc_score, uAUC_me
from lifecycle_rec.recommendation.model import RecommendationModel, collate, write_json
from lifecycle_rec.recommendation.slots import CollaborativeSlotModel, SlotLayout, collate_records, inject_record

def arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--base-model', required=True)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--native-mf', type=Path, required=True)
    p.add_argument('--collaborative-mf', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    return p

def build_model(args):
    native = RecommendationModel(args.base_model, args.native_mf, 8, 16, 0.05, ['q_proj', 'v_proj'], gradient_checkpointing=False, last_token_only=False, attn_implementation='sdpa')
    saved = load_file(str(args.adapter / 'adapter_model.safetensors'))
    set_peft_model_state_dict(native.llm, saved, adapter_name='default')
    loaded = get_peft_model_state_dict(native.llm)
    assert saved.keys() == loaded.keys()
    assert all((torch.equal(v, loaded[k].detach().cpu()) for k, v in saved.items()))
    mf = torch.load(args.collaborative_mf, map_location='cpu', weights_only=True)
    if mf['user_embedding.weight'].shape != (6041, 256) or mf['item_embedding.weight'].shape != (3687, 256):
        raise ValueError('MF shape mismatch')
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    assert tokenizer.encode('Yes', add_special_tokens=False) == [9454]
    layout = SlotLayout.from_tokenizer(tokenizer)
    torch.manual_seed(20260709)
    model = CollaborativeSlotModel(native, mf, mid_times=10, placeholder_id=layout.placeholder_id).to(args.device)
    model.trainable_names()
    return (model, tokenizer, layout)

def metrics(records, scores):
    labels = np.asarray([r['label'] for r in records])
    users = np.asarray([r['uid'] for r in records])
    scores = np.asarray(scores)
    groups = reorganize_by_user(users, scores, labels)
    return dict(auc=float(roc_auc_score(labels, scores)), uauc=float(uAUC_me(groups)[0]), ndcg5=float(ndcg_at_k(groups, 5)), map5=float(map_at_k(groups, 5)))

@torch.no_grad()
def score(model, records, pad_id, device, batch_size=4, collaborative=True):
    model.eval()
    result = []
    for start in range(0, len(records), batch_size):
        batch_records = records[start:start + batch_size]
        if collaborative:
            batch = collate_records(batch_records, pad_id, device)
            values = model(batch)
        else:
            batch = collate(batch_records, pad_id, device=device)
            values = model.native(batch['input_ids'], batch['attention_mask'], batch['lengths'], batch['uid'], batch['iid'], 9454, use_mf_prefix=False)
        values = values.float().cpu().numpy()
        if not np.isfinite(values).all():
            raise FloatingPointError(start)
        result.extend(values.tolist())
    return np.asarray(result)

def learning_rate(step):
    if step < 200:
        return 1e-05 + 9e-05 * step / 200
    return 1e-05 + 4.5e-05 * (1 + math.cos(math.pi * (step - 200) / 1799))

def objective(values):
    return values['auc'] / 4 + values['uauc'] / 3

def main():
    args = arguments().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    random.seed(20260709)
    np.random.seed(20260709)
    torch.manual_seed(20260709)
    torch.cuda.manual_seed_all(20260709)
    model, tokenizer, layout = build_model(args)
    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    raw = cache['datasets']
    datasets = {k: [inject_record(r, layout) for r in raw[k]] for k in ('train_chain', 'train_plain', 'valid')}
    plan = cache['schedule'][:8000]
    assert len(plan) == 8000 and sum((k == 'chain' for k, i in plan)) == 6400
    params = list(model.cf_projector.parameters())
    optimizer = torch.optim.AdamW([dict(params=[p for p in params if p.ndim >= 2], weight_decay=0.001), dict(params=[p for p in params if p.ndim < 2], weight_decay=0.0)], lr=0.0001)
    for step in range(2000):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group['lr'] = learning_rate(step)
        total_loss = 0.0
        for kind, index in plan[step * 4:step * 4 + 4]:
            batch = collate_records([datasets['train_' + kind][index]], layout.placeholder_id, args.device)
            with torch.amp.autocast(device_type=str(args.device).split(':')[0], dtype=torch.bfloat16):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(model(batch).float(), batch['label'])
            if not torch.isfinite(loss):
                raise FloatingPointError(step)
            (loss / 4).backward()
            total_loss += float(loss.detach()) / 4
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        if not any((p.grad is not None for p in params)):
            raise RuntimeError('Missing projector gradient')
        if not any((p.grad is not None and torch.count_nonzero(p.grad) for p in params)) and total_loss > 1e-06:
            raise RuntimeError('Zero projector gradient')
        optimizer.step()
        if not all((torch.isfinite(p).all() for p in params)):
            raise FloatingPointError(step)
        if step == 0 or (step + 1) % 25 == 0:
            print(dict(step=step + 1, loss=total_loss, lr=learning_rate(step), grad_norm=float(norm)), flush=True)
    torch.save(model.cf_projector.state_dict(), args.output_dir / 'projector.pt')
    write_json(args.output_dir / 'selection.json', dict(collaborative=True, checkpoint=dict(step=2000)))
if __name__ == '__main__':
    main()
