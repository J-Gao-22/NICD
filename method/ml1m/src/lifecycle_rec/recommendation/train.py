import argparse
import gc
import random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from peft import get_peft_model_state_dict
from safetensors.torch import save_file
from transformers import AutoTokenizer
from lifecycle_rec.recommendation.continuation import build_model, make_schedule
from lifecycle_rec.recommendation.model import RecommendationModel, collate
from lifecycle_rec.recommendation.training import train_stage

def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--base-model', required=True)
    p.add_argument('--mf-checkpoint', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = torch.load(args.cache, map_location='cpu', weights_only=False)
    datasets = data['datasets']
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    assert tokenizer.encode('Yes', add_special_tokens=False) == [9454]
    seed(20260709)
    model = RecommendationModel(args.base_model, args.mf_checkpoint, 8, 16, 0.05, ['q_proj', 'v_proj']).to('cuda')
    train_stage(1, model, datasets['train_plain'], datasets['train_chain'], tokenizer, 9454, args.output_dir, 4, 20260709, 0.0001, 'chain80_plain20', False, max_train_steps=10270)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    seed(20260709)
    config = SimpleNamespace(base_model=args.base_model, mf_checkpoint=args.mf_checkpoint, source_model_dir=args.output_dir)
    model = build_model(config, 'recommendation')
    params = [x for x in model.parameters() if x.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-05, weight_decay=0.001)
    plan = make_schedule(len(datasets['train_chain']), len(datasets['train_plain']), 400, 4, random.Random(20260709))
    assert plan == data['schedule'][41080:]
    model.train()
    for step in range(400):
        records = [datasets['train_' + kind][index] for kind, index in plan[step * 4:step * 4 + 4]]
        batch = collate(records, tokenizer.pad_token_id)
        optimizer.zero_grad(set_to_none=True)
        scores = model(batch['input_ids'], batch['attention_mask'], batch['lengths'], batch['uid'], batch['iid'], 9454, use_mf_prefix=False)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(scores.float(), batch['label'])
        if not torch.isfinite(loss):
            raise FloatingPointError(step)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 25 == 0:
            print(dict(step=step + 1, loss=float(loss.detach())), flush=True)
    output = args.output_dir / 'adapter'
    output.mkdir(exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in get_peft_model_state_dict(model.llm, adapter_name='recommendation').items()}, str(output / 'adapter_model.safetensors'))
    model.llm.peft_config['recommendation'].save_pretrained(output)
    tokenizer.save_pretrained(output)
if __name__ == '__main__':
    main()
