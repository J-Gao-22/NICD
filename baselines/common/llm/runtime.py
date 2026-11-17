import argparse
import contextlib
import importlib
import json
import random
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch


def settings(dataset, method):
    if dataset not in ('ml1m', 'book') or method not in ('tallrec', 'collm'):
        raise ValueError((dataset, method))
    native = (dataset, method) == ('book', 'tallrec')
    return dict(dataset=dataset, method=method, history=30 if dataset == 'ml1m' else 10,
                seed=42 if native else 20260709, context=2048,
                steps=40000 if native else (10670 if method == 'tallrec' else (10270 if dataset == 'ml1m' else 10000)),
                batch=1 if native else 4, microbatch=1 if native else 2,
                eval_batch=2 if native else 4, native=native)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def to_device(batch, device):
    import torch
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [to_device(v, device) for v in batch]
    if isinstance(batch, tuple):
        return tuple(to_device(v, device) for v in batch)
    return batch


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_adapter_state(path):
    import torch
    path = Path(path)
    if (path / 'adapter_model.safetensors').exists():
        from safetensors.torch import load_file
        return load_file(str(path / 'adapter_model.safetensors'))
    return torch.load(path / 'adapter_model.bin', map_location='cpu', weights_only=True)


def build_model(args, config):
    import torch
    from peft import get_peft_model_state_dict
    from transformers import AutoModelForCausalLM
    if config['dataset'] == 'ml1m':
        from .model_ml1m import MiniGPT4Rec_v3
    else:
        from .model_book import MiniGPT4Rec_v3
    mf = torch.load(args.mf, map_location='cpu', weights_only=True)
    users, dim = mf['user_embedding.weight'].shape
    items, item_dim = mf['item_embedding.weight'].shape
    if dim != 256 or item_dim != dim:
        raise ValueError('MF embedding size must be 256')
    lora = SimpleNamespace(use_lora=True, r=8, alpha=16,
                           target_modules=['q_proj', 'v_proj'], dropout=.05)
    kwargs = dict(rec_model='MF', rec_config=SimpleNamespace(user_num=users, item_num=items, embedding_size=dim),
                  pretrained_rec=str(args.mf), freeze_rec=True, rec_precision='fp16',
                  llama_model=str(args.backbone), max_txt_len=2048, end_sym='###',
                  prompt_path=str(Path(__file__).parent / 'prompts' / config['dataset'] / (config['method'] + '.txt')),
                  prompt_template='{}', proj_token_num=1, proj_drop=0, proj_mid=10,
                  lora_config=lora, freeze_lora=config['method'] == 'collm',
                  freeze_proj=config['method'] == 'tallrec')
    if config['dataset'] == 'book' and config['method'] == 'collm':
        original = AutoModelForCausalLM.from_pretrained

        def loader(*positional, **keywords):
            keywords.update(torch_dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True)
            return original(*positional, **keywords)

        with patch.object(AutoModelForCausalLM, 'from_pretrained', side_effect=loader):
            model = MiniGPT4Rec_v3(**kwargs)

        def bfloat_autocast(self, dtype=None):
            return torch.amp.autocast('cuda', dtype=torch.bfloat16)

        model.maybe_autocast = MethodType(bfloat_autocast, model)
        model.has_print_prompt = True
        model.has_pri_decode = True
    else:
        model = MiniGPT4Rec_v3(**kwargs)
    if args.adapter is not None:
        model.llama_model_lora.load_adapter(str(args.adapter), adapter_name='group0', is_trainable=False)
        model.llama_model_lora.set_adapter('group0')
        model.llama_model_lora.delete_adapter('default')
        saved = load_adapter_state(args.adapter)
        loaded = get_peft_model_state_dict(model.llama_model_lora, adapter_name='group0')
        if saved.keys() != loaded.keys() or not all(torch.equal(saved[k], loaded[k].cpu()) for k in saved):
            raise RuntimeError('Adapter load mismatch')
    if config['method'] == 'collm':
        model.llama_model_lora.requires_grad_(False)
    model.rec_encoder.load_state_dict(mf, strict=True)
    if args.projector is not None:
        state = torch.load(args.projector, map_location='cpu', weights_only=True)
        if 'model' in state:
            state = state['model']
        if any(k.startswith('llama_proj.') for k in state):
            state = {k.removeprefix('llama_proj.'): v for k, v in state.items() if k.startswith('llama_proj.')}
        model.llama_proj.load_state_dict(state, strict=True)
    model.set_mode('v3')
    model.set_answer_type('v2')
    model.llama_model.config.use_cache = False
    if model.llama_model.config.model_type != 'qwen2' or model.llama_model.config.num_hidden_layers != 28:
        raise ValueError('Qwen2.5-7B-Instruct is required')
    if [model.llama_tokenizer.encode(s, add_special_tokens=False) for s in ['Yes', 'No']] != [[9454], [2753]]:
        raise ValueError('Unexpected Yes/No token mapping')
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    if args.operation == 'evaluate':
        model.requires_grad_(False)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    if args.operation == 'train':
        if not names or not all(n.startswith('llama_proj.') if config['method'] == 'collm' else 'lora_' in n for n in names):
            raise RuntimeError('Unexpected trainable parameters')
    return model.to(args.device)


def make_dataset(args, config, split):
    import numpy as np
    if int(np.__version__.split('.')[0]) < 2:
        for suffix in ['', '.numeric', '.multiarray', '._multiarray_umath']:
            sys.modules.setdefault('numpy._core' + suffix, importlib.import_module('numpy.core' + suffix))
    from .data import AmazonOOData
    return AmazonOOData(ann_paths=[(args.data / split).as_posix()], seq_len=config['history'],
                        user2group='', use_ids=False, use_desc=False)


def evaluate(model, args, config, split, output):
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader, default_collate
    from ..metrics import roc_auc_score, reorganize_by_user, uAUC_me, ndcg_at_k, map_at_k
    dataset = make_dataset(args, config, split)
    frame = pd.read_pickle(args.data / (split + '_ood2.pkl')).reset_index(drop=True)
    if config['native']:
        batch_size = config['eval_batch'] if split == 'valid_small' else 4
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    else:
        loader = (default_collate([dataset[i] for i in range(start, min(len(dataset), start + config['eval_batch']))])
                  for start in range(0, len(dataset), config['eval_batch']))
    scores = np.full(len(frame), np.nan, dtype=np.float16 if config['native'] else np.float64)
    model.eval()
    offset = 0
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, args.device)
            size = len(batch['label'])
            expected = frame.iloc[offset:offset + size]
            actual = list(zip(batch['UserID'].cpu().tolist(), batch['TargetItemID'].cpu().tolist(), batch['label'].cpu().tolist()))
            if actual != list(expected[['uid', 'iid', 'label']].itertuples(index=False, name=None)):
                raise RuntimeError('Evaluation row identity mismatch')
            values = model.generate_for_samples_v3(batch)['logits'].float().cpu().numpy()
            if not np.isfinite(values).all():
                raise RuntimeError('Nonfinite scores')
            scores[offset:offset + size] = values
            offset += size
            if offset % 4096 < size or offset == len(frame):
                print(json.dumps(dict(split=split, processed=offset, total=len(frame))), flush=True)
    if offset != len(frame) or not np.isfinite(scores).all():
        raise RuntimeError('Incomplete evaluation')
    candidates = reorganize_by_user(frame.uid.to_numpy(), scores, frame.label.to_numpy())
    metrics = dict(auc=float(roc_auc_score(frame.label, scores)), uauc=float(uAUC_me(candidates)[0]),
                   ndcg5=float(ndcg_at_k(candidates, 5)), map5=float(map_at_k(candidates, 5)))
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / 'scores.npy', scores)
    report = dict(rows=len(frame), metrics=metrics)
    write_json(output / 'metrics.json', report)
    print(json.dumps(dict(split=split, **report)), flush=True)
    return metrics


def save_weights(model, path, method):
    import torch
    path.mkdir(parents=True, exist_ok=True)
    if method == 'tallrec':
        model.llama_model_lora.save_pretrained(path / 'adapter')
    else:
        torch.save(model.llama_proj.state_dict(), path / 'projector.pt')


def reload_weights(model, path, method):
    import torch
    if method == 'tallrec':
        from peft import set_peft_model_state_dict
        name = model.llama_model_lora.active_adapter
        set_peft_model_state_dict(model.llama_model_lora, load_adapter_state(path / 'adapter'), adapter_name=name)
    else:
        model.llama_proj.load_state_dict(torch.load(path / 'projector.pt', map_location='cpu', weights_only=True), strict=True)


def optimizer_for(model, config):
    import torch
    params = [p for p in model.parameters() if p.requires_grad]
    if config['dataset'] == 'ml1m':
        return torch.optim.AdamW(params, lr=1e-4, weight_decay=.001)
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (no_decay if parameter.ndim < 2 or 'bias' in name or 'ln' in name or 'bn' in name else decay).append(parameter)
    return torch.optim.AdamW([dict(params=decay, weight_decay=.001), dict(params=no_decay, weight_decay=0.)], lr=1e-4)


def train_fixed(model, args, config):
    import torch
    from .stream import SampleStream
    from .schedule import learning_rate
    optimizer = optimizer_for(model, config)
    params = [p for p in model.parameters() if p.requires_grad]
    stream = SampleStream(make_dataset(args, config, 'train'), config['seed'])
    best = -float('inf')
    if config['dataset'] == 'ml1m':
        validation_steps = set(range(2000, config['steps'] + 1, 2000)) | {10270, config['steps']}
    else:
        validation_steps = {250, 500, 1000, 2000, 4000, 8000, 10000}
    for step in range(config['steps']):
        if config['method'] == 'collm':
            model.eval()
            model.llama_proj.train()
        else:
            model.train()
        if config['dataset'] == 'book':
            lr = learning_rate(step, config['steps'])
        else:
            lr = 1e-5 if config['method'] == 'tallrec' and step >= 10270 else 1e-4
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.
        for _ in range(2):
            batch = to_device(stream.next(2), args.device)
            with (torch.amp.autocast('cuda', dtype=torch.bfloat16) if config['dataset'] == 'book' else contextlib.nullcontext()):
                loss = model(batch)['loss']
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            (loss / 2).backward()
            loss_value += float(loss.detach()) / 2
        norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        if not any(p.grad is not None for p in params):
            raise RuntimeError('Missing gradients')
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in params):
            raise RuntimeError('Nonfinite parameters')
        if (step + 1) % 25 == 0 or step == 0:
            print(json.dumps(dict(step=step + 1, loss=loss_value, grad_norm=float(norm), lr=lr)), flush=True)
        if step + 1 in validation_steps:
            metrics = evaluate(model, args, config, 'valid', args.output / ('valid_' + str(step + 1)))
            objective = metrics['auc'] / 4 + metrics['uauc'] / 3
            if objective > best:
                best = objective
                save_weights(model, args.output / 'best', config['method'])
                write_json(args.output / 'best' / 'selection.json', dict(step=step + 1, objective=objective, metrics=metrics))
    save_weights(model, args.output / 'last', config['method'])


def train_native(model, args, config):
    import torch
    from torch.utils.data import DataLoader
    from .schedule import LinearWarmupCosineLRScheduler
    optimizer = optimizer_for(model, config)
    scheduler = LinearWarmupCosineLRScheduler(optimizer, max_epoch=50, iters_per_epoch=800,
                                              min_lr=8e-5, init_lr=1e-4, warmup_steps=200, warmup_start_lr=1e-5)
    loader = DataLoader(make_dataset(args, config, 'train'), batch_size=1, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)
    batches = iter(loader)
    scaler = torch.amp.GradScaler('cuda')
    best, best_auc, best_uauc, not_change = -float('inf'), -float('inf'), -float('inf'), 0
    optimizer.zero_grad()
    for epoch in range(50):
        model.train()
        for step in range(800):
            try:
                batch = next(batches)
            except StopIteration:
                batches = iter(loader)
                batch = next(batches)
            batch = to_device(batch, args.device)
            scheduler.step(epoch, step)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                loss = model(batch)['loss']
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            if (step + 1) % 25 == 0:
                print(json.dumps(dict(epoch=epoch, step=epoch * 800 + step + 1, loss=float(loss.detach()), lr=optimizer.param_groups[0]['lr'])), flush=True)
        metrics = evaluate(model, args, config, 'valid_small', args.output / ('valid_' + str(epoch)))
        objective = metrics['auc'] / 4 + metrics['uauc'] / 3
        if metrics['auc'] > best_auc:
            best_auc, not_change = metrics['auc'], 0
        if objective > best:
            best, not_change = objective, 0
            save_weights(model, args.output / 'best', 'tallrec')
            write_json(args.output / 'best' / 'selection.json', dict(epoch=epoch, step=(epoch + 1) * 800, objective=objective, metrics=metrics))
        if metrics['uauc'] > best_uauc:
            best_uauc, not_change = metrics['uauc'], 0
        not_change += 1
        if not_change > 15:
            break
    save_weights(model, args.output / 'last', 'tallrec')


def main(dataset, method, operation):
    config = settings(dataset, method)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--backbone', type=Path, required=True)
    parser.add_argument('--mf', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=method == 'collm' or operation == 'evaluate')
    parser.add_argument('--projector', type=Path, required=method == 'collm' and operation == 'evaluate')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    if operation == 'evaluate':
        parser.add_argument('--split', choices=['valid', 'valid_small', 'test'], default='test')
    args = parser.parse_args()
    args.operation = operation
    if operation == 'train' and (args.adapter is not None and method == 'tallrec' or args.projector is not None):
        parser.error('Training requires fresh trainable weights')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory must be empty')
    import torch
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        parser.error('A CUDA device is required')
    torch.cuda.set_device(torch.device(args.device))
    seed_all(config['seed'])
    if not config['native']:
        torch.backends.cuda.enable_cudnn_sdp(False)
    if dataset == 'book' and method == 'collm':
        torch.backends.cuda.matmul.allow_tf32 = True
    model = build_model(args, config)
    args.output.mkdir(parents=True, exist_ok=True)
    if operation == 'evaluate':
        evaluate(model, args, config, args.split, args.output)
        return
    if config['native']:
        train_native(model, args, config)
    else:
        train_fixed(model, args, config)
    reload_weights(model, args.output / 'best', method)
    evaluate(model, args, config, 'valid', args.output / 'final_valid')
    evaluate(model, args, config, 'test', args.output / 'final_test')
