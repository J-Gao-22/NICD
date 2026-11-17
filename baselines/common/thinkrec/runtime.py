import argparse
import gc
import importlib
import inspect
import json
import math
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace

from ..llm.runtime import load_adapter_state, seed_all, to_device, write_json


def pickle_compatibility():
    import numpy as np
    if int(np.__version__.split('.')[0]) < 2:
        for suffix in ['', '.numeric', '.multiarray', '._multiarray_umath']:
            sys.modules.setdefault('numpy._core' + suffix, importlib.import_module('numpy.core' + suffix))


def configuration(dataset, stage, continuation=False):
    if dataset not in ('ml1m', 'book') or stage not in (1, 2, 3, 4):
        raise ValueError((dataset, stage))
    c = dict(dataset=dataset, stage=stage, history=30 if dataset == 'ml1m' else 10,
             seed=20260709, context=2048, use_ids=stage in (2, 4), use_desc=True,
             microbatch=2, batch=4, lr=1e-5 if stage == 3 else 1e-4,
             epochs=0, examples={1: 42680, 2: 41080, 3: 20540, 4: 0}[stage],
             native=continuation, prompt='adaptive' if stage == 4 else ('collaborative' if stage == 2 else 'text'))
    if dataset == 'book':
        c.update(examples={1: 804135, 2: 240000, 3: 16000, 4: 0}[stage],
                 microbatch=4 if stage in (1, 2) else 2,
                 batch=8 if stage == 1 else 4, lr={1: 1e-4, 2: 2e-5, 3: 1e-5, 4: 0.}[stage])
        if continuation:
            if stage not in (2, 3):
                raise ValueError(stage)
            c.update(seed=42, microbatch=1, batch=1, epochs=4 if stage == 2 else 2,
                     examples=3200 if stage == 2 else 1600, lr=1e-4 if stage == 2 else 1e-5)
    elif continuation:
        raise ValueError('Continuation is only configured for Book')
    return c


def prepare_groups(data, mf, output):
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
    pickle_compatibility()
    users = []
    for split in ['train', 'valid', 'test']:
        users.extend(pd.read_pickle(data / (split + '_ood2.pkl')).uid.tolist())
    users = list(set(users))
    state = torch.load(mf, map_location='cpu', weights_only=True)
    embeddings = state['user_embedding.weight'][torch.tensor(users)].float().numpy()
    scaled = StandardScaler().fit_transform(embeddings)
    keyword = 'metric' if 'metric' in inspect.signature(AgglomerativeClustering).parameters else 'affinity'
    labels = AgglomerativeClustering(n_clusters=2, linkage='ward', **{keyword: 'euclidean'}).fit_predict(scaled)
    if not np.isfinite(scaled).all():
        raise ValueError('Nonfinite embeddings')
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(user_id=users, cluster=labels)).to_csv(output, index=False)
    return output


def read_groups(path):
    import pandas as pd
    frame = pd.read_csv(path)
    if list(frame.columns) != ['user_id', 'cluster'] or not frame.user_id.is_unique:
        raise ValueError('Expected unique user_id,cluster routing rows')
    if set(frame.cluster) != {0, 1}:
        raise ValueError('Expected two nonempty groups')
    return frame


def training_datasets(args, config, group=None):
    import numpy as np
    from torch.utils.data import Subset
    from ..llm.data import AmazonOOData
    pickle_compatibility()
    datasets = {}
    users = set(read_groups(args.routing).query('cluster == @group').user_id) if group is not None else None
    for split in ['train', 'reason']:
        dataset = AmazonOOData(ann_paths=[(args.data / split).as_posix()], seq_len=config['history'],
                              user2group='', use_ids=config['use_ids'], use_desc=True)
        if users is not None:
            indices = np.flatnonzero(dataset.annotation.UserID.isin(users).to_numpy()).tolist()
            dataset = Subset(dataset, indices)
        if not len(dataset):
            raise ValueError('Empty training dataset: ' + split)
        datasets[split] = dataset
    return datasets


def save_adapter(model, path):
    import torch
    from peft import get_peft_model_state_dict
    path.mkdir(parents=True, exist_ok=True)
    name = model.llama_model_lora.active_adapter
    state = get_peft_model_state_dict(model.llama_model_lora, adapter_name=name)
    torch.save({k: v.detach().cpu().clone() for k, v in state.items()}, path / 'adapter_model.bin')
    model.llama_model_lora.peft_config[name].save_pretrained(path)


def save_projector(model, path):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu().clone() for k, v in model.llama_proj.state_dict().items()}, path)


def build_model(args, config, adapters=None, projector=None):
    import torch
    from peft import get_peft_model_state_dict
    from .model import MiniGPT4Rec_v3
    mf = torch.load(args.mf, map_location='cpu', weights_only=True)
    users, dim = mf['user_embedding.weight'].shape
    items, item_dim = mf['item_embedding.weight'].shape
    if dim != 256 or item_dim != 256:
        raise ValueError('Expected MF embedding size 256')
    stage = config['stage']
    model = MiniGPT4Rec_v3(
        rec_model='MF', rec_config=SimpleNamespace(user_num=users, item_num=items, embedding_size=dim),
        pretrained_rec=str(args.mf), freeze_rec=True, llama_model=str(args.backbone),
        freeze_proj=stage != 2, freeze_lora=stage != 1, max_txt_len=2048,
        proj_token_num=1, proj_drop=0, proj_mid=10, prompt_template='{}', end_sym='###',
        prompt_path=str(Path(__file__).parent / 'prompts' / config['dataset'] / (config['prompt'] + '.txt')),
        lora_config=SimpleNamespace(use_lora=True, r=8, alpha=16, dropout=.05, target_modules=['q_proj', 'v_proj']),
        loss_config=SimpleNamespace(alpha=.1, beta=.9, theta=.9, gamma=.1))
    if adapters:
        for i, path in enumerate(adapters):
            name = 'group' + str(i)
            model.llama_model_lora.load_adapter(str(path), adapter_name=name, is_trainable=stage == 3)
            saved = load_adapter_state(path)
            loaded = get_peft_model_state_dict(model.llama_model_lora, adapter_name=name)
            if saved.keys() != loaded.keys() or not all(torch.equal(saved[k], loaded[k].detach().cpu()) for k in saved):
                raise RuntimeError('Adapter load mismatch')
        model.llama_model_lora.set_adapter('group0')
        model.llama_model_lora.delete_adapter('default')
    if stage in (2, 3, 4):
        model.llama_model_lora.requires_grad_(False)
    if stage == 3:
        depth = model.llama_model.config.num_hidden_layers
        for name, parameter in model.llama_model_lora.named_parameters():
            match = re.search(r'layers\.(\d+)\.', name)
            if 'lora_' in name and match and int(match.group(1)) >= depth - 8:
                parameter.requires_grad = True
    if projector is not None:
        model.llama_proj.load_state_dict(torch.load(projector, map_location='cpu', weights_only=True), strict=True)
    model.rec_encoder.load_state_dict(mf, strict=True)
    model.set_mode('v3')
    model.set_answer_type('v2')
    model.llama_model.config.use_cache = False
    if model.llama_model.config.model_type != 'qwen2' or model.llama_model.config.num_hidden_layers != 28:
        raise ValueError('Qwen2.5-7B-Instruct is required')
    if [model.llama_tokenizer.encode(x, add_special_tokens=False) for x in ['Yes', 'No']] != [[9454], [2753]]:
        raise ValueError('Unexpected answer token IDs')
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    model = model.to(args.device)
    if stage == 4:
        if len(adapters or []) != 3 or projector is None:
            raise ValueError('Final evaluation requires two group adapters, a general adapter and projector')
        model.requires_grad_(False)
        model.set_user2group(str(args.routing))
        model.eval()
    else:
        trainable = [name for name, p in model.named_parameters() if p.requires_grad]
        if not trainable or not all(name.startswith('llama_proj.') if stage == 2 else 'lora_' in name for name in trainable):
            raise RuntimeError('Unexpected trainable parameters')
    return model


def fixed_training(model, datasets, config, device):
    import torch
    from ..llm.stream import SampleStream
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config['lr'], weight_decay=.001)
    streams = {kind: SampleStream(datasets[kind], config['seed'] + i) for i, kind in enumerate(['train', 'reason'])}
    mixer = random.Random(config['seed'] + 1)
    if config['stage'] == 2:
        model.eval()
        model.llama_proj.train()
    else:
        model.train()
    seen, step = 0, 0
    while seen < config['examples']:
        lr = 1e-5 if config['dataset'] == 'ml1m' and config['stage'] == 1 and step >= 10270 else config['lr']
        for g in optimizer.param_groups:
            g['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        update = min(config['batch'], config['examples'] - seen)
        remaining, loss_value = update, 0.
        while remaining:
            count = min(config['microbatch'], remaining)
            kind = 'reason' if mixer.random() < .2 else 'train'
            batch = to_device(streams[kind].next(count), device)
            loss = model(batch)['loss']
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            (loss * (count / update)).backward()
            loss_value += float(loss.detach()) * count / update
            remaining -= count
        norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in params):
            raise RuntimeError('Nonfinite parameters')
        seen += update
        step += 1
        if step == 1 or step % 25 == 0 or seen == config['examples']:
            print(json.dumps(dict(stage=config['stage'], step=step, examples=seen, loss=loss_value, lr=lr, grad_norm=float(norm))), flush=True)


def native_training(model, datasets, config, device):
    import torch
    from torch.utils.data import DataLoader
    from ..llm.schedule import LinearWarmupCosineLRScheduler
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (no_decay if parameter.ndim < 2 or 'bias' in name or 'ln' in name or 'bn' in name else decay).append(parameter)
    optimizer = torch.optim.AdamW([dict(params=decay, weight_decay=.001), dict(params=no_decay, weight_decay=0.)], lr=config['lr'])
    scheduler = LinearWarmupCosineLRScheduler(optimizer, max_epoch=config['epochs'], iters_per_epoch=800,
                                              min_lr=8e-5, init_lr=config['lr'], warmup_steps=200,
                                              warmup_start_lr=1e-5 if config['stage'] == 2 else 1e-6)
    seed_all(config['seed'])
    loaders = {kind: DataLoader(datasets[kind], batch_size=1, shuffle=True, drop_last=True,
                                num_workers=2, pin_memory=True) for kind in ['reason', 'train']}
    batches = {kind: iter(loader) for kind, loader in loaders.items()}
    scaler = torch.amp.GradScaler('cuda')
    model.train()
    optimizer.zero_grad()
    for epoch in range(config['epochs']):
        for step in range(800):
            kind = random.choices(['reason', 'train'], weights=[.2, .8], k=1)[0]
            try:
                batch = next(batches[kind])
            except StopIteration:
                batches[kind] = iter(loaders[kind])
                batch = next(batches[kind])
            batch = to_device(batch, device)
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
                print(json.dumps(dict(stage=config['stage'], step=epoch * 800 + step + 1,
                                      loss=float(loss.detach()), lr=optimizer.param_groups[0]['lr'])), flush=True)


def train_phase(args, config, output, adapters=None, projector=None, group=None):
    import torch
    seed_all(config['seed'])
    model = build_model(args, config, adapters, projector)
    try:
        datasets = training_datasets(args, config, group)
        if config['native']:
            native_training(model, datasets, config, args.device)
        else:
            fixed_training(model, datasets, config, args.device)
        if config['stage'] == 2:
            save_projector(model, output)
        else:
            save_adapter(model, output)
        return output
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def evaluate_final(model, args, config, split, output):
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import default_collate
    from ..llm.data import AmazonOOData
    from ..metrics import roc_auc_score, reorganize_by_user, uAUC_me, ndcg_at_k, map_at_k
    pickle_compatibility()
    frame = pd.read_pickle(args.data / (split + '_ood2.pkl')).reset_index(drop=True)
    dataset = AmazonOOData(ann_paths=[(args.data / split).as_posix()], seq_len=config['history'],
                          user2group='', use_ids=True, use_desc=True)
    scores = np.full(len(frame), np.nan, dtype=np.float64)
    route = read_groups(args.routing)
    if not set(frame.uid).issubset(set(route.user_id)):
        raise ValueError('Routing does not cover all evaluation users')
    model.eval()
    with torch.no_grad():
        grouped = frame.groupby('uid', sort=True).indices
        for user_index, (uid, positions) in enumerate(grouped.items()):
            for start in range(0, len(positions), 4):
                indices = positions[start:start + 4].tolist()
                batch = to_device(default_collate([dataset[i] for i in indices]), args.device)
                values = model.generate_for_samples_v3(batch)['logits'].float().cpu().numpy()
                if not np.isfinite(values).all():
                    raise RuntimeError('Nonfinite scores')
                actual = list(zip(batch['UserID'].cpu().tolist(), batch['TargetItemID'].cpu().tolist(), batch['label'].cpu().tolist()))
                if actual != list(frame.iloc[indices][['uid', 'iid', 'label']].itertuples(index=False, name=None)):
                    raise RuntimeError('Evaluation row identity mismatch')
                scores[indices] = values
            if (user_index + 1) % 500 == 0 or user_index + 1 == len(grouped):
                print(json.dumps(dict(split=split, processed=int(np.isfinite(scores).sum()), total=len(frame))), flush=True)
    if not np.isfinite(scores).all():
        raise RuntimeError('Incomplete evaluation')
    grouped = reorganize_by_user(frame.uid.to_numpy(), scores, frame.label.to_numpy())
    metrics = dict(auc=float(roc_auc_score(frame.label, scores)), uauc=float(uAUC_me(grouped)[0]),
                   ndcg5=float(ndcg_at_k(grouped, 5)), map5=float(map_at_k(grouped, 5)))
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / 'scores.npy', scores)
    write_json(output / 'metrics.json', dict(rows=len(frame), metrics=metrics))
    print(json.dumps(dict(split=split, rows=len(frame), metrics=metrics)), flush=True)
    return metrics


def run_final(args, dataset, general, projector, groups):
    import torch
    config = configuration(dataset, 4)
    seed_all(config['seed'])
    model = build_model(args, config, list(groups) + [general], projector)
    try:
        for split in ([args.split] if getattr(args, 'split', None) else ['valid', 'test']):
            evaluate_final(model, args, config, split, args.output / ('final_' + split))
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def train_pipeline(args, dataset):
    args.output.mkdir(parents=True, exist_ok=True)
    if args.routing is None:
        args.routing = prepare_groups(args.data, args.mf, args.output / 'routing.csv')
    else:
        read_groups(args.routing)
    general = train_phase(args, configuration(dataset, 1), args.output / 'stage1/adapter')
    stage2_base = args.output / ('stage2/base_projector.pt' if dataset == 'book' else 'stage2/projector.pt')
    projector = train_phase(args, configuration(dataset, 2), stage2_base, adapters=[general])
    if dataset == 'book':
        projector = train_phase(args, configuration(dataset, 2, True), args.output / 'stage2/projector.pt',
                                adapters=[general], projector=projector)
    groups = []
    for group in [0, 1]:
        base = args.output / 'stage3' / ('group_' + str(group))
        adapter = train_phase(args, configuration(dataset, 3), base / ('base_adapter' if dataset == 'book' else 'adapter'),
                              adapters=[general], projector=projector, group=group)
        if dataset == 'book':
            adapter = train_phase(args, configuration(dataset, 3, True), base / 'adapter',
                                  adapters=[adapter], projector=projector, group=group)
        groups.append(adapter)
    run_final(args, dataset, general, projector, groups)


def main(dataset, operation):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--mf', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    if operation == 'prepare_groups':
        args = parser.parse_args()
        if args.output.exists():
            parser.error('Output file already exists')
        prepare_groups(args.data, args.mf, args.output)
        return
    parser.add_argument('--backbone', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--routing', type=Path, required=operation == 'evaluate')
    if operation == 'evaluate':
        parser.add_argument('--split', choices=['valid', 'test'])
        parser.add_argument('--general', type=Path, required=True)
        parser.add_argument('--projector', type=Path, required=True)
        parser.add_argument('--groups', type=Path, nargs=2, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory must be empty')
    if operation == 'train':
        for split in ['train', 'reason']:
            if not (args.data / (split + '_ood2.pkl')).is_file():
                parser.error('Missing training input: ' + split)
    import torch
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        parser.error('A CUDA device is required')
    torch.cuda.set_device(torch.device(args.device))
    torch.backends.cuda.enable_cudnn_sdp(False)
    pickle_compatibility()
    if operation == 'train':
        train_pipeline(args, dataset)
    else:
        run_final(args, dataset, args.general, args.projector, args.groups)
