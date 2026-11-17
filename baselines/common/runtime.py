import argparse
import contextlib
import hashlib
import importlib
import io
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, TensorDataset
from common.config import CONFIG
from common.data import seq_dataset_eval
from common.metrics import map_at_k, ndcg_at_k, reorganize_by_user, roc_auc_score, uAUC_me
from common.models import GSASRec, LightGCN, MatrixFactorization, SASRec

for name, module in [('numpy._core', np.core), ('numpy._core.numeric', np.core.numeric), ('numpy._core.multiarray', np.core.multiarray)]:
    sys.modules.setdefault(name, module)

def frames(folder):
    return {s: pd.read_pickle(folder / (s + '_ood2.pkl')).reset_index(drop=True) for s in ['train', 'valid', 'test']}

def measure(frame, scores):
    with contextlib.redirect_stdout(io.StringIO()):
        grouped = reorganize_by_user(frame.uid.to_numpy(), scores, frame.label.to_numpy())
        return dict(auc=float(roc_auc_score(frame.label, scores)), uauc=float(uAUC_me(grouped)[0]), ndcg5=float(ndcg_at_k(grouped, 5)), map5=float(map_at_k(grouped, 5)))

def graph(frame, users, items):
    positive = frame.loc[frame.label > 0, ['uid', 'iid', 'label']].to_numpy().copy()
    positive[:, 1] += users
    reverse = positive.copy()
    reverse[:, 0] = positive[:, 1]
    reverse[:, 1] = positive[:, 0]
    pairs = np.concatenate([positive, reverse])
    adjacency = sp.csr_matrix((pairs[:, 2], (pairs[:, 0], pairs[:, 1])), shape=(users + items, users + items)).todok()
    degree = np.array(adjacency.sum(axis=1))
    inverse = np.zeros_like(degree, dtype=float)
    np.power(degree, -0.5, out=inverse, where=degree != 0)
    diagonal = sp.diags(inverse.flatten())
    normalized = diagonal.dot(adjacency).dot(diagonal).tocsr().tocoo().astype(np.float32)
    indices = torch.tensor(np.array([normalized.row, normalized.col]), dtype=torch.long)
    return torch.sparse_coo_tensor(indices, torch.tensor(normalized.data), normalized.shape).coalesce()

def create_model(method, config, data, device):
    users = int(max((f.uid.max() for f in data.values())) + 1)
    items = int(max((f.iid.max() for f in data.values())) + 1)
    dimension = config['embedding_size']
    if method == 'mf':
        model = MatrixFactorization(SimpleNamespace(user_num=users, item_num=items, embedding_size=dimension))
    elif method == 'lightgcn':
        model = LightGCN(SimpleNamespace(user_num=users, item_num=items, embed_size=dimension, gcn_layers=config['gcn_layer'], keep_prob=0.6, A_split=False, dropout=False, pretrain=0, init_emb=0.1))
        model._set_graph(graph(data['train'], users, items))
    else:
        options = SimpleNamespace(user_num=users, item_num=items, hidden_units=dimension, num_blocks=2, num_heads=1, dropout_rate=0.2, l2_emb=0.0001, maxlen=config['maxlen'])
        model = (SASRec if method == 'sasrec' else GSASRec)(options)
    return model.to(device)

def predict(model, method, config, frame, device):
    model.eval()
    if method in ['sasrec', 'gsasrec']:
        dataset = seq_dataset_eval(frame[['uid', 'iid', 'his', 'label']].values, max_len=config['maxlen'])
    else:
        dataset = TensorDataset(torch.tensor(frame.uid.to_numpy(), dtype=torch.long), torch.tensor(frame.iid.to_numpy(), dtype=torch.long))
    result = []
    with torch.inference_mode():
        if method == 'lightgcn':
            users, items = model.computer()
        for batch in DataLoader(dataset, batch_size=1024, shuffle=False):
            uid, iid = (batch[0].long().to(device), batch[1].long().to(device))
            if method in ['sasrec', 'gsasrec']:
                scores = model.forward_eval(uid, iid, batch[2].long().to(device))
            elif method == 'lightgcn':
                scores = (users[uid] * items[iid]).sum(-1)
            else:
                scores = model(uid, iid)
            result.append(scores.cpu().numpy().reshape(-1))
    scores = np.concatenate(result)
    assert np.isfinite(scores).all()
    return scores

def train_mf(data, output, device):
    random.seed(2025)
    np.random.seed(2025)
    torch.manual_seed(2025)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2025)
    model = create_model('mf', dict(embedding_size=256), {s: data[s] for s in ['train', 'valid']}, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    train = data['train']
    values = [torch.tensor(train.uid.to_numpy(), dtype=torch.long), torch.tensor(train.iid.to_numpy(), dtype=torch.long), torch.tensor(train.label.to_numpy(), dtype=torch.float32)]
    loader = DataLoader(TensorDataset(*values), batch_size=1024, shuffle=True)
    best_auc, stale = (-1.0, 0)
    for epoch in range(5000):
        model.train()
        loss_sum = 0.0
        for batch in loader:
            uid, iid, label = [x.to(device) for x in batch]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(uid, iid).reshape(-1), label)
            assert torch.isfinite(loss).item()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            parameters = [p for p in model.parameters() if p.grad is not None]
            norm = torch.nn.utils.clip_grad_norm_(parameters, float('inf'), error_if_nonfinite=True)
            assert norm.item() > 0
            optimizer.step()
            loss_sum += loss.item()
        scores = predict(model, 'mf', {}, data['valid'], device)
        auc = float(roc_auc_score(data['valid'].label, scores))
        if auc > best_auc:
            best_auc, stale = (auc, 0)
            torch.save(model.state_dict(), output / 'best.pt')
        else:
            stale += 1
        print(json.dumps(dict(epoch=epoch, loss=loss_sum / len(loader), valid_auc=auc, best_valid_auc=best_auc)), flush=True)
        if stale >= 50 or (epoch > 500 and best_auc < 0.52):
            break

def main(dataset, method, mode):
    item = CONFIG[dataset][method]
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--device', default='cpu' if mode == 'evaluate' else 'cuda')
    parser.add_argument('--threads', type=int, default=2 if mode == 'evaluate' else 4)
    if mode == 'evaluate':
        parser.add_argument('--checkpoint', type=Path, required=True)
        parser.add_argument('--split', choices=['valid', 'test'], default='test')
        parser.add_argument('--scores', type=Path)
    else:
        parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    data = frames(args.data_dir)
    config = dict(item['config'])
    if mode == 'evaluate':
        checkpoint = args.checkpoint
        model = create_model(method, config, data, device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
        frame = data[args.split]
        scores = predict(model, method, config, frame, device)
        if args.scores is not None:
            np.save(args.scores, scores)
        print(json.dumps(measure(frame, scores)))
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        if method == 'mf':
            if dataset != 'ml1m':
                raise ValueError('No verified training entry')
            train_mf(data, args.output_dir, device)
        else:
            if device.type != 'cuda':
                raise ValueError('CUDA required')
            module = importlib.import_module('common.' + method)
            module.embedding_size = config['embedding_size']
            module.run_a_trail(config, data_dir=str(args.data_dir), save_mode=True, save_file=str(args.output_dir / 'best.pt'), need_train=True, warm_or_cold=None)
