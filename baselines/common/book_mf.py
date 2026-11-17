from __future__ import annotations
import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from common.config import CONFIG

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')
    tmp.replace(path)

class MFData(Dataset):

    def __init__(self, df: pd.DataFrame):
        self.uid = torch.tensor(df['uid'].astype(int).values, dtype=torch.long)
        self.iid = torch.tensor(df['iid'].astype(int).values, dtype=torch.long)
        self.label = torch.tensor(df['label'].astype(float).values, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.label.numel())

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {'uid': self.uid[idx], 'iid': self.iid[idx], 'label': self.label[idx]}

class MF(nn.Module):

    def __init__(self, user_num: int, item_num: int, dim: int):
        super().__init__()
        self.user_embedding = nn.Embedding(user_num, dim, padding_idx=0)
        self.item_embedding = nn.Embedding(item_num, dim, padding_idx=0)
        nn.init.normal_(self.user_embedding.weight, std=0.02)
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        with torch.no_grad():
            self.user_embedding.weight[0].zero_()
            self.item_embedding.weight[0].zero_()

    def forward(self, uid: torch.Tensor, iid: torch.Tensor) -> torch.Tensor:
        return (self.user_embedding(uid) * self.item_embedding(iid)).sum(dim=-1)

def metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    labels = np.array([r['label'] for r in rows], dtype=np.float32)
    scores = np.array([r['score'] for r in rows], dtype=np.float32)
    users = np.array([r['uid'] for r in rows])
    out = {'auc': float(roc_auc_score(labels, scores)) if len(set(labels.tolist())) > 1 else 0.0}
    per_user = defaultdict(list)
    for r in rows:
        per_user[int(r['uid'])].append(r)
    uauc = []
    ndcg5 = []
    map5 = []
    hr5 = []
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

@torch.no_grad()
def evaluate(model: MF, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    rows = []
    for b in loader:
        uid = b['uid'].to(device)
        iid = b['iid'].to(device)
        score = model(uid, iid).detach().cpu().numpy().tolist()
        for u, i, y, s in zip(b['uid'].numpy().tolist(), b['iid'].numpy().tolist(), b['label'].numpy().tolist(), score):
            rows.append({'uid': int(u), 'iid': int(i), 'label': int(y), 'score': float(s)})
    model.train()
    return metrics(rows)

def main() -> None:
    config = CONFIG['book']['mf']['config']
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--embedding-size', type=int, default=config['embedding_size'])
    ap.add_argument('--epochs', type=int, default=config['epochs'])
    ap.add_argument('--batch-size', type=int, default=config['batch_size'])
    ap.add_argument('--lr', type=float, default=config['lr'])
    ap.add_argument('--weight-decay', type=float, default=config['wd'])
    ap.add_argument('--seed', type=int, default=config['seed'])
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / 'training_config.json', {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
    train = pd.read_pickle(args.data_dir / 'train_ood2.pkl')
    valid = pd.read_pickle(args.data_dir / 'valid_ood2.pkl')
    test = pd.read_pickle(args.data_dir / 'test_ood2.pkl')
    user_num = int(max(train.uid.max(), valid.uid.max(), test.uid.max())) + 1
    item_num = int(max(train.iid.max(), valid.iid.max(), test.iid.max())) + 1
    device = args.device
    model = MF(user_num, item_num, args.embedding_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    train_loader = DataLoader(MFData(train), batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(MFData(valid), batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(MFData(test), batch_size=args.batch_size, shuffle=False, num_workers=2)
    best_auc = -1.0
    best_report = None
    log = []
    for epoch in range(args.epochs):
        losses = []
        for b in train_loader:
            opt.zero_grad(set_to_none=True)
            uid = b['uid'].to(device)
            iid = b['iid'].to(device)
            label = b['label'].to(device)
            score = model(uid, iid)
            loss = loss_fn(score.float(), label.float())
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        valid_m = evaluate(model, valid_loader, device)
        row = {'epoch': epoch, 'train_loss': float(np.mean(losses)), 'valid': valid_m}
        print(json.dumps(row, ensure_ascii=False), flush=True)
        log.append(row)
        if valid_m['auc'] > best_auc:
            best_auc = valid_m['auc']
            best_report = row
            torch.save({'model_state_dict': model.state_dict(), 'user_num': user_num, 'item_num': item_num, 'embedding_size': args.embedding_size, 'best_valid': valid_m}, args.output_dir / 'best_mf.pt')
    test_m = evaluate(model, test_loader, device)
    report = {'best': best_report, 'last_test': test_m, 'user_num': user_num, 'item_num': item_num, 'embedding_size': args.embedding_size, 'train_rows': len(train), 'valid_rows': len(valid), 'test_rows': len(test), 'log': log}
    write_json(args.output_dir / 'mf_report.json', report)
    torch.save({'model_state_dict': model.state_dict(), 'user_num': user_num, 'item_num': item_num, 'embedding_size': args.embedding_size, 'last_test': test_m}, args.output_dir / 'last_mf.pt')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    best = torch.load(args.output_dir / 'best_mf.pt', map_location='cpu', weights_only=True)
    torch.save(best['model_state_dict'], args.output_dir / 'best.pt')
if __name__ == '__main__':
    main()
