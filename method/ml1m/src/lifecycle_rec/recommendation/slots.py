from dataclasses import dataclass
from typing import Dict, List
import torch
from torch import nn

@dataclass(frozen=True)
class SlotLayout:
    user_header: tuple
    block: tuple
    offsets: tuple
    placeholder_id: int

    @classmethod
    def from_tokenizer(cls, tokenizer):
        encode = lambda s: tuple(tokenizer.encode(s, add_special_tokens=False))
        header = encode('<|im_start|>user\n')
        left = encode('Collaborative user feature: ')
        middle = encode('\nCollaborative target-item feature: ')
        right = encode('\n\n')
        placeholder = tokenizer.pad_token_id
        if placeholder is None:
            placeholder = tokenizer.eos_token_id
        assert placeholder is not None and header and left and middle and right
        block = left + (placeholder,) + middle + (placeholder,) + right
        return cls(header, block, (len(left), len(left) + 1 + len(middle)), int(placeholder))

def inject_record(record: Dict, layout: SlotLayout, max_length: int=2048):
    ids = [int(v) for v in record['_ids']]
    assert len(ids) == int(record['_length']), 'ORIGINAL_LENGTH_MISMATCH'
    assert '_cf_slot_positions' not in record, 'DOUBLE_INJECTION'
    h = list(layout.user_header)
    hits = [i for i in range(len(ids) - len(h) + 1) if ids[i:i + len(h)] == h]
    assert len(hits) == 1, 'AMBIGUOUS_OR_MISSING_USER_ROLE'
    pos = hits[0] + len(h)
    added = list(layout.block)
    new_ids = ids[:pos] + added + ids[pos:]
    assert len(new_ids) <= max_length, 'CF_CONTEXT_OVERFLOW_NO_SILENT_TRUNCATION'
    assert new_ids[:pos] + new_ids[pos + len(added):] == ids
    return dict(record, _ids=new_ids, _length=len(new_ids), _cf_slot_positions=[pos + v for v in layout.offsets], _cf_added_tokens=len(added), _cf_original_length=len(ids))

def collate_records(records: List[Dict], pad_id: int, device):
    assert records
    width = max((r['_length'] for r in records))
    ids = []
    masks = []
    for r in records:
        assert len(r['_ids']) == r['_length']
        ids.append(r['_ids'] + [pad_id] * (width - r['_length']))
        masks.append([1] * r['_length'] + [0] * (width - r['_length']))
    return dict(input_ids=torch.tensor(ids, dtype=torch.long, device=device), attention_mask=torch.tensor(masks, dtype=torch.long, device=device), lengths=torch.tensor([r['_length'] for r in records], dtype=torch.long, device=device), uid=torch.tensor([int(r['uid']) for r in records], dtype=torch.long, device=device), iid=torch.tensor([int(r['iid']) for r in records], dtype=torch.long, device=device), cf_slot_positions=torch.tensor([r['_cf_slot_positions'] for r in records], dtype=torch.long, device=device), label=torch.tensor([float(r['label']) for r in records], dtype=torch.float32, device=device))

class FrozenMFTables(nn.Module):

    def __init__(self, state):
        super().__init__()
        assert set(state) == {'user_embedding.weight', 'item_embedding.weight'}
        u = state['user_embedding.weight'].detach().clone().float()
        i = state['item_embedding.weight'].detach().clone().float()
        assert u.ndim == i.ndim == 2 and u.shape[1] == i.shape[1]
        assert torch.isfinite(u).all() and torch.isfinite(i).all()
        self.user_embedding = nn.Embedding.from_pretrained(u, freeze=True)
        self.item_embedding = nn.Embedding.from_pretrained(i, freeze=True)
        self.dimension = u.shape[1]

    def forward(self, uid, iid):
        assert uid.dtype == iid.dtype == torch.long and uid.shape == iid.shape and (uid.ndim == 1)
        assert torch.all((uid >= 0) & (uid < self.user_embedding.num_embeddings)), 'USER_ID_OUT_OF_RANGE'
        assert torch.all((iid >= 0) & (iid < self.item_embedding.num_embeddings)), 'ITEM_ID_OUT_OF_RANGE'
        return (self.user_embedding(uid), self.item_embedding(iid))

class CollaborativeSlotModel(nn.Module):

    def __init__(self, native_recommendation, mf_state, mid_times=10, placeholder_id=None):
        super().__init__()
        self.native = native_recommendation
        for p in self.native.parameters():
            p.requires_grad = False
        self.cf_mf = FrozenMFTables(mf_state)
        hidden = int(self.native.llm.config.hidden_size)
        d = self.cf_mf.dimension
        self.cf_projector = nn.Sequential(nn.Linear(d, d * mid_times), nn.ReLU(), nn.Linear(d * mid_times, hidden))
        self.placeholder_id = placeholder_id
        self.train(True)

    def train(self, mode=True):
        super().train(mode)
        self.native.eval()
        self.cf_mf.eval()
        self.cf_projector.train(mode)
        return self

    def projected_features(self, uid, iid, mode='normal', permutation=None):
        assert mode in ('normal', 'zero', 'shuffled')
        assert not self.training or mode == 'normal', 'DIAGNOSTIC_MODE_DURING_TRAINING'
        user, item = self.cf_mf(uid, iid)
        features = torch.stack([self.cf_projector(user), self.cf_projector(item)], dim=1)
        if mode == 'zero':
            features = torch.zeros_like(features)
        if mode == 'shuffled':
            assert permutation is not None and permutation.dtype == torch.long
            assert permutation.shape == uid.shape
            assert torch.equal(torch.sort(permutation).values, torch.arange(len(uid), device=uid.device))
            features = features[permutation]
        return features

    def compose_inputs(self, batch, feature_mode='normal', permutation=None):
        ids = batch['input_ids']
        mask = batch['attention_mask']
        lengths = batch['lengths']
        positions = batch['cf_slot_positions']
        n, width = ids.shape
        assert ids.dtype == torch.long and mask.shape == ids.shape
        assert positions.dtype == torch.long and positions.shape == (n, 2)
        assert lengths.shape == (n,) and torch.all((lengths > 0) & (lengths <= width))
        assert torch.all((mask == 0) | (mask == 1)), 'NONBINARY_ATTENTION_MASK'
        expected = torch.arange(width, device=ids.device)[None, :] < lengths[:, None]
        assert torch.equal(mask.bool(), expected), 'RIGHT_PADDING_REQUIRED'
        assert torch.all(positions[:, 0] < positions[:, 1])
        assert torch.all((positions >= 0) & (positions < lengths[:, None] - 1)), 'INVALID_CF_SLOT_POSITION'
        rows = torch.arange(n, device=ids.device)[:, None].expand(n, 2)
        assert torch.all(mask[rows, positions] == 1), 'MASKED_CF_SLOT'
        if self.placeholder_id is not None:
            assert torch.all(ids[rows, positions] == self.placeholder_id), 'CF_PLACEHOLDER_MISMATCH'
        embeddings = self.native.llm.get_input_embeddings()(ids)
        features = self.projected_features(batch['uid'], batch['iid'], feature_mode, permutation)
        assert torch.isfinite(features).all(), 'NONFINITE_CF_FEATURES'
        mixed = embeddings.clone()
        mixed[rows, positions] = features.to(dtype=embeddings.dtype)
        return (mixed, mask, lengths - 1)

    def forward(self, batch, yes_id=9454, feature_mode='normal', permutation=None):
        embeddings, mask, answer = self.compose_inputs(batch, feature_mode, permutation)
        row = torch.arange(len(answer), device=answer.device)
        if self.native.last_token_only:
            causal = self.native.llm.get_base_model()
            hidden = causal.model(inputs_embeds=embeddings, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state
            logits = causal.lm_head(hidden[row, answer, :])
        else:
            output = self.native.llm(inputs_embeds=embeddings, attention_mask=mask, use_cache=False)
            logits = output.logits[row, answer, :]
        return logits[:, yes_id]

    def trainable_names(self):
        names = [n for n, p in self.named_parameters() if p.requires_grad]
        assert names and all((n.startswith('cf_projector.') for n in names))
        return names
