import contextlib
import importlib
import io
import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import numpy as np
    import pandas as pd
    import torch
except ImportError as error:
    raise unittest.SkipTest(str(error))

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'baselines'))
torch.set_num_threads(2)


def frames(directory):
    directory.mkdir()
    rows = [dict(uid=1+i%4, iid=1+i%8, his=[[], [1], [1, 2], list(range(1, 9))][i%4], label=(i//4)%2) for i in range(32)]
    data = {split: pd.DataFrame(rows) for split in ('train', 'valid', 'test')}
    for split, frame in data.items():
        frame.to_pickle(directory/(split+'_ood2.pkl'))
    return data


class TraditionalRuntimeTests(unittest.TestCase):
    def test_models_update_and_reload(self):
        from common.runtime import create_model, predict
        for dataset in ('ml1m', 'book'):
            for method in ('mf', 'lightgcn', 'sasrec', 'gsasrec'):
                with self.subTest(dataset=dataset, method=method), tempfile.TemporaryDirectory() as directory:
                    data = frames(Path(directory)/'data')
                    config = dict(embedding_size=8, gcn_layer=2, maxlen=5)
                    torch.manual_seed(17)
                    model = create_model(method, config, data, 'cpu')
                    before = [p.detach().clone() for p in model.parameters()]
                    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
                    users = torch.tensor(data['train'].uid.values)
                    items = torch.tensor(data['train'].iid.values)
                    labels = torch.tensor(data['train'].label.values, dtype=torch.float32)
                    from common.data import seq_dataset_eval
                    history = torch.stack([torch.as_tensor(seq_dataset_eval(data['train'][['uid', 'iid', 'his', 'label']].values, 5)[i][2], dtype=torch.long) for i in range(len(labels))])
                    for _ in range(2):
                        model.train()
                        optimizer.zero_grad(set_to_none=True)
                        values = model(history, items) if method in ('sasrec', 'gsasrec') else model(users, items)
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(values.reshape(-1).float(), labels)
                        self.assertTrue(torch.isfinite(loss))
                        loss.backward()
                        self.assertTrue(any(p.grad is not None and p.grad.count_nonzero() for p in model.parameters()))
                        optimizer.step()
                    self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.parameters())))
                    expected = predict(model, method, config, data['test'], 'cpu')
                    checkpoint = Path(directory)/'model.pt'
                    torch.save(model.state_dict(), checkpoint)
                    restored = create_model(method, config, data, 'cpu')
                    restored.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
                    np.testing.assert_array_equal(expected, predict(restored, method, config, data['test'], 'cpu'))

    def test_native_training_loops(self):
        for dataset in ('ml1m', 'book'):
            for method in ('lightgcn', 'sasrec', 'gsasrec'):
                with self.subTest(dataset=dataset, method=method), tempfile.TemporaryDirectory() as directory:
                    folder = Path(directory)
                    data = frames(folder/'data')
                    module = importlib.import_module('common.'+method)
                    config = dict(embedding_size=8, gcn_layer=2, maxlen=5, lr=.001, wd=.0001, batch_size=8, epoch=2, eval_epoch=1, patience=1)
                    checkpoint = folder/'best.pt'
                    with patch.object(torch.Tensor, 'cuda', lambda self, *a, **k: self.cpu()), patch.object(torch.nn.Module, 'cuda', lambda self, *a, **k: self.cpu()), patch.object(module, 'embedding_size', 8, create=True), contextlib.redirect_stdout(io.StringIO()):
                        module.run_a_trail(config, data_dir=str(folder/'data'), save_mode=True, save_file=str(checkpoint))
                    self.assertTrue(checkpoint.is_file())
                    from common.runtime import create_model, predict
                    model = create_model(method, config, data, 'cpu')
                    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
                    self.assertTrue(np.isfinite(predict(model, method, config, data['test'], 'cpu')).all())
        self.assertFalse(torch.cuda.is_initialized())

    def test_mf_training_entries(self):
        from common import book_mf
        from common.runtime import train_mf
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            data = frames(folder/'data')
            output = folder/'ml1m'
            output.mkdir()
            with contextlib.redirect_stdout(io.StringIO()):
                train_mf(data, output, torch.device('cpu'))
            self.assertTrue((output/'best.pt').is_file())
            output = folder/'book'
            argv = ['train', '--data-dir', str(folder/'data'), '--output-dir', str(output), '--device', 'cpu', '--epochs', '2', '--batch-size', '16', '--embedding-size', '8']
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                book_mf.main()
            native = torch.load(output/'best_mf.pt', map_location='cpu', weights_only=True)['model_state_dict']
            exported = torch.load(output/'best.pt', map_location='cpu', weights_only=True)
            self.assertTrue(all(torch.equal(value, exported[key]) for key, value in native.items()))


class Tokens(dict):
    def __getattr__(self, name):
        return self[name]

    def to(self, device):
        return Tokens({key: value.to(device) for key, value in self.items()})


class Tokenizer:
    unk_token = '<x>'
    unk_token_id = 2
    pad_token_id = 0
    pad_token = '<pad>'
    bos_token = '<s>'
    eos_token = '</s>'
    padding_side = 'left'

    def encode(self, text, **kwargs):
        if text == 'Yes':
            return [9454]
        if text == 'No':
            return [2753]
        return [2 if word in ('<x>', '<|object_ref_end|>') else 3+sum(map(ord, word))%20 for word in re.findall(r'<\|object_ref_end\|>|<x>|[^\s<>]+', text)] or [3]

    def __call__(self, text, **kwargs):
        if isinstance(text, str):
            return Tokens(input_ids=self.encode(text))
        rows = [self.encode(value)[:kwargs.get('max_length', 2048)] for value in text]
        width = max(map(len, rows))
        masks = [[1]*len(row) for row in rows]
        if self.padding_side == 'left':
            masks = [[0]*(width-len(row))+mask for row, mask in zip(rows, masks)]
            rows = [[0]*(width-len(row))+row for row in rows]
        else:
            masks = [mask+[0]*(width-len(row)) for row, mask in zip(rows, masks)]
            rows = [row+[0]*(width-len(row)) for row in rows]
        return Tokens(input_ids=torch.tensor(rows), attention_mask=torch.tensor(masks))

    def batch_decode(self, values):
        return [str(value) for value in values]


class LLMRuntimeTests(unittest.TestCase):
    def test_book_evaluation_batches(self):
        from common.llm import runtime
        frame = pd.DataFrame([dict(uid=1+i//6, iid=i+1, label=i%2) for i in range(12)])
        dataset = [dict(UserID=row.uid, TargetItemID=row.iid, label=row.label) for row in frame.itertuples()]

        class Model:
            def __init__(self):
                self.batches = []

            def eval(self):
                return self

            def generate_for_samples_v3(self, batch):
                self.batches.append(len(batch['label']))
                return {'logits': batch['TargetItemID'].float()}

        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(data=Path(directory), device='cpu')
            config = runtime.settings('book', 'tallrec')
            for split, expected in [('valid_small', 2), ('valid', 4), ('test', 4)]:
                model = Model()
                with patch.object(runtime, 'make_dataset', return_value=dataset), patch.object(pd, 'read_pickle', return_value=frame):
                    runtime.evaluate(model, args, config, split, Path(directory)/split)
                self.assertEqual(model.batches, [expected]*(12//expected))
                self.assertEqual(np.load(Path(directory)/split/'scores.npy').dtype, np.dtype('float16'))

    def test_tallrec_and_collm_updates(self):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
            from common.llm import runtime
        except ImportError as error:
            self.skipTest(str(error))

        def model_loader(*args, **kwargs):
            config = Qwen2Config(vocab_size=10000, hidden_size=16, intermediate_size=32, num_hidden_layers=28, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=256)
            return Qwen2ForCausalLM(config).to(dtype=torch.bfloat16)

        for dataset in ('ml1m', 'book'):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                torch.save({'user_embedding.weight': torch.randn(5, 256), 'item_embedding.weight': torch.randn(9, 256)}, folder/'mf.pt')
                args = SimpleNamespace(mf=folder/'mf.pt', backbone=Path('Qwen2.5-7B-Instruct'), device='cpu', adapter=None, projector=None, operation='train')
                batch = dict(UserID=torch.tensor([1, 2]), TargetItemID=torch.tensor([4, 5]), InteractedItemIDs_pad=torch.tensor([[1, 2], [2, 3]]), InteractedNum=torch.tensor([2, 2]), InteractedItemTitles=['alpha; beta', 'beta; gamma'], TargetItemTitle=['delta', 'epsilon'], label=torch.tensor([1, 0]), reason=['Yes', 'No'])
                for method in ('tallrec', 'collm'):
                    config = runtime.settings(dataset, method)
                    if method == 'collm':
                        args.adapter = folder/'tallrec/adapter'
                    with patch.object(AutoModelForCausalLM, 'from_pretrained', side_effect=model_loader), patch.object(AutoTokenizer, 'from_pretrained', side_effect=lambda *a, **k: Tokenizer()), contextlib.redirect_stdout(io.StringIO()):
                        model = runtime.build_model(args, config)
                        optimizer = runtime.optimizer_for(model, config)
                        parameters = [p for p in model.parameters() if p.requires_grad]
                        before = [p.detach().clone() for p in parameters]
                        model.train() if method == 'tallrec' else model.eval()
                        for _ in range(2):
                            optimizer.zero_grad(set_to_none=True)
                            with torch.amp.autocast('cpu', dtype=torch.bfloat16):
                                loss = model(batch)['loss']
                            self.assertTrue(torch.isfinite(loss))
                            loss.backward()
                            self.assertTrue(any(p.grad is not None and p.grad.count_nonzero() for p in parameters))
                            optimizer.step()
                        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, parameters)))
                        runtime.save_weights(model, folder/method, method)
                        model.eval()
                        with torch.amp.autocast('cpu', dtype=torch.bfloat16):
                            expected = model.generate_for_samples_v3(batch)['logits'].detach().clone()
                        with torch.no_grad():
                            for parameter in parameters:
                                parameter.add_(1)
                        runtime.reload_weights(model, folder/method, method)
                        with torch.amp.autocast('cpu', dtype=torch.bfloat16):
                            actual = model.generate_for_samples_v3(batch)['logits'].detach()
                        self.assertTrue(torch.equal(expected, actual))
                        del model
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
