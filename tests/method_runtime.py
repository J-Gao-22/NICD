import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def check_arguments(dataset):
    root = ROOT/'method'/dataset
    sys.path.insert(0, str(root/'src'))
    from lifecycle_rec.cli import STAGES
    original = argparse.ArgumentParser.parse_args

    class Parsed(Exception):
        pass

    def parse(parser, *args, **kwargs):
        original(parser, *args, **kwargs)
        raise Parsed()

    checked = []
    for path in sorted((root/'configs').glob('*.json')):
        settings = json.loads(path.read_text(encoding='utf-8'))
        if dataset == 'book' and path.stem == 'prepare-inputs':
            module = importlib.import_module('lifecycle_rec.'+STAGES[path.stem])
            for command, arguments in [('profiles', ['--source-profile', 'source.jsonl', '--merged-profile', 'merged.jsonl', '--output-dir', 'output']), ('restore', ['--data-dir', 'frames', '--canonical-profile', 'profiles.jsonl', '--raw-reviews', 'reviews.json', '--output-dir', 'output'])]:
                values = module.build_parser().parse_args([command]+arguments)
                assert values.command == command
                checked.append(path.stem+':'+command)
            continue
        argv = ['check']
        for name, value in settings['arguments'].items():
            if value is True:
                argv.append(name)
            elif value is False or value is None:
                continue
            elif isinstance(value, list):
                argv.extend([name]+[str(item) for item in value])
            else:
                argv.extend([name, str(value)])
        with patch.object(sys, 'argv', argv), patch.object(argparse.ArgumentParser, 'parse_args', parse):
            try:
                importlib.import_module('lifecycle_rec.'+STAGES[path.stem]).main()
            except Parsed:
                checked.append(path.stem)
            else:
                raise AssertionError(('Parser not reached', path.stem))
    return checked


def check_training(dataset):
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
    from lifecycle_rec.recommendation.model import RecommendationModel, collate
    from lifecycle_rec.recommendation.slots import SlotLayout
    from lifecycle_rec.recommendation import collaborative
    from test_baseline_runtime import Tokenizer
    torch.set_num_threads(2)

    def model_loader(*args, **kwargs):
        config = Qwen2Config(vocab_size=10000, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=256)
        return Qwen2ForCausalLM(config)

    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        native_mf = folder/'native.pt'
        torch.save(dict(user_num=4, item_num=5, embedding_size=3, model_state_dict={'user_embedding.weight': torch.randn(4, 3), 'item_embedding.weight': torch.randn(5, 3)}), native_mf)
        with patch.object(AutoModelForCausalLM, 'from_pretrained', side_effect=model_loader), patch.object(AutoTokenizer, 'from_pretrained', side_effect=lambda *a, **k: Tokenizer()):
            native = RecommendationModel('Qwen', native_mf, 8, 16, .05, ['q_proj', 'v_proj'], gradient_checkpointing=False, last_token_only=dataset=='book')
            for name, parameter in native.named_parameters():
                parameter.requires_grad = 'lora_' in name
            parameters = [p for p in native.parameters() if p.requires_grad]
            before = [p.detach().clone() for p in parameters]
            records = [dict(_ids=[1, 2, 3], _length=3, uid=1, iid=2, label=1), dict(_ids=[1, 2, 3, 4], _length=4, uid=2, iid=3, label=0)]
            batch = collate(records, 0, device='cpu')
            optimizer = torch.optim.AdamW(parameters, lr=1e-4)
            for _ in range(2):
                native.train()
                optimizer.zero_grad(set_to_none=True)
                values = native(batch['input_ids'], batch['attention_mask'], batch['lengths'], batch['uid'], batch['iid'], 9454, use_mf_prefix=False)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(values.float(), batch['label'])
                assert torch.isfinite(loss)
                loss.backward()
                assert any(p.grad is not None and p.grad.count_nonzero() for p in parameters)
                optimizer.step()
            assert any(not torch.equal(a, b) for a, b in zip(before, parameters))
            adapter = folder/'adapter'
            adapter.mkdir()
            save_file({k: v.detach().cpu().contiguous() for k, v in get_peft_model_state_dict(native.llm).items()}, str(adapter/'adapter_model.safetensors'))
            native.llm.peft_config['default'].save_pretrained(adapter)
            users, items = (6041, 3687) if dataset == 'ml1m' else (22967, 34154)
            mf = folder/'mf.pth'
            torch.save({'user_embedding.weight': torch.randn(users, 256), 'item_embedding.weight': torch.randn(items, 256)}, mf)
            args = SimpleNamespace(base_model='Qwen', native_mf=native_mf, adapter=adapter, collaborative_mf=mf, device='cpu')
            built = collaborative.build_model(args)
            if dataset == 'ml1m':
                model, tokenizer, layout = built
            else:
                model, layout = built
            header = list(layout.user_header)
            records = [dict(_ids=header+[3, 4], _length=len(header)+2, uid=1, iid=2, label=1), dict(_ids=header+[3, 4, 5], _length=len(header)+3, uid=2, iid=3, label=0)]
            if dataset == 'ml1m':
                from lifecycle_rec.recommendation.slots import collate_records, inject_record
                batch = collate_records([inject_record(row, layout) for row in records], layout.placeholder_id, 'cpu')
            else:
                batch = collaborative.batch(records, layout, 'cpu')
            frozen = {name: value.detach().clone() for name, value in model.named_parameters() if not value.requires_grad}
            parameters = list(model.cf_projector.parameters())
            before = [p.detach().clone() for p in parameters]
            optimizer = torch.optim.AdamW(parameters, lr=1e-4)
            for _ in range(2):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(model(batch).float(), batch['label'])
                assert torch.isfinite(loss)
                loss.backward()
                assert any(p.grad is not None and p.grad.count_nonzero() for p in parameters)
                optimizer.step()
            assert any(not torch.equal(a, b) for a, b in zip(before, parameters))
            for name, parameter in model.named_parameters():
                if name in frozen:
                    assert parameter.grad is None and torch.equal(frozen[name], parameter)
            model.eval()
            values = model(batch).detach()
            batch['label'] = 1-batch['label']
            assert torch.equal(values, model(batch).detach())
            path = folder/'projector.pt'
            torch.save(model.cf_projector.state_dict(), path)
            with torch.no_grad():
                for parameter in parameters:
                    parameter.add_(1)
            model.cf_projector.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
            assert torch.equal(values, model(batch).detach())
        assert not torch.cuda.is_initialized()
    return dict(lora_updates=2, projector_updates=2, frozen_exact=True, label_independent=True, projector_reload_exact=True)


def main():
    dataset = sys.argv[1]
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    result = dict(arguments=check_arguments(dataset), training=check_training(dataset))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
