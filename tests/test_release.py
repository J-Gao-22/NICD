import ast
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_python_sources(self):
        for folder in ('method', 'baselines'):
            for path in (ROOT/folder).rglob('*.py'):
                with self.subTest(path=path.relative_to(ROOT)):
                    source = path.read_text(encoding='utf-8')
                    compile(source, str(path), 'exec')
                    self.assertNotRegex(source, r'(?:/data/|/home/|[A-Z]:\\Users\\|172\.18\.)')

    def test_first_party_imports(self):
        packages = [(ROOT/'baselines', 'common')]
        packages += [(ROOT/'method'/dataset/'src', 'lifecycle_rec') for dataset in ('ml1m', 'book')]
        for base, prefix in packages:
            for path in (base/prefix).rglob('*.py'):
                for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                    if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(prefix+'.'):
                        target = base.joinpath(*node.module.split('.'))
                        with self.subTest(path=path, module=node.module):
                            self.assertTrue(target.with_suffix('.py').is_file() or (target/'__init__.py').is_file())

    def test_requirements_includes(self):
        for path in list((ROOT/'baselines').rglob('requirements*.txt'))+list((ROOT/'method').rglob('*.txt')):
            if path.parent.name != 'requirements' and not path.name.startswith('requirements'):
                continue
            for line in path.read_text(encoding='utf-8').splitlines():
                if line.startswith('-r '):
                    self.assertTrue((path.parent/line[3:].strip()).is_file(), path)

    def test_launchers(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            for dataset in ('ml1m', 'book'):
                root = ROOT/'method'/dataset
                for config in sorted((root/'configs').glob('*.json')):
                    script = root/'scripts'/(config.stem.replace('-', '_')+'.py')
                    with self.subTest(dataset=dataset, stage=config.stem):
                        self.assertTrue(script.is_file())
                        result = subprocess.run([sys.executable, '-B', str(script), '--data-root', str(temp/'data'), '--model-root', str(temp/'models'), '--output-root', str(temp/'outputs'), '--dry-run'], capture_output=True, text=True)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        command = json.loads(result.stdout)
                        self.assertEqual(command[3], config.stem)
                        self.assertFalse(any('${' in argument for argument in command))

    def test_external_output_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            for dataset in ('ml1m', 'book'):
                root = ROOT/'method'/dataset
                result = subprocess.run([sys.executable, '-B', str(root/'scripts/evaluate.py'), '--data-root', directory, '--model-root', str(Path(directory)/'models'), '--output-root', str(Path(directory)/'outputs'), '--dry-run'], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)

    def test_repository_output_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            for dataset in ('ml1m', 'book'):
                root = ROOT/'method'/dataset
                result = subprocess.run([sys.executable, '-B', str(root/'scripts/evaluate.py'), '--data-root', str(Path(directory)/'data'), '--model-root', str(Path(directory)/'models'), '--output-root', str(ROOT/'runs'), '--dry-run'], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)

    def test_no_result_artifacts(self):
        suffixes = {'.npy', '.npz', '.pt', '.pth', '.safetensors', '.bin', '.pkl', '.pickle', '.jsonl', '.csv', '.tsv', '.parquet'}
        for name in ('method', 'baselines'):
            for path in (ROOT/name).rglob('*'):
                if path.is_file():
                    self.assertNotIn(path.suffix.lower(), suffixes, path)
                    self.assertFalse(re.search(r'(?:_scores|_report|_metrics)\.json$', path.name), path)

    def test_final_only_mainline_training(self):
        forbidden = {'evaluate_loss', 'eval_loss', 'generation_eval', 'metrics', 'score', 'evaluate'}
        for dataset in ('ml1m', 'book'):
            root = ROOT/'method'/dataset/'src/lifecycle_rec'
            for folder, name in [('generators', 'train_states.py'), ('generators', 'train_transitions.py'), ('recommendation', 'train.py'), ('recommendation', 'training.py'), ('recommendation', 'collaborative.py')]:
                path = root/folder/name
                if not path.exists():
                    continue
                for node in ast.parse(path.read_text(encoding='utf-8')).body:
                    if isinstance(node, ast.FunctionDef) and node.name in ('main', 'run', 'train', 'train_stage'):
                        calls = {item.func.id for item in ast.walk(node) if isinstance(item, ast.Call) and isinstance(item.func, ast.Name)}
                        self.assertFalse(calls & forbidden, (path, node.name, calls & forbidden))

    def test_generator_optimizer_updates(self):
        for dataset in ('ml1m', 'book'):
            for path in (ROOT/'method'/dataset/'src/lifecycle_rec/generators').glob('train_*.py'):
                tree=ast.parse(path.read_text(encoding='utf-8'))
                calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='step' and isinstance(n.func.value,ast.Name) and n.func.value.id in ('optimizer','opt')]
                self.assertEqual(len(calls),1,path)

    def test_book_mf_configuration(self):
        namespace = {}
        exec((ROOT/'baselines/common/config.py').read_text(encoding='utf-8'), namespace)
        self.assertEqual(namespace['CONFIG']['book']['mf']['config'], dict(embedding_size=256, epochs=20, batch_size=4096, lr=.01, wd=.0001, seed=20260708, optimizer='AdamW'))

    def test_book_teacher_artifact_names(self):
        base = ROOT/'method/book/src/lifecycle_rec/teacher'
        for name in ('judge.py', 'repair_transitions.py', 'judge_repairs.py'):
            source = (base/name).read_text(encoding='utf-8')
            self.assertNotIn('edge_nodes_v3.jsonl', source)
            self.assertIn('transition_candidates.jsonl', source)
        source = (base/'transition_corpora.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        argument = next(node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument' and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == '--base-candidate')
        self.assertEqual(next(keyword.value.value for keyword in argument.keywords if keyword.arg == 'nargs'), '+')

    def test_thinkrec_routing_ownership(self):
        source = ROOT/'baselines/common/thinkrec/runtime.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        evaluate = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'evaluate_final')
        calls = [node for node in ast.walk(evaluate) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertFalse(any(node.func.attr == 'adaptive_lora' for node in calls))
        self.assertEqual(sum(node.func.attr == 'eval' for node in calls), 1)

    def test_ml1m_final_precision_and_batch(self):
        root = ROOT/'method/ml1m/src/lifecycle_rec/recommendation'
        tree = ast.parse((root/'collaborative.py').read_text(encoding='utf-8'))
        score = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'score')
        self.assertFalse(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'autocast' for node in ast.walk(score)))
        tree = ast.parse((root/'evaluate.py').read_text(encoding='utf-8'))
        call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'score')
        self.assertEqual(next(keyword.value.value for keyword in call.keywords if keyword.arg == 'batch_size'), 4)


if __name__ == '__main__':
    unittest.main()
