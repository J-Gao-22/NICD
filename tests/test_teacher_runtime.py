import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    path = ROOT/'method/ml1m/src/lifecycle_rec/teacher'/(name+'.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')


class TeacherRuntimeTests(unittest.TestCase):
    def test_pack_population_and_hashes(self):
        module = load('state_tasks')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = root/'samples.jsonl'
            packs = root/'packs.jsonl'
            output = root/'tasks.jsonl'
            write(samples, [dict(user_id=1, item_id=2)])
            rows = [dict(user_id=1, item_id=2, segment_id='S1', candidate_id=candidate, pack_text='evidence', source_pack_hash=hashlib.sha256(b'evidence').hexdigest()) for candidate in ('C2', 'C1', 'C3')]
            write(packs, rows)
            argv = ['check', '--packs', str(packs), '--sample-list', str(samples), '--output', str(output)]
            with patch.object(sys, 'argv', argv):
                module.main()
            result = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([row['candidate_id'] for row in result], ['C2', 'C1', 'C3'])
            self.assertTrue(all(row['pack_hash'] == rows[0]['source_pack_hash'] for row in result))
            rows[0]['pack_text'] = 'changed'
            write(packs, rows)
            with patch.object(sys, 'argv', argv), self.assertRaisesRegex(ValueError, 'Pack hash mismatch'):
                module.main()
            self.assertEqual(result, [json.loads(line) for line in output.read_text().splitlines()])

    def test_state_generation_retry_and_artifacts(self):
        module = load('state_generation')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = dict(user_id=1, item_id=2, segment_id='S1', candidate_id='C1', pack_text='evidence', pack_hash=hashlib.sha256(b'evidence').hexdigest())
            write(root/'tasks.jsonl', [task])
            (root/'prompt.txt').write_text('{{candidate_id}} {{analysis_lens}} {{lens_instruction}}', encoding='utf-8')
            candidate = dict(candidate_id='C1', analysis_lens=module.LENSES['C1'][0], candidate_status='proposed', state_type='mixed', state_label='state', state_summary='summary', positive_analysis={}, negative_analysis={}, decision_cues={})
            responses = [('not json', {}, {}), (json.dumps({'candidate': candidate}), {'avg_token_nll': 1.}, {'prompt_tokens': 10, 'completion_tokens': 20})]
            argv = ['check', '--prompt', str(root/'prompt.txt'), '--pack-tasks', str(root/'tasks.jsonl'), '--output-dir', str(root/'states'), '--workers', '1', '--logprobs']
            with patch.object(sys, 'argv', argv), patch.object(module, 'call_chat', side_effect=responses) as request, contextlib.redirect_stdout(io.StringIO()):
                module.main()
            self.assertEqual(request.call_count, 2)
            self.assertEqual(request.call_args_list[0].args[3], 768)
            self.assertEqual(request.call_args_list[1].args[2][-1]['content'], module.retry_instruction(["ValueError('no_json_object')"]))
            attempts = list(module.read_jsonl(root/'states/state_generation_attempts.jsonl'))
            self.assertEqual([row['status'] for row in attempts], ['request_or_parse_error', 'success'])
            result = list(module.read_jsonl(root/'states/state_candidates.jsonl'))
            self.assertEqual(result[0]['model_output']['candidate'], candidate)
            self.assertEqual(result[0]['pack_hash'], task['pack_hash'])
            self.assertEqual(len(list((root/'states/raw_outputs').glob('*.txt'))), 2)
            self.assertEqual(list(module.read_jsonl(root/'states/state_generation_failures.jsonl')), [])

    def test_transition_alternatives_and_regeneration(self):
        root = ROOT/'method/ml1m/configs'
        outputs = []
        for variant in ('e1', 'e2', 'e3'):
            config = json.loads((root/('teacher-transitions-'+variant+'.json')).read_text())['arguments']
            self.assertEqual(config['--generation-variant'], variant.upper())
            self.assertEqual(config['--seed'], 20260725)
            self.assertEqual(config['--parallel'], 10)
            outputs.append(config['--output-dir'])
        self.assertEqual(len(set(outputs)), 3)
        merged = json.loads((root/'teacher-artifacts.json').read_text())['arguments']['--transition-dirs']
        self.assertEqual(merged, outputs)
        config = json.loads((root/'regenerate-transitions.json').read_text())['arguments']
        self.assertTrue(config['--packs'].endswith('/state_repair/repair_transition_packs.jsonl'))
        self.assertTrue(config['--output-dir'].endswith('/state_repair/transitions'))
        prompt = ROOT/'method/ml1m/prompts/transitions.txt'
        self.assertEqual(hashlib.sha256(prompt.read_bytes()).hexdigest(), '9511604dc92395924e9c0d66885f77dc40eb7b0a0f02829d6f2860cf83c3bfa3')


if __name__ == '__main__':
    unittest.main()
