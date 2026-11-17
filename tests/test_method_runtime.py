import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest


class MethodRuntimeTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers') and importlib.util.find_spec('peft'), 'Recommendation runtime required')
    def test_mainline_stage_arguments_and_training(self):
        for dataset in ('ml1m', 'book'):
            with self.subTest(dataset=dataset):
                result = subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('method_runtime.py')), dataset], capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2'))
                self.assertEqual(result.returncode, 0, result.stderr[-6000:]+result.stdout[-2000:])


if __name__ == '__main__':
    unittest.main()
