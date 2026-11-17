import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from lifecycle_rec.paths import PROJECT_ROOT


def expand(value, variables):
    if isinstance(value, str):
        def replace(match):
            name = match.group(1)
            if name not in variables or not variables[name]:
                raise ValueError(('Missing variable', name))
            return str(variables[name])
        return re.sub(r'\$\{([A-Z_]+)\}', replace, value)
    if isinstance(value, list):
        return [expand(item, variables) for item in value]
    return value


def external(path):
    resolved = Path(path).expanduser().resolve()
    repository = PROJECT_ROOT.parents[1]
    if resolved == repository or repository in resolved.parents:
        raise ValueError(('External directory required', str(resolved)))
    return str(resolved)


def main(stage):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT/'configs'/(stage+'.json'))
    parser.add_argument('--data-root', default=os.environ.get('DATA_ROOT'), required=not os.environ.get('DATA_ROOT'))
    parser.add_argument('--model-root', default=os.environ.get('MODEL_ROOT'), required=not os.environ.get('MODEL_ROOT'))
    parser.add_argument('--output-root', default=os.environ.get('OUTPUT_ROOT'), required=not os.environ.get('OUTPUT_ROOT'))
    parser.add_argument('--python')
    parser.add_argument('--dry-run', action='store_true')
    args, extra = parser.parse_known_args()
    settings = json.loads(args.config.read_text(encoding='utf-8'))
    if set(settings) != {'runtime', 'arguments'} or not isinstance(settings['arguments'], dict):
        raise ValueError('Invalid stage configuration')
    variables = dict(os.environ, PROJECT_ROOT=str(PROJECT_ROOT), REPOSITORY_ROOT=str(PROJECT_ROOT.parents[1]), DATA_ROOT=external(args.data_root), MODEL_ROOT=external(args.model_root), OUTPUT_ROOT=external(args.output_root))
    output_path = Path(variables['OUTPUT_ROOT'])
    input_paths = [Path(variables[name]) for name in ('DATA_ROOT', 'MODEL_ROOT')]
    if any(output_path == path or output_path in path.parents or path in output_path.parents for path in input_paths):
        raise ValueError('Output directory overlaps an input directory')
    runtime = settings['runtime']
    if runtime not in ('generation', 'recommendation'):
        raise ValueError(('Invalid runtime', runtime))
    python = args.python or os.environ.get(runtime.upper()+'_PYTHON') or sys.executable
    command = [python, '-B', str(PROJECT_ROOT/'run.py'), stage]
    for name, value in settings['arguments'].items():
        if not name.startswith('--'):
            raise ValueError(('Invalid argument', name))
        value = expand(value, variables)
        if value is True:
            command.append(name)
        elif value is False or value is None:
            continue
        else:
            command.append(name)
            command.extend(str(item) for item in value) if isinstance(value, list) else command.append(str(value))
    command.extend(extra)
    if args.dry_run:
        print(json.dumps(command))
        return
    environment = dict(variables, PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1')
    subprocess.run(command, check=True, cwd=PROJECT_ROOT, env=environment)
