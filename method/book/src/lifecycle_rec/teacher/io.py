from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
import re
import socket
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open('r', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]

def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with temporary.open('w', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)

def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with temporary.open('w', encoding='utf-8', newline='\n') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)

def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
        handle.flush()
        os.fsync(handle.fileno())

def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def _json_type_matches(value: Any, expected: str) -> bool:
    if expected == 'object':
        return isinstance(value, dict)
    if expected == 'array':
        return isinstance(value, list)
    if expected == 'string':
        return isinstance(value, str)
    if expected == 'integer':
        return isinstance(value, int) and (not isinstance(value, bool))
    if expected == 'number':
        return isinstance(value, (int, float)) and (not isinstance(value, bool))
    if expected == 'boolean':
        return isinstance(value, bool)
    if expected == 'null':
        return value is None
    raise ValueError(f'unsupported JSON schema type: {expected}')

def validate_json_schema(value: Any, schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    root = schema

    def resolve(node: dict[str, Any]) -> dict[str, Any]:
        if '$ref' not in node:
            return node
        reference = node['$ref']
        if not reference.startswith('#/'):
            raise ValueError(f'unsupported external schema reference: {reference}')
        target: Any = root
        for part in reference[2:].split('/'):
            target = target[part.replace('~1', '/').replace('~0', '~')]
        return target

    def visit(current: Any, node: dict[str, Any], path: str) -> None:
        node = resolve(node)
        expected = node.get('type')
        if expected and (not _json_type_matches(current, expected)):
            errors.append(f'schema:{path}:expected {expected}, got {type(current).__name__}')
            return
        if 'const' in node and current != node['const']:
            errors.append(f"schema:{path}:must equal {node['const']!r}")
        if 'enum' in node and current not in node['enum']:
            errors.append(f'schema:{path}:not in allowed enum')
        if isinstance(current, str) and len(current) < int(node.get('minLength', 0)):
            errors.append(f'schema:{path}:string is shorter than minLength')
        if isinstance(current, int) and (not isinstance(current, bool)) and ('minimum' in node) and (current < node['minimum']):
            errors.append(f'schema:{path}:value is below minimum')
        if isinstance(current, int) and (not isinstance(current, bool)) and ('maximum' in node) and (current > node['maximum']):
            errors.append(f'schema:{path}:value is above maximum')
        if isinstance(current, dict):
            required = node.get('required', [])
            for key in required:
                if key not in current:
                    errors.append(f'schema:{path}:missing required property {key!r}')
            properties = node.get('properties', {})
            if node.get('additionalProperties') is False:
                for key in current:
                    if key not in properties:
                        errors.append(f'schema:{path}:additional property {key!r} is not allowed')
            for key, child in current.items():
                if key in properties:
                    visit(child, properties[key], f'{path}.{key}')
        if isinstance(current, list):
            if len(current) < int(node.get('minItems', 0)):
                errors.append(f'schema:{path}:array is shorter than minItems')
            if 'maxItems' in node and len(current) > int(node['maxItems']):
                errors.append(f'schema:{path}:array is longer than maxItems')
            if node.get('uniqueItems'):
                serialized = [json.dumps(item, ensure_ascii=False, sort_keys=True) for item in current]
                if len(serialized) != len(set(serialized)):
                    errors.append(f'schema:{path}:array items are not unique')
            item_schema = node.get('items')
            if isinstance(item_schema, dict):
                for index, child in enumerate(current):
                    visit(child, item_schema, f'{path}[{index}]')
    visit(value, schema, '$')
    return errors

def normalize_model_json_text(raw_text: str) -> tuple[str, dict[str, Any]]:
    text = raw_text.strip()
    metadata = {'raw_output_normalized': False, 'normalization_type': None}
    if text.startswith('```'):
        first_newline = text.find('\n')
        if first_newline != -1:
            fence_header = text[:first_newline].strip().lower()
            body = text[first_newline + 1:].strip()
            if fence_header in {'```', '```json'}:
                if body.endswith('```'):
                    body = body[:-3].strip()
                text = body
                metadata = {'raw_output_normalized': True, 'normalization_type': 'markdown_code_fence_removed'}
    return (text, metadata)

def check_endpoint(endpoint: str, model_name: str, timeout: int=20) -> None:
    url = endpoint.rstrip('/')
    if url.endswith('/v1'):
        models_url = url + '/models'
    else:
        models_url = url + '/v1/models'
    with urllib.request.urlopen(models_url, timeout=timeout) as response:
        payload = json.loads(response.read().decode('utf-8'))
    served = {str(row.get('id')) for row in payload.get('data', [])}
    if model_name not in served:
        raise RuntimeError(f'endpoint does not serve {model_name}; served={sorted(served)}')
