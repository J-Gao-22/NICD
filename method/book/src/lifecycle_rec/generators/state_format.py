import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from lifecycle_rec.generators.grounding import validate as validate_grounding
from lifecycle_rec.generators.repair import repair_output

def extract_json(text):
    text = (text or '').strip()
    if text.startswith('```'):
        text = text.strip('`')
        if '\n' in text:
            text = text.split('\n', 1)[1]
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except Exception:
        start, end = (text.find('{'), text.rfind('}'))
        if start >= 0 and end > start:
            try:
                value = json.loads(text[start:end + 1])
                return value if isinstance(value, dict) else None
            except Exception:
                return None
    return None

def validate_schema(value):
    if not isinstance(value, dict) or set(value) != {'interest_state_node'}:
        return ['bad_outer_schema']
    node = value['interest_state_node']
    required = {'state_type', 'state_label', 'state_summary', 'positive_analysis', 'negative_analysis', 'decision_cues'}
    errors = [f'missing_{key}' for key in required if key not in node]
    cues = node.get('decision_cues') or {}
    if not isinstance(cues.get('author_cues'), dict):
        errors.append('author_cues_not_object')
    return errors
