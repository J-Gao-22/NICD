from __future__ import annotations
import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
SYSTEM_PROMPT = 'You generate lifecycle transition edge nodes between two adjacent user interest states.\n\nThe input is the original Edge Node v3 state-transition pack:\n- EDGE TASK gives from/to state ids and the expected transition direction.\n- FROM STATE NODE is the earlier state.\n- TO STATE NODE is the later state and is closer to the target interaction.\n- TRANSITION ANALYSIS DATA gives program-computed retained, emerging, fading, polarity-conflict, and people-related comparison signals.\n\nYour output should be concise and recommendation-useful:\n- Analyze positive transition and negative caution transition separately.\n- Use retained / emerging / fading as the main angles.\n- Fading positive interests mean reduced positive weight, not dislike.\n- Negative cautions are soft non-causal caution signals, not hard rejection rules.\n- Do not output recommendation_update, people_signal_transition, signal-level why fields, support item ids, judge scores, target/gold labels, or repair metadata.\n- Output exactly one JSON object with top-level key edge_node.'
USER_INSTRUCTION = 'Generate the compact edge node from this input pack.\nKeep from_state_id and to_state_id exactly.\nUse concise transition analysis.\nretained/emerging/fading arrays should contain semantic signal strings, not objects.\nOnly include polarity_conflicts when explicitly supported; otherwise use an empty list.'
TRANSITION_TYPES = {'positive_continuity', 'positive_shift', 'caution_emergence', 'caution_relief', 'polarity_conflict', 'mixed_update'}
NEGATIVE_DIRECTIVE_RE = re.compile('\\b(avoid|reject|downweight|penalize|do not recommend|do not rank|exclude|treat cautiously|less suitable|rank lower|deprioritize|filter out)\\b', re.I)
FORBIDDEN_RE = re.compile('(target|gold|candidate_set|recommendation_judge|ranked_items|selected_item_id|repair_action|support_item_ids|people_signal_transition|recommendation_update)', re.I)

def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def validate_output(output: dict[str, Any], from_segment: str, to_segment: str) -> list[str]:
    errors: list[str] = []
    if set(output.keys()) != {'edge_node'}:
        errors.append('top_level_key_not_edge_node_only')
    node = output.get('edge_node') if isinstance(output, dict) else None
    if not isinstance(node, dict):
        return errors + ['missing_edge_node']
    required = {'from_state_id', 'to_state_id', 'transition_label', 'transition_type', 'positive_transition', 'negative_caution_transition', 'polarity_conflicts', 'transition_summary'}
    missing = required - set(node)
    if missing:
        errors.append('missing_fields:' + ','.join(sorted(missing)))
    if node.get('from_state_id') != from_segment:
        errors.append('from_state_id_mismatch')
    if node.get('to_state_id') != to_segment:
        errors.append('to_state_id_mismatch')
    if node.get('transition_type') not in TRANSITION_TYPES:
        errors.append('invalid_transition_type')
    for section in ('positive_transition', 'negative_caution_transition'):
        sec = node.get(section)
        if not isinstance(sec, dict):
            errors.append(f'{section}_not_object')
            continue
        for key in ('retained', 'emerging', 'fading'):
            if not isinstance(sec.get(key), list):
                errors.append(f'{section}.{key}_not_list')
            elif any((not isinstance(x, str) for x in sec.get(key))):
                errors.append(f'{section}.{key}_contains_non_string')
    if not isinstance(node.get('polarity_conflicts'), list):
        errors.append('polarity_conflicts_not_list')
    pos = node.get('positive_transition') or {}
    neg = node.get('negative_caution_transition') or {}
    if not any((pos.get(k) for k in ('retained', 'emerging', 'fading'))) and (not any((neg.get(k) for k in ('retained', 'emerging', 'fading')))):
        errors.append('empty_transition_signals')
    text = canonical_json(output)
    if NEGATIVE_DIRECTIVE_RE.search(text):
        errors.append('hard_negative_directive_present')
    for key in output.keys():
        if FORBIDDEN_RE.search(str(key)):
            errors.append(f'forbidden_key:{key}')
    for key in node.keys():
        if FORBIDDEN_RE.search(str(key)):
            errors.append(f'forbidden_key:edge_node.{key}')
    return errors
