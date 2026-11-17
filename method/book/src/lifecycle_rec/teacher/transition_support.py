from __future__ import annotations
import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any
from transformers import AutoTokenizer
from lifecycle_rec.teacher import transition_signals as edgev2
from lifecycle_rec.teacher import io as stategen
PROMPT_VERSION = 'edge_transition_node_v3_state_transition'
TRANSITION_TYPES = {'positive_continuity', 'positive_shift', 'caution_emergence', 'caution_relief', 'polarity_conflict', 'mixed_update'}
FORBIDDEN_FIELD_RE = re.compile('(target|gold|candidate_set|recommendation_judge|ranked_items|selected_item_id|repair_action)', re.I)
NEG_DIRECTIVE_RE = re.compile('\\b(avoid|reject|downweight|penalize|do not recommend|do not rank|exclude|treat cautiously|less suitable|rank lower|deprioritize|filter out)\\b', re.I)

def norm(value: Any) -> str:
    return re.sub('\\s+', ' ', str(value or '').strip()).lower()

def short(value: Any, words: int) -> str:
    return ' '.join(re.findall('\\S+', str(value or ''))[:words])

def stable_list(values: list[Any], limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = short(value, 8)
        key = norm(text)
        if not text or key in seen:
            continue
        out.append(text)
        seen.add(key)
        if len(out) >= limit:
            break
    return out

def unwrap_state(row: dict[str, Any]) -> dict[str, Any]:
    if isinstance(row.get('state'), dict):
        return row['state']
    if isinstance(row.get('selected_state_node'), dict):
        return row['selected_state_node']
    if isinstance(row.get('state_node'), dict):
        return row['state_node']
    if isinstance(row.get('clean_state_node'), dict):
        return row['clean_state_node']
    cand = row.get('selected_candidate') or {}
    if isinstance(cand, dict):
        if isinstance(cand.get('interest_state_summary'), dict):
            return cand['interest_state_summary']
        return cand
    return row

def list_terms_from_objects(rows: Any, *keys: str, limit: int=8) -> list[str]:
    values: list[str] = []
    if not isinstance(rows, list):
        return []
    for row in rows:
        if isinstance(row, dict):
            for key in keys:
                value = row.get(key)
                if isinstance(value, list):
                    values.extend((str(x) for x in value))
                elif value:
                    values.append(str(value))
        elif row:
            values.append(str(row))
    return stable_list(values, limit)

def state_core_payload(selected_row: dict[str, Any]) -> dict[str, Any]:
    state = unwrap_state(selected_row)
    summary = state.get('summary') or state.get('state_summary')
    positive = state.get('positive_analysis') or {}
    negative = state.get('negative_analysis') or {}
    cues = state.get('decision_cues') or {}
    payload = {'state_id': str(selected_row.get('segment_id') or state.get('state_id') or ''), 'state_type': state.get('state_type') or state.get('polarity') or '', 'state_label': short(state.get('state_label'), 16), 'state_summary': short(summary, 90), 'positive_analysis': positive, 'negative_analysis': negative, 'decision_cues': cues}
    if not positive:
        payload['positive_analysis'] = {'core_factors': [{'factor': short(x.get('factor') if isinstance(x, dict) else x, 14), 'support_terms': list_terms_from_objects([x], 'support_terms', limit=5) if isinstance(x, dict) else []} for x in (state.get('core_preference_factors') or state.get('core_factors') or [])[:4]], 'analysis': short(summary, 60)}
    if not negative:
        payload['negative_analysis'] = {'caution_boundaries': [{'boundary': short(x.get('boundary') if isinstance(x, dict) else x, 14), 'support_terms': list_terms_from_objects([x], 'support_terms', limit=5) if isinstance(x, dict) else [], 'not_causal': True} for x in (state.get('negative_preference_boundaries') or state.get('caution_boundaries') or [])[:4]], 'analysis': ''}
    if not cues:
        match_cues = state.get('match_cues') or []
        payload['decision_cues'] = {'positive_cues': stable_list([term for group in match_cues if isinstance(group, list) for term in group], 10), 'negative_cues': list_terms_from_objects(payload['negative_analysis'].get('caution_boundaries'), 'boundary', 'support_terms', limit=8), 'people_cues': {'directors': [], 'actors': []}}
    return payload

def flatten_positive_terms(state_payload: dict[str, Any]) -> list[str]:
    pos = state_payload.get('positive_analysis') or {}
    cues = state_payload.get('decision_cues') or {}
    values: list[Any] = []
    values.extend(cues.get('positive_cues') or [])
    values.extend(cues.get('positive_match_cues') or [])
    values.extend(list_terms_from_objects(pos.get('core_factors') or [], 'factor', 'support_terms', limit=30))
    values.extend(list_terms_from_objects(pos.get('liked_patterns') or [], 'pattern', 'support_terms', limit=30))
    values.extend(list_terms_from_objects(pos.get('positive_patterns') or [], 'pattern', 'support_terms', limit=30))
    return stable_list(values, 20)

def flatten_negative_terms(state_payload: dict[str, Any]) -> list[str]:
    neg = state_payload.get('negative_analysis') or {}
    cues = state_payload.get('decision_cues') or {}
    values: list[Any] = []
    values.extend(cues.get('negative_cues') or [])
    values.extend(cues.get('negative_match_cues') or [])
    values.extend(list_terms_from_objects(neg.get('caution_boundaries') or [], 'boundary', 'support_terms', limit=30))
    values.extend(list_terms_from_objects(neg.get('rejected_or_risky_patterns') or [], 'pattern', 'support_terms', limit=30))
    values.extend(list_terms_from_objects(neg.get('negative_patterns') or [], 'pattern', 'support_terms', limit=30))
    return stable_list(values, 20)

def flatten_people(state_payload: dict[str, Any]) -> tuple[list[str], list[str]]:
    cues = state_payload.get('decision_cues') or {}
    people = cues.get('people_cues') or {}
    if not people:
        people = cues.get('director_actor_cues') or {}
    if isinstance(people, dict):
        directors = []
        actors = []
        for key in ('directors', 'liked_directors', 'rejected_directors'):
            directors.extend((x for x in people.get(key) or [] if str(x).lower() != 'none'))
        for key in ('actors', 'liked_actors', 'rejected_actors'):
            actors.extend((x for x in people.get(key) or [] if str(x).lower() != 'none'))
        return (stable_list(directors, 6), stable_list(actors, 8))
    return ([], [])

def transition_rows_to_terms(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    out = []
    for row in rows or []:
        term = str(row.get('value') or row.get('term') or row.get('entity_key') or '').strip()
        if not term:
            continue
        out.append({'signal': short(term, 8), 'relation': row.get('relation'), 'from_support_count': len(row.get('from_support_item_ids') or []), 'to_support_count': len(row.get('to_support_item_ids') or [])})
        if len(out) >= limit:
            break
    return out

def compare_terms(from_terms: list[str], to_terms: list[str]) -> dict[str, list[str]]:
    f = {norm(x): x for x in from_terms}
    t = {norm(x): x for x in to_terms}
    return {'retained': [f[k] for k in f.keys() & t.keys()][:8], 'emerging': [t[k] for k in t.keys() - f.keys()][:8], 'fading': [f[k] for k in f.keys() - t.keys()][:8]}

def suggest_type(pos_cmp: dict[str, list[str]], neg_cmp: dict[str, list[str]], conflicts: list[str]) -> str:
    if conflicts:
        return 'polarity_conflict'
    if len(neg_cmp.get('emerging', [])) >= 2:
        return 'caution_emergence'
    if len(neg_cmp.get('fading', [])) >= 2:
        return 'caution_relief'
    if len(pos_cmp.get('retained', [])) >= 2 and len(pos_cmp.get('emerging', [])) <= 1:
        return 'positive_continuity'
    if pos_cmp.get('emerging'):
        return 'positive_shift'
    return 'mixed_update'

def build_transition_analysis(transition: dict[str, Any], from_state: dict[str, Any], to_state: dict[str, Any]) -> dict[str, Any]:
    from_pos, to_pos = (flatten_positive_terms(from_state), flatten_positive_terms(to_state))
    from_neg, to_neg = (flatten_negative_terms(from_state), flatten_negative_terms(to_state))
    pos_cmp = compare_terms(from_pos, to_pos)
    neg_cmp = compare_terms(from_neg, to_neg)
    support = transition.get('interest_change_support') or {}
    retained_rows = edgev2.pick_transition_rows(support.get('retained') or [], 6)
    emerging_rows = edgev2.pick_transition_rows(support.get('emerging') or [], 6)
    fading_rows = edgev2.pick_transition_rows(support.get('fading') or [], 6)
    row_retained = [x['signal'] for x in transition_rows_to_terms(retained_rows, 6)]
    row_emerging = [x['signal'] for x in transition_rows_to_terms(emerging_rows, 6)]
    row_fading = [x['signal'] for x in transition_rows_to_terms(fading_rows, 6)]
    pos_cmp['retained'] = stable_list(pos_cmp['retained'] + row_retained, 8)
    pos_cmp['emerging'] = stable_list(pos_cmp['emerging'] + row_emerging, 8)
    pos_cmp['fading'] = stable_list(pos_cmp['fading'] + row_fading, 8)
    negative_shift = transition.get('negative_shift') or {}
    for key in ('retained', 'emerging', 'fading'):
        rows = transition_rows_to_terms(negative_shift.get(key) or [], 5)
        neg_cmp[key] = stable_list(neg_cmp[key] + [x['signal'] for x in rows], 6)
    from_pos_keys, to_pos_keys = ({norm(x) for x in from_pos}, {norm(x) for x in to_pos})
    from_neg_keys, to_neg_keys = ({norm(x) for x in from_neg}, {norm(x) for x in to_neg})
    conflict_keys = (from_pos_keys | to_pos_keys) & (from_neg_keys | to_neg_keys)
    conflict_terms = stable_list([x for x in from_pos + to_pos + from_neg + to_neg if norm(x) in conflict_keys], 8)
    from_dirs, from_actors = flatten_people(from_state)
    to_dirs, to_actors = flatten_people(to_state)
    people = {'retained_directors': sorted(set(from_dirs) & set(to_dirs))[:3], 'emerging_directors': sorted(set(to_dirs) - set(from_dirs))[:3], 'fading_directors': sorted(set(from_dirs) - set(to_dirs))[:3], 'retained_actors': sorted(set(from_actors) & set(to_actors))[:4], 'emerging_actors': sorted(set(to_actors) - set(from_actors))[:4], 'fading_actors': sorted(set(from_actors) - set(to_actors))[:4]}
    return {'temporal_direction': 'FROM state occurs before TO state; TO state is closer to the target interaction.', 'from_state_type': from_state.get('state_type') or '', 'to_state_type': to_state.get('state_type') or '', 'program_suggested_transition_type': suggest_type(pos_cmp, neg_cmp, conflict_terms), 'positive_signal_comparison': {'retained_positive_terms': pos_cmp['retained'], 'emerging_positive_terms': pos_cmp['emerging'], 'fading_positive_terms': pos_cmp['fading']}, 'negative_caution_comparison': {'retained_negative_terms': neg_cmp['retained'], 'emerging_negative_terms': neg_cmp['emerging'], 'fading_negative_terms': neg_cmp['fading']}, 'polarity_conflict_terms': conflict_terms, 'people_signal_comparison': people, 'program_notes': ['Broad genre or tone terms should not become hard negative evidence.', 'Fading positive interests mean reduced weight, not dislike.', 'Negative cues are non-causal soft cautions unless specific and repeatedly supported.']}

def build_pack_text(row: dict[str, Any]) -> str:
    sections = [('EDGE TASK', row['edge_task']), ('FROM STATE NODE', row['from_state_node']), ('TO STATE NODE', row['to_state_node']), ('TRANSITION ANALYSIS DATA', row['transition_analysis_data'])]
    lines: list[str] = []
    for title, payload in sections:
        lines.append(f'[{title}]')
        lines.append(json.dumps(payload, ensure_ascii=False, separators=(',', ':')))
        lines.append('')
    return '\n'.join(lines).strip() + '\n'

def load_selected(path: Path) -> dict[str, dict[str, Any]]:
    rows = stategen.read_jsonl(path) if path.exists() else []
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get('needs_regeneration'):
            continue
        sample_id = str(row.get('sample_id'))
        segment_id = str(row.get('segment_id') or row.get('state_id'))
        if sample_id and segment_id:
            out[f'{sample_id}||{segment_id}'] = row
    return out

def build_edge_packs(analysis_dir: Path, selected_state_nodes: Path, tokenizer_path: str, prompt_path: Path, schema_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    transitions_path = analysis_dir / 'transition_packages.jsonl'
    transitions = stategen.read_jsonl(transitions_path)
    selected = load_selected(selected_state_nodes)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    packs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for transition in transitions:
        sample_id = str(transition.get('sample_id'))
        user_id = transition.get('user_id')
        from_seg = str(transition.get('from_segment'))
        to_seg = str(transition.get('to_segment'))
        identity = {'sample_id': sample_id, 'user_id': user_id, 'from_segment': from_seg, 'to_segment': to_seg}
        from_row = selected.get(f'{sample_id}||{from_seg}')
        to_row = selected.get(f'{sample_id}||{to_seg}')
        if not from_row or not to_row:
            missing = []
            if not from_row:
                missing.append('missing_from_selected_state_node')
            if not to_row:
                missing.append('missing_to_selected_state_node')
            failures.append({**identity, 'failure_type': 'missing_selected_state_node', 'missing_inputs': missing})
            continue
        from_state = state_core_payload(from_row)
        to_state = state_core_payload(to_row)
        analysis = build_transition_analysis(transition, from_state, to_state)
        if not any(analysis['positive_signal_comparison'].values()) and (not any(analysis['negative_caution_comparison'].values())):
            failures.append({**identity, 'failure_type': 'insufficient_transition_analysis'})
            continue
        row = {**identity, 'prompt_version': PROMPT_VERSION, 'edge_task': {'edge_id': f'{from_seg}->{to_seg}', 'from_state_id': from_seg, 'to_state_id': to_seg, 'expected_output': 'Analyze positive and negative transition updates between adjacent states.'}, 'from_state_node': from_state, 'to_state_node': to_state, 'transition_analysis_data': analysis, 'source_file_hashes': {'transition_packages': stategen.sha256_file(transitions_path) if transitions_path.exists() else None, 'selected_state_nodes': stategen.sha256_file(selected_state_nodes) if selected_state_nodes.exists() else None, 'prompt': stategen.sha256_file(prompt_path) if prompt_path.exists() else None, 'schema': stategen.sha256_file(schema_path) if schema_path.exists() else None}}
        text = build_pack_text(row)
        row['pack_text'] = text
        row['pack_token_count'] = len(tokenizer.encode(text))
        row['pack_hash'] = stategen.sha256_bytes(text.encode('utf-8'))
        packs.append(row)
    stats = {'edge_generation_version': PROMPT_VERSION, 'transition_count': len(transitions), 'pack_build_success': len(packs), 'edge_pack_build_failures': len(failures), 'selected_state_node_count': len(selected), 'pack_token_count': _dist([int(x['pack_token_count']) for x in packs]), 'failure_type_counts': dict(Counter((x.get('failure_type') for x in failures)))}
    return (packs, failures, stats)

def _dist(values: list[int]) -> dict[str, Any]:
    if not values:
        return {'count': 0}
    vals = sorted(values)

    def pct(p: float) -> int:
        return vals[min(len(vals) - 1, int(round((len(vals) - 1) * p)))]
    return {'count': len(vals), 'min': vals[0], 'p50': pct(0.5), 'p90': pct(0.9), 'p95': pct(0.95), 'max': vals[-1], 'mean': sum(vals) / len(vals)}

def validate_edge_node(output: dict[str, Any], pack: dict[str, Any], schema: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    schema_errors = stategen.validate_json_schema(output, schema)
    errors: list[str] = []
    warnings: list[str] = []
    node = output.get('edge_node') if isinstance(output, dict) else None
    if not isinstance(node, dict):
        return (schema_errors, ['missing_edge_node'], warnings)
    if node.get('from_state_id') != pack.get('from_segment'):
        errors.append('from_state_id_mismatch')
    if node.get('to_state_id') != pack.get('to_segment'):
        errors.append('to_state_id_mismatch')
    if node.get('transition_type') not in TRANSITION_TYPES:
        errors.append('invalid_transition_type')
    text = json.dumps(output, ensure_ascii=False).lower()
    if NEG_DIRECTIVE_RE.search(text):
        errors.append('hard_negative_directive_present')

    def walk(value: Any, path: str='$') -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if FORBIDDEN_FIELD_RE.search(str(key)):
                    errors.append(f'forbidden_field:{path}.{key}')
                walk(child, f'{path}.{key}')
        elif isinstance(value, list):
            for idx, child in enumerate(value):
                walk(child, f'{path}[{idx}]')
    walk(output)
    for item in (node.get('positive_transition') or {}).get('fading') or []:
        if item.get('interpretation') != 'reduced_weight_not_negative':
            errors.append('fading_positive_missing_reduced_weight_interpretation')
    for group in ('retained', 'emerging', 'fading'):
        for item in (node.get('negative_caution_transition') or {}).get(group) or []:
            if item.get('not_causal') is not True:
                errors.append(f'negative_caution_missing_not_causal:{group}')
    if not any(((node.get('positive_transition') or {}).get(k) for k in ('retained', 'emerging', 'fading'))) and (not any(((node.get('negative_caution_transition') or {}).get(k) for k in ('retained', 'emerging', 'fading')))):
        errors.append('empty_transition_updates')
    return (schema_errors, errors, warnings)
