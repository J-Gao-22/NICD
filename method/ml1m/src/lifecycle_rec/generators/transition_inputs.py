from __future__ import annotations
import argparse
import asyncio
import collections
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib import request
from lifecycle_rec.generators import transition_format as edge_sft
NEGATIVE_WORDING_CLEAN_RE = re.compile('\\b(avoid(?:ing)?|reject(?:ing)?|downweight(?:ing)?|penaliz(?:e|ing)|do not recommend|do not rank|exclude|treat cautiously|less suitable|rank lower|deprioritize|filter out)\\b', re.I)

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n')
        f.flush()

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    tmp.replace(path)

def clean(text: Any) -> str:
    return re.sub('\\s+', ' ', str(text or '').replace('\n', ' ')).strip()

def trunc_words(text: Any, n: int) -> str:
    ws = clean(text).split()
    return ' '.join(ws[:n]).rstrip(' ,.;:') + ('...' if len(ws) > n else '')

def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def cap_list(values: Any, n: int=10) -> list[str]:
    out: list[str] = []
    for x in values or []:
        if isinstance(x, list):
            for y in x:
                yy = clean(y)
                if yy and yy.lower() != 'none':
                    out.append(yy)
        else:
            xx = clean(x)
            if xx and xx.lower() != 'none':
                out.append(xx)
    dedup: list[str] = []
    seen: set[str] = set()
    for x in out:
        k = x.lower()
        if k not in seen:
            seen.add(k)
            dedup.append(x)
    return dedup[:n]

def clean_caution_text(x: Any) -> str:
    s = clean(x)
    s = re.sub('(?i)^avoid films with a strong focus on\\s+', '', s)
    s = re.sub('(?i)^avoid films that\\s+', '', s)
    s = re.sub('(?i)^avoid\\s+', '', s)
    s = NEGATIVE_WORDING_CLEAN_RE.sub('', s)
    s = re.sub('(?i)\\bfilms?\\s+(with|that|which)\\b', '', s)
    return trunc_words(s.strip(' -:;,.'), 10)

def node_from_state_row(row: dict[str, Any]) -> dict[str, Any]:
    return (row.get('postprocessed_output') or row.get('model_output') or {}).get('interest_state_node') or {}

def terms_from_state_node(node: dict[str, Any], positive: bool) -> list[str]:
    out: list[str] = []
    if positive:
        for pat in (node.get('positive_analysis') or {}).get('liked_patterns') or []:
            if pat.get('pattern'):
                out.append(pat['pattern'])
            out.extend(pat.get('support_terms') or [])
        cues = node.get('decision_cues') or {}
        out.extend(cues.get('positive_match_cues') or [])
    else:
        for pat in (node.get('negative_analysis') or {}).get('rejected_or_risky_patterns') or []:
            if pat.get('pattern'):
                out.append(clean_caution_text(pat['pattern']))
            out.extend((clean_caution_text(t) for t in pat.get('support_terms') or []))
        cues = node.get('decision_cues') or {}
        out.extend((clean_caution_text(t) for t in cues.get('negative_match_cues') or []))
        out.extend((clean_caution_text(t) for t in cues.get('low_priority_broad_negative_cues') or []))
    return cap_list(out, 14 if positive else 10)

def compact_state_node_payload(row: dict[str, Any]) -> dict[str, Any]:
    node = node_from_state_row(row)
    return {'state_id': row.get('segment_id'), 'state_type': node.get('state_type') or 'unknown_state', 'state_label': node.get('state_label') or '', 'state_summary': node.get('state_summary') or '', 'positive_analysis': node.get('positive_analysis') or {}, 'negative_analysis': node.get('negative_analysis') or {}, 'decision_cues': node.get('decision_cues') or {}}

def state_core_payload(row: dict[str, Any]) -> dict[str, Any]:
    return compact_state_node_payload(row)

def ordered_segments(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:

    def key(r: dict[str, Any]) -> tuple[int, str]:
        sid = str(r.get('segment_id') or '')
        m = re.search('(\\d+)', sid)
        return (int(m.group(1)) if m else 999, sid)
    return sorted(rows, key=key)

def set_diff_ordered(a: list[str], b: list[str]) -> list[str]:
    bs = {x.lower() for x in b}
    return [x for x in a if x.lower() not in bs]

def set_inter_ordered(a: list[str], b: list[str]) -> list[str]:
    bs = {x.lower() for x in b}
    return [x for x in a if x.lower() in bs]

def people_cues_from_state(node: dict[str, Any], kind: str) -> list[str]:
    cues = node.get('decision_cues') or {}
    da = cues.get('director_actor_cues') or {}
    if kind == 'directors':
        keys = ['liked_directors', 'rejected_directors', 'risky_directors', 'directors']
    else:
        keys = ['liked_actors', 'rejected_actors', 'risky_actors', 'actors']
    vals: list[str] = []
    for key in keys:
        vals.extend(cap_list(da.get(key), 8))
    return [v for v in cap_list(vals, 12) if v.lower() != 'none']

def build_transition_analysis(f: dict[str, Any], t: dict[str, Any]) -> dict[str, Any]:
    f_pos, t_pos = (terms_from_state_node(f, True), terms_from_state_node(t, True))
    f_neg, t_neg = (terms_from_state_node(f, False), terms_from_state_node(t, False))
    retained_pos = set_inter_ordered(t_pos, f_pos)[:8]
    emerging_pos = set_diff_ordered(t_pos, f_pos)[:8]
    fading_pos = set_diff_ordered(f_pos, t_pos)[:8]
    retained_neg = set_inter_ordered(t_neg, f_neg)[:6]
    emerging_neg = set_diff_ordered(t_neg, f_neg)[:6]
    fading_neg = set_diff_ordered(f_neg, t_neg)[:6]
    conflict_seen: set[str] = set()
    conflicts: list[str] = []
    caution_lc = {v.lower() for v in t_neg + f_neg}
    for term in t_pos + f_pos:
        key = term.lower()
        if key in caution_lc and key not in conflict_seen:
            conflicts.append(term)
            conflict_seen.add(key)
        if len(conflicts) >= 8:
            break
    f_directors, t_directors = (people_cues_from_state(f, 'directors'), people_cues_from_state(t, 'directors'))
    f_actors, t_actors = (people_cues_from_state(f, 'actors'), people_cues_from_state(t, 'actors'))
    if conflicts:
        ttype = 'polarity_conflict'
    elif len(retained_pos) >= 2 and len(emerging_pos) <= 1:
        ttype = 'positive_continuity'
    elif len(emerging_pos) >= 2:
        ttype = 'positive_shift'
    elif len(emerging_neg) >= 2:
        ttype = 'caution_emergence'
    elif len(fading_neg) >= 2:
        ttype = 'caution_relief'
    else:
        ttype = 'mixed_update'
    return {'temporal_direction': 'FROM state occurs before TO state; TO state is closer to the target interaction.', 'from_state_type': f.get('state_type') or 'unknown_state', 'to_state_type': t.get('state_type') or 'unknown_state', 'program_suggested_transition_type': ttype, 'positive_signal_comparison': {'retained_positive_terms': retained_pos, 'emerging_positive_terms': emerging_pos, 'fading_positive_terms': fading_pos}, 'negative_caution_comparison': {'retained_negative_terms': retained_neg, 'emerging_negative_terms': emerging_neg, 'fading_negative_terms': fading_neg}, 'polarity_conflict_terms': conflicts, 'people_signal_comparison': {'retained_directors': set_inter_ordered(t_directors, f_directors)[:6], 'emerging_directors': set_diff_ordered(t_directors, f_directors)[:6], 'fading_directors': set_diff_ordered(f_directors, t_directors)[:6], 'retained_actors': set_inter_ordered(t_actors, f_actors)[:6], 'emerging_actors': set_diff_ordered(t_actors, f_actors)[:6], 'fading_actors': set_diff_ordered(f_actors, t_actors)[:6]}, 'program_notes': ['Broad genre or tone terms should not become hard negative evidence.', 'Fading positive interests mean reduced weight, not dislike.', 'Negative cues are non-causal soft cautions unless specific and repeatedly supported.']}

def build_edge_pack_text(job: dict[str, Any]) -> str:
    edge_task = {'edge_id': job['edge_id'], 'from_state_id': job['from_segment'], 'to_state_id': job['to_segment'], 'expected_output': 'Analyze positive and negative transition updates between adjacent states.'}
    return '\n\n'.join(['[EDGE TASK]\n' + json.dumps(edge_task, ensure_ascii=False, sort_keys=False, separators=(',', ':')), '[FROM STATE NODE]\n' + json.dumps(job['from_state_node'], ensure_ascii=False, sort_keys=False, separators=(',', ':')), '[TO STATE NODE]\n' + json.dumps(job['to_state_node'], ensure_ascii=False, sort_keys=False, separators=(',', ':')), '[TRANSITION ANALYSIS DATA]\n' + json.dumps(job['transition_analysis_data'], ensure_ascii=False, sort_keys=False, separators=(',', ':'))])
EDGE_RUNTIME_SYSTEM_PROMPT = edge_sft.SYSTEM_PROMPT + '\n\nRuntime wording constraints:\n- Keep the edge node compact. Do not repeat the same sentence or phrase.\n- Each analysis field must be one short sentence.\n- transition_summary must be one short sentence.\n- retained/emerging/fading arrays must contain compact semantic terms, not sentences.\n- In negative_caution_transition, never use avoid / avoiding / downweight / reject / penalize / less suitable.\n- Describe negative evidence as "non-causal caution around <term>" or "soft caution signal for <term>".\n- Negative caution is a contrast signal, not a command to reject movies.\n'
EDGE_RUNTIME_USER_INSTRUCTION = edge_sft.USER_INSTRUCTION + '\nLength rules:\n- transition_label: <= 10 words.\n- positive_transition.analysis: one short sentence, <= 22 words.\n- negative_caution_transition.analysis: one short sentence, <= 22 words.\n- transition_summary: one short sentence, <= 30 words.\n- retained/emerging/fading: each item must be a short term, <= 6 words.\n- Do not repeat any sentence.\n\nFor negative caution text, use neutral wording only:\nGood: "non-causal caution around animation and family-friendly tone"\nBad: "avoid films with animation"\n'
SHORT_TERM_SCHEMA = {'type': 'string', 'maxLength': 60}
EDGE_SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['edge_node'], 'properties': {'edge_node': {'type': 'object', 'additionalProperties': False, 'required': ['from_state_id', 'to_state_id', 'transition_label', 'transition_type', 'positive_transition', 'negative_caution_transition', 'polarity_conflicts', 'transition_summary'], 'properties': {'from_state_id': {'type': 'string'}, 'to_state_id': {'type': 'string'}, 'transition_label': {'type': 'string', 'maxLength': 80}, 'transition_type': {'type': 'string', 'enum': sorted(edge_sft.TRANSITION_TYPES)}, 'positive_transition': {'type': 'object', 'additionalProperties': False, 'required': ['analysis', 'retained', 'emerging', 'fading'], 'properties': {'analysis': {'type': 'string', 'maxLength': 180}, 'retained': {'type': 'array', 'items': SHORT_TERM_SCHEMA, 'maxItems': 5}, 'emerging': {'type': 'array', 'items': SHORT_TERM_SCHEMA, 'maxItems': 5}, 'fading': {'type': 'array', 'items': SHORT_TERM_SCHEMA, 'maxItems': 5}}}, 'negative_caution_transition': {'type': 'object', 'additionalProperties': False, 'required': ['analysis', 'retained', 'emerging', 'fading'], 'properties': {'analysis': {'type': 'string', 'maxLength': 180}, 'retained': {'type': 'array', 'items': SHORT_TERM_SCHEMA, 'maxItems': 5}, 'emerging': {'type': 'array', 'items': SHORT_TERM_SCHEMA, 'maxItems': 5}, 'fading': {'type': 'array', 'items': SHORT_TERM_SCHEMA, 'maxItems': 5}}}, 'polarity_conflicts': {'type': 'array', 'items': SHORT_TERM_SCHEMA, 'maxItems': 4}, 'transition_summary': {'type': 'string', 'maxLength': 240}}}}}

def extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith('```'):
        text = re.sub('^```(?:json)?', '', text).strip()
        text = re.sub('```$', '', text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start, end = (text.find('{'), text.rfind('}'))
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return None
    return None

def minimal_structural_repair(obj: dict[str, Any]) -> dict[str, Any]:
    node = obj.get('edge_node') if isinstance(obj, dict) else None
    if isinstance(node, dict):
        neg = node.get('negative_caution_transition')
        if 'polarity_conflicts' not in node and isinstance(neg, dict) and isinstance(neg.get('polarity_conflicts'), list):
            node['polarity_conflicts'] = neg.pop('polarity_conflicts')
    return obj

def make_edge_jobs(state_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_sample: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in state_rows:
        by_sample[str(row['sample_id'])].append(row)
    jobs = []
    for sid, rows in by_sample.items():
        ordered = ordered_segments(rows)
        for i in range(len(ordered) - 1):
            fs, ts = (str(ordered[i]['segment_id']), str(ordered[i + 1]['segment_id']))
            fcore, tcore = (state_core_payload(ordered[i]), state_core_payload(ordered[i + 1]))
            analysis = build_transition_analysis(fcore, tcore)
            job = {'sample_id': sid, 'split': ordered[i].get('split'), 'user_id': ordered[i].get('user_id'), 'item_id': ordered[i].get('item_id'), 'from_segment': fs, 'to_segment': ts, 'edge_id': f'{fs}->{ts}', 'from_state_node': fcore, 'to_state_node': tcore, 'transition_analysis_data': analysis}
            job['pack_text'] = build_edge_pack_text(job)
            job['pack_hash'] = stable_hash({'from': fcore, 'to': tcore, 'analysis': analysis, 'prompt': 'transitions'})
            jobs.append(job)
    return jobs
