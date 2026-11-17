from __future__ import annotations
import argparse
import collections
import hashlib
import json
import re
from pathlib import Path
from typing import Any
SYSTEM_EDGE = "You are a movie recommendation classifier.\n\nYou will receive a user's ordered lifecycle interest chain before a target movie and the target movie's semantic profile.\nThe lifecycle chain is chronological: S1 is earlier, S2 is later, and S3 is the most recent when present.\nEach state summarizes one segment of prior user interactions.\nTransition nodes are placed between adjacent states. Use them only as ranking cues:\n- emerging positive interests in a later state may receive recency weight;\n- retained positive interests may indicate stable anchors;\n- fading interests are reduced context, not dislike;\n- negative/caution signals are soft non-causal risk signals, not hard rejection rules.\n\nDecision policy:\n1. First identify the strongest positive preference matches between the target and the lifecycle chain.\n2. Use transition nodes to distinguish recent emerging interests from stable retained interests.\n3. Then check whether the target matches any specific caution boundary.\n4. Do not answer No only because the target shares a broad genre or common term with a caution signal.\n5. If the target strongly matches recent or repeated positive cues and only weakly overlaps with caution signals, prefer Yes.\n6. Answer No when specific caution matches are stronger than positive preference matches, or when the target lacks meaningful positive matches.\n\nOutput format:\nAnswer with exactly one token: Yes or No.\nDo not output a reason, JSON, markdown, or analysis."

def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def clean(x: Any) -> str:
    return re.sub('\\s+', ' ', str(x or '').replace('\n', ' ')).strip()

def target_section(user_prompt: str) -> str:
    marker = '\nTarget movie:\n'
    return 'Target movie:\n' + user_prompt.split(marker, 1)[1]

def state_text_from_base_user(base_user: str) -> list[tuple[str, str]]:
    chain = base_user.split('\nTarget movie:\n', 1)[0]
    parts = re.split('(?m)^\\[STATE\\s+(S\\d+)\\]\\s*$', chain)
    out: list[tuple[str, str]] = []
    for i in range(1, len(parts), 2):
        sid = parts[i]
        body = parts[i + 1].strip()
        out.append((sid, f'[STATE {sid}]\n{body}'.strip()))
    return out

def edge_cue(edge: dict[str, Any]) -> str:
    node = edge.get('edge') or edge.get('edge_node') or edge
    label = clean(node.get('transition_label'))
    pos = node.get('positive_transition') or {}
    neg = node.get('negative_caution_transition') or {}
    emerging = [clean(x) for x in pos.get('emerging') or [] if clean(x)]
    retained = [clean(x) for x in pos.get('retained') or [] if clean(x)]
    fading = [clean(x) for x in pos.get('fading') or [] if clean(x)]
    neg_em = [clean(x) for x in neg.get('emerging') or [] if clean(x)]
    lines = [f"[TRANSITION {edge['from_segment']} -> {edge['to_segment']}]"]
    if label:
        lines.append(f'Transition label: {label}')
    lines.append('Ranking cue: emerging positive interests are the main recency signal.')
    lines.append('Emerging positive interests: ' + (', '.join(emerging[:5]) if emerging else 'none'))
    return '\n'.join(lines)

def build_edge_test_data(base_data_dir: Path, edge_output_dir: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base_chain = read_jsonl(base_data_dir / 'test_chain_yesno.jsonl')
    base_plain = read_jsonl(base_data_dir / 'test_plain_yesno.jsonl')
    edges = read_jsonl(edge_output_dir / 'test_edge_mintrim_outputs.jsonl')
    by_sample: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for e in edges:
        by_sample[str(e['sample_id'])].append(e)
    for sid in list(by_sample):
        by_sample[sid].sort(key=lambda r: (str(r['from_segment']), str(r['to_segment'])))
    chain_rows = []
    missing = 0
    for row in base_chain:
        states = state_text_from_base_user(row['user'])
        if len(states) <= 1:
            new_user = row['user']
        else:
            expected = [(states[i][0], states[i + 1][0]) for i in range(len(states) - 1)]
            sample_edges = {(str(e['from_segment']), str(e['to_segment'])): e for e in by_sample.get(str(row['sample_id']), [])}
            if any((pair not in sample_edges for pair in expected)):
                missing += 1
                continue
            blocks = ['Lifecycle interest chain before the target:']
            for idx, (sid, text) in enumerate(states):
                blocks.append(text)
                if idx < len(states) - 1:
                    pair = (sid, states[idx + 1][0])
                    blocks.append(edge_cue(sample_edges[pair]))
            new_user = '\n\n'.join(blocks) + '\n\n' + target_section(row['user'])
        nr = dict(row)
        nr['system'] = SYSTEM_EDGE
        nr['user'] = new_user
        nr['input_type'] = 'chain_edge'
        nr['chain_num_edges'] = max(0, int(nr.get('chain_num_states', 1)) - 1)
        nr['chain_content_hash'] = stable_hash({'system': nr['system'], 'user': nr['user']})
        chain_rows.append(nr)
    write_jsonl(out_dir / 'test_chain_yesno.jsonl', chain_rows)
    write_jsonl(out_dir / 'test_plain_yesno.jsonl', base_plain)
    report = {'base_chain_rows': len(base_chain), 'edge_rows': len(edges), 'test_chain_rows': len(chain_rows), 'missing_edge_chain_rows': missing, 'test_plain_rows': len(base_plain)}
    write_json(out_dir / 'data_report.json', report)
    return report
