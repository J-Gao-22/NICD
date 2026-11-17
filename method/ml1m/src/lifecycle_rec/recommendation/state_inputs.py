from __future__ import annotations
import argparse
import collections
import hashlib
import json
import re
from pathlib import Path
from typing import Any
BROAD_TERMS = {'action', 'adventure', 'comedy', 'drama', 'romance', 'thriller', 'sci-fi', 'science fiction', 'fantasy', 'crime'}
NEGATIVE_WORDING_CLEAN_RE = re.compile('(?i)\\b(avoid|reject|downweight|penalize|do not recommend|do not rank|less suitable|treat cautiously)\\b')
FORBIDDEN_KEY = re.compile('(^|[\\s,{])"?(gold_item_id|target_item_id|is_gold|candidate_source|candidate_origin|internal_trace|target_rank|selected_item_id|ranked_items|teacher_ranking|repair_action_id)"?\\s*:', re.I)
CONCLUSION = re.compile('\\b(would enjoy|would not enjoy|recommend this|not recommend|aligned with (the )?target|less aligned with (the )?target|positive sample|negative sample)\\b', re.I)
CHAIN_SYSTEM = "You are a movie recommendation classifier.\n\nYou will receive a user's ordered lifecycle interest chain before a target movie and the target movie's semantic profile.\nThe lifecycle chain is chronological: S1 is earlier, S2 is later, and S3 is the most recent when present.\nEach state summarizes one segment of prior user interactions:\n- positive analysis means patterns observed in movies the user liked in that segment.\n- negative analysis means non-causal caution/risk patterns observed in movies the user disliked in that segment.\n- decision cues summarize positive cues, negative cues, and director/actor cues that may help compare the target movie.\n\nSome caution signals include derived helper fields:\n- caution_strength: weak, medium, or strong\n- caution_specificity: broad_genre, semantic_pattern, or concrete_style_or_theme\nUse these helper fields only to interpret how specific a caution signal is.\n\nDecision policy:\n1. First identify the strongest positive preference matches between the target and the lifecycle chain.\n2. Then check whether the target matches any specific caution boundary.\n3. Treat negative/caution boundaries as soft non-causal risk signals, not hard rejection rules.\n4. Do not answer No only because the target shares a broad genre or common term with a caution signal.\n5. If the target strongly matches recent or repeated positive preference cues and only weakly overlaps with caution signals, prefer Yes.\n6. Answer No when specific caution matches are stronger than positive preference matches, or when the target lacks meaningful positive preference matches.\n\nOutput format:\nAnswer with exactly one token: Yes or No.\nDo not output a reason, JSON, markdown, or analysis."

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open('r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n')
    tmp.replace(path)

def write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    tmp.replace(path)

def clean(x: Any) -> str:
    return re.sub('\\s+', ' ', str(x or '')).strip()

def trunc_words(x: Any, n: int) -> str:
    words = clean(x).split()
    return ' '.join(words[:n])

def cap_list(values: Any, n: int) -> list[str]:
    out: list[str] = []
    if isinstance(values, list):
        src = values
    elif values is None:
        src = []
    else:
        src = [values]
    for x in src:
        if isinstance(x, list):
            out.extend((clean(y) for y in x))
        else:
            out.append(clean(x))
    dedup, seen = ([], set())
    for x in out:
        if x and x.lower() != 'none' and (x.lower() not in seen):
            seen.add(x.lower())
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

def caution_strength(pattern: dict[str, Any]) -> str:
    ev = len(pattern.get('evidence_titles') or [])
    terms = len(pattern.get('support_terms') or [])
    if ev >= 3 or terms >= 5:
        return 'strong'
    if ev >= 2 or terms >= 3:
        return 'medium'
    return 'weak'

def caution_specificity(pattern: dict[str, Any]) -> str:
    terms = [clean(x).lower() for x in pattern.get('support_terms') or [] if clean(x)]
    if terms and all((t in BROAD_TERMS for t in terms)):
        return 'broad_genre'
    if any((len(t.split()) >= 2 for t in terms)):
        return 'concrete_style_or_theme'
    return 'semantic_pattern'

def node_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return (row.get('postprocessed_output') or row.get('model_output') or {}).get('interest_state_node') or {}

def render_people(cues: dict[str, Any]) -> str:
    da = cues.get('director_actor_cues') or {}
    chunks = []
    for label, key in [('Liked directors', 'liked_directors'), ('Liked actors', 'liked_actors'), ('Risky directors', 'rejected_directors'), ('Risky actors', 'rejected_actors')]:
        vals = [v for v in cap_list(da.get(key), 2) if len(v) >= 2]
        if vals:
            chunks.append(f"{label}: {', '.join(vals)}")
    return '; '.join(chunks) if chunks else 'none'

def render_state(segment_id: str, node: dict[str, Any]) -> str:
    lines = [f'[STATE {segment_id}]']
    if node.get('state_label'):
        lines.append('State label: ' + clean(node['state_label']))
    if node.get('state_summary'):
        lines.append('State summary: ' + trunc_words(node['state_summary'], 60))
    liked = (node.get('positive_analysis') or {}).get('liked_patterns') or []
    lines.append('Positive preference signals:')
    if liked:
        for pat in liked[:3]:
            name = clean(pat.get('pattern') or '')
            terms = cap_list(pat.get('support_terms'), 10)
            ev = cap_list(pat.get('evidence_titles'), 4)
            tail = []
            if terms:
                tail.append(', '.join(terms))
            if ev:
                tail.append('evidence=' + ', '.join(ev))
            lines.append('- ' + name + (f" ({' | '.join(tail)})" if tail else ''))
    else:
        lines.append('- none')
    risky = (node.get('negative_analysis') or {}).get('rejected_or_risky_patterns') or []
    lines.append('Negative / caution signals:')
    if risky:
        for pat in risky[:3]:
            name = clean_caution_text(pat.get('pattern') or '')
            terms = cap_list([clean_caution_text(x) for x in pat.get('support_terms') or []], 10)
            ev = cap_list(pat.get('evidence_titles'), 4)
            tail = []
            if terms:
                tail.append('terms=' + ', '.join(terms))
            if ev:
                tail.append('evidence=' + ', '.join(ev))
            tail.append('caution_strength=' + caution_strength(pat))
            tail.append('caution_specificity=' + caution_specificity(pat))
            lines.append('- ' + name + ' | ' + ' | '.join(tail))
    else:
        lines.append('- none')
    cues = node.get('decision_cues') or {}
    pos = cap_list(cues.get('positive_match_cues'), 10)
    neg = cap_list([clean_caution_text(x) for x in cues.get('negative_match_cues') or []], 10)
    low = cap_list([clean_caution_text(x) for x in cues.get('low_priority_broad_negative_cues') or []], 10)
    lines.append('Decision cues:')
    lines.append('Positive cues: ' + (', '.join(pos) if pos else 'none'))
    neg_text = ', '.join(neg) if neg else 'none'
    if low:
        neg_text += ' | low_priority_broad_negative_cues: ' + ', '.join(low)
    lines.append('Negative cues: ' + neg_text)
    lines.append('Director / actor cues: ' + render_people(cues))
    return '\n'.join(lines)

def sample_id_from_src(src: dict[str, Any]) -> str:
    return f"u{int(src['user_id'])}_target_{int(src['item_id'])}"

def target_section(user_prompt: str) -> str:
    marker = '\nTarget movie:\n'
    if marker not in user_prompt:
        raise ValueError('target_section_missing')
    return 'Target movie:\n' + user_prompt.split(marker, 1)[1].split('\n\nWould the user enjoy', 1)[0].rstrip() + '\n\nWould the user enjoy the target movie?'

def target_title(section: str) -> str:
    m = re.search('(?m)^title:\\s*(.+?)\\s*$', section)
    return m.group(1).strip() if m else ''

def norm(text: str) -> str:
    return re.sub('[^a-z0-9]+', ' ', text.lower()).strip()

def order_key(row: dict[str, Any]) -> tuple[int, str]:
    sid = str(row.get('segment_id') or '')
    m = re.search('(\\d+)', sid)
    return (int(m.group(1)) if m else 999, sid)

def make_user_prompt(chain_text: str, target_text: str) -> str:
    return f"Lifecycle interest chain before the target:\n\n{chain_text}\n\n{target_text}\n\nQuestion:\nBased on the ordered lifecycle interest chain and the target movie's semantic profile, would the user have a positive interaction with the target movie?\n\nAnswer:"

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--generated-dir', type=Path, required=True)
    ap.add_argument('--plain-data-dir', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--all-interactions', type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    src_idx: dict[tuple[str, int, int], dict[str, Any]] = {}
    for r in read_jsonl(args.all_interactions):
        src_idx[str(r['split']), int(r['thinkrec_uid']), int(r['thinkrec_iid'])] = r
    report: dict[str, Any] = {'splits': {}}
    all_excluded = []
    for split in ['train', 'valid', 'test']:
        plain_name = f'{split}_plain_yesno.jsonl'
        plain_rows = read_jsonl(args.plain_data_dir / plain_name)
        grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for r in read_jsonl(args.generated_dir / f'{split}_state_nodes.jsonl'):
            grouped[str(r['sample_id'])].append(r)
        failures = collections.defaultdict(list)
        for r in read_jsonl(args.generated_dir / f'{split}_state_node_failures.jsonl'):
            failures[str(r.get('sample_id'))].append(r)
        out, excluded = ([], [])
        stats = collections.Counter()
        for i, p in enumerate(plain_rows):
            idx = int(p.get('row_index', i))
            src = src_idx.get((split, int(p['uid']), int(p['iid'])))
            sample_id = sample_id_from_src(src) if src else f"missing_u{p['uid']}_target_{p['iid']}"
            reasons = []
            rows = sorted(grouped.get(sample_id, []), key=order_key)
            if failures.get(sample_id):
                reasons.append('state_segment_generation_failed')
            if not rows:
                reasons.append('no_generated_state_segments')
            chain_text = '\n\n'.join((render_state(str(r.get('segment_id') or f'S{j + 1}'), node_from_row(r)) for j, r in enumerate(rows)))
            if FORBIDDEN_KEY.search(chain_text):
                reasons.append('forbidden_structured_marker')
            if CONCLUSION.search(chain_text):
                reasons.append('conclusion_leakage')
            try:
                section = target_section(str(p['user']))
            except ValueError as exc:
                section = ''
                reasons.append(str(exc))
            title = target_title(section)
            if len(norm(title)) >= 5 and norm(title) in norm(chain_text):
                reasons.append('exact_target_title_in_chain')
            if reasons:
                excluded.append({'split': split, 'row_index': idx, 'sample_id': sample_id, 'reason_codes': sorted(set(reasons))})
                stats.update(reasons)
                continue
            out.append({'record_id': f'{split}::state::{sample_id}', 'sample_id': sample_id, 'source_user_id': int(src['user_id']) if src else None, 'source_item_id': int(src['item_id']) if src else int(p['iid']), 'row_index': idx, 'uid': int(p['uid']), 'iid': int(p['iid']), 'label': int(p['label']), 'split': split, 'input_type': 'chain', 'system': CHAIN_SYSTEM, 'user': make_user_prompt(chain_text, section), 'chain_valid': True, 'chain_num_states': len(rows), 'chain_num_edges': 0, 'chain_content_hash': hashlib.sha256(chain_text.encode('utf-8')).hexdigest()})
        write_jsonl(args.output_dir / f'{split}_chain_yesno.jsonl', out)
        write_jsonl(args.output_dir / plain_name, plain_rows)
        all_excluded.extend(excluded)
        report['splits'][split] = {'plain_rows': len(plain_rows), 'generated_sample_groups': len(grouped), 'chain_rows': len(out), 'excluded_rows': len(excluded), 'exclusion_reasons': dict(stats), 'label_distribution': dict(collections.Counter((str(r['label']) for r in out))), 'chain_num_states_distribution': dict(collections.Counter((str(r['chain_num_states']) for r in out)))}
    write_jsonl(args.output_dir / 'chain_leakage_exclusions.jsonl', all_excluded)
    write_json(args.output_dir / 'data_report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
if __name__ == '__main__':
    main()
