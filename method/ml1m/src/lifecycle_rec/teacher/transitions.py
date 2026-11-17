from __future__ import annotations
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    tmp.replace(path)

def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')
    tmp.replace(path)

def seg_num(seg: Any) -> int:
    m = re.search('\\d+', str(seg or ''))
    return int(m.group(0)) if m else 0

def candidate_payload(row: dict[str, Any]) -> dict[str, Any]:
    cand = (row.get('model_output') or {}).get('candidate')
    return cand if isinstance(cand, dict) else {}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--state-candidates', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--candidate-id', default='C1')
    args = ap.parse_args()
    rows = read_jsonl(args.state_candidates)
    selected: dict[tuple[int, int, str], dict[str, Any]] = {}
    skipped = Counter()
    for row in rows:
        if str(row.get('candidate_id')) != args.candidate_id:
            skipped['non_c1'] += 1
            continue
        cand = candidate_payload(row)
        if cand.get('candidate_status') != 'proposed':
            skipped['not_proposed'] += 1
            continue
        try:
            uid = int(row['user_id'])
            iid = int(row['item_id'])
        except Exception:
            skipped['missing_user_or_item'] += 1
            continue
        seg = str(row.get('segment_id') or '')
        if not seg:
            skipped['missing_segment_id'] += 1
            continue
        selected[uid, iid, seg] = row
    by_sample: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for (_uid, _iid, _seg), row in selected.items():
        by_sample[int(row['user_id']), int(row['item_id'])].append(row)
    for rs in by_sample.values():
        rs.sort(key=lambda r: seg_num(r.get('segment_id')))
    selected_rows: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    for (uid, iid), rs in sorted(by_sample.items()):
        sample_id = f'u{uid}_target_{iid}'
        for row in rs:
            cand = candidate_payload(row)
            selected_rows.append({'sample_id': sample_id, 'user_id': uid, 'item_id': iid, 'segment_id': str(row.get('segment_id')), 'selected_candidate_id': args.candidate_id, 'selected_candidate': cand, 'state_node': cand, 'source_pack_hash': row.get('pack_hash'), 'source_state_candidates': str(args.state_candidates), 'source_logprob_summary': row.get('logprob_summary')})
        for left, right in zip(rs, rs[1:]):
            transitions.append({'sample_id': sample_id, 'user_id': uid, 'item_id': iid, 'from_segment': str(left.get('segment_id')), 'to_segment': str(right.get('segment_id')), 'interest_change_support': {'retained': [], 'emerging': [], 'fading': []}, 'negative_shift': {'retained': [], 'emerging': [], 'fading': []}})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = args.output_dir / 'analysis_inputs'
    write_jsonl(args.output_dir / 'selected_state_nodes_c1.jsonl', selected_rows)
    write_jsonl(analysis_dir / 'transition_packages.jsonl', transitions)
    report = {'candidate_id': args.candidate_id, 'source_candidates': str(args.state_candidates), 'source_rows': len(rows), 'selected_state_rows': len(selected_rows), 'unique_samples': len(by_sample), 'segment_count_distribution': dict(Counter((len(v) for v in by_sample.values()))), 'transition_edges': len(transitions), 'skipped': dict(skipped), 'output_selected_state_nodes': str(args.output_dir / 'selected_state_nodes_c1.jsonl'), 'output_transition_packages': str(analysis_dir / 'transition_packages.jsonl')}
    write_json(args.output_dir / 'state_c1_edge_input_report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
if __name__ == '__main__':
    main()
