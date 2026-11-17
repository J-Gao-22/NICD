import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from lifecycle_rec.generators.transition_format import SCHEMA_VERSION, SYSTEM_PROMPT, USER_INSTRUCTION, mintrim_output, validate_output, write_json, write_jsonl
BOOK_USER_INSTRUCTION = USER_INSTRUCTION + '\nAlways include transition_type. Choose exactly one of: positive_continuity, positive_shift, caution_emergence, caution_relief, polarity_conflict, mixed_update. Include every required edge_node field.'

def rows(path: Path):
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)

def pack_index(path: Path) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    by_hash = {}
    by_sample = defaultdict(list)
    for row in rows(path):
        key = row['pack_hash']
        if key in by_hash:
            raise ValueError(f'duplicate pack_hash: {key}')
        by_hash[key] = row
        by_sample[row['sample_id']].append(row)
    return (by_hash, by_sample)

def candidate_index(paths: list[Path]) -> dict[tuple[str, str], dict]:
    result = {}
    for path in paths:
        for row in rows(path):
            key = (row['pack_hash'], str(row.get('generation_variant') or row.get('edge_candidate_id')))
            if key in result:
                raise ValueError(f'duplicate candidate: {key}')
            result[key] = row
    return result

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--selected-states', type=Path, required=True)
    ap.add_argument('--edge-repair-results', type=Path, required=True)
    ap.add_argument('--state-repair-results', type=Path, required=True)
    ap.add_argument('--base-packs', type=Path, required=True)
    ap.add_argument('--regenerated-packs', type=Path, required=True)
    ap.add_argument('--base-candidate', type=Path, nargs='+', required=True)
    ap.add_argument('--regenerated-candidates', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=2026)
    args = ap.parse_args()
    status_by_sample = {}
    for row in rows(args.selected_states):
        sample_id = row['sample_id']
        status = row['final_source_status']
        if sample_id in status_by_sample and status_by_sample[sample_id] != status:
            raise ValueError(f'conflicting final status for {sample_id}')
        status_by_sample[sample_id] = status
    edge_repairs = {row['sample_id']: row for row in rows(args.edge_repair_results)}
    state_repairs = {row['sample_id']: row for row in rows(args.state_repair_results)}
    base_packs, base_by_sample = pack_index(args.base_packs)
    regen_packs, regen_by_sample = pack_index(args.regenerated_packs)
    base_candidates = candidate_index(args.base_candidate)
    regen_candidates = candidate_index([args.regenerated_candidates])
    output = []
    failures = []
    status_counts = Counter()
    for sample_id, status in status_by_sample.items():
        status_counts[status] += 1
        old_packs = base_by_sample.get(sample_id, [])
        selected_regen = {}
        if status == 'state_repaired_success':
            repair_key = state_repairs.get(sample_id, {}).get('repair_attempt_key')
            if not repair_key:
                failures.append({'sample_id': sample_id, 'reason': 'missing_successful_state_repair_key'})
                continue
            selected_regen = {(row['from_segment'], row['to_segment']): row for row in regen_by_sample.get(sample_id, []) if row.get('repair_attempt_key') == repair_key}
        edges = {(p['from_segment'], p['to_segment']) for p in old_packs} | set(selected_regen)
        for from_seg, to_seg in sorted(edges):
            base_pack = next((p for p in old_packs if p['from_segment'] == from_seg and p['to_segment'] == to_seg), None)
            pack = selected_regen.get((from_seg, to_seg), base_pack)
            slot = 'E1'
            source = 'regenerated' if pack is not base_pack else 'base'
            if status == 'edge_repaired_success':
                repair = edge_repairs.get(sample_id, {})
                if repair.get('target_node') == f'{from_seg}->{to_seg}':
                    slot = repair.get('new_candidate_id')
            if not pack or slot not in ('E1', 'E2', 'E3'):
                failures.append({'sample_id': sample_id, 'edge': [from_seg, to_seg], 'reason': 'missing_pack_or_slot'})
                continue
            if source == 'regenerated' and pack['pack_hash'] not in regen_packs:
                raise AssertionError('regenerated pack hash missing')
            candidate_by_hash = regen_candidates if source == 'regenerated' else base_candidates
            candidate = candidate_by_hash.get((pack['pack_hash'], slot))
            if candidate is None:
                failures.append({'sample_id': sample_id, 'edge': [from_seg, to_seg], 'slot': slot, 'source': source, 'reason': 'missing_selected_candidate'})
                continue
            if any((candidate.get(key) != pack.get(key) for key in ('sample_id', 'from_segment', 'to_segment', 'pack_hash'))):
                raise ValueError(f'candidate/pack mismatch for {sample_id} {from_seg}->{to_seg}')
            pack_text = pack.get('pack_text')
            if not isinstance(pack_text, str) or not pack_text:
                failures.append({'sample_id': sample_id, 'edge': [from_seg, to_seg], 'reason': 'missing_pack_text'})
                continue
            marker = re.search('"(?:target[^"\\n]*|gold[^"\\n]*|candidate_set|judge_score|label)"\\s*:|^\\s*(?:TARGET BOOK|TARGET ITEM|GOLD LABEL)\\b', pack_text, flags=re.I | re.M)
            if marker:
                failures.append({'sample_id': sample_id, 'edge': [from_seg, to_seg], 'reason': 'possible_target_leakage', 'marker': marker.group(), 'context': pack_text[max(0, marker.start() - 60):marker.end() + 60]})
                continue
            node = mintrim_output(candidate)
            errors = validate_output(node, from_seg, to_seg)
            if errors:
                failures.append({'sample_id': sample_id, 'edge': [from_seg, to_seg], 'reason': 'invalid_output', 'errors': errors})
                continue
            input_obj = {'instruction': BOOK_USER_INSTRUCTION, 'pack_text': pack_text}
            input_hash = hashlib.sha256(json.dumps(input_obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            output.append({'record_id': f'{sample_id}::{from_seg}_to_{to_seg}::{slot}::{input_hash[:12]}', 'sample_id': sample_id, 'from_segment': from_seg, 'to_segment': to_seg, 'edge_candidate_slot': slot, 'final_source_status': status, 'teacher_model': candidate.get('model_name'), 'pack_hash': pack['pack_hash'], 'input': input_obj, 'output': node})
    sample_ids = sorted({row['sample_id'] for row in output})
    random.Random(args.seed).shuffle(sample_ids)
    n_train = int(len(sample_ids) * 0.8)
    n_valid = int(len(sample_ids) * 0.1)
    split_of = {sample_id: 'train' if i < n_train else 'valid' if i < n_train + n_valid else 'test' for i, sample_id in enumerate(sample_ids)}
    splits = {split: [row for row in output if split_of[row['sample_id']] == split] for split in ('train', 'valid', 'test')}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / 'edge_transition_sft_all.jsonl', output)
    for split, split_rows in splits.items():
        write_jsonl(args.output_dir / f'edge_transition_sft_{split}.jsonl', split_rows)
    write_jsonl(args.output_dir / 'edge_transition_sft_valid_dedup.jsonl', splits['valid'])
    write_jsonl(args.output_dir / 'edge_transition_sft_test_dedup.jsonl', splits['test'])
    write_jsonl(args.output_dir / 'edge_transition_sft_failures.jsonl', failures)
    report = {'selected_samples': len(status_by_sample), 'selected_status_counts': dict(status_counts), 'edge_records': len(output), 'edge_samples': len(sample_ids), 'source_counts': dict(Counter((row['final_source_status'] for row in output))), 'slot_counts': dict(Counter((row['edge_candidate_slot'] for row in output))), 'split_counts': {key: len(value) for key, value in splits.items()}, 'failures': len(failures), 'failure_reasons': dict(Counter((row['reason'] for row in failures)))}
    write_json(args.output_dir / 'edge_transition_sft_report.json', report)
    write_json(args.output_dir / 'edge_transition_sft_config.json', {'system_prompt': SYSTEM_PROMPT, 'user_instruction': BOOK_USER_INSTRUCTION, 'source_files': {key: str(value) for key, value in vars(args).items() if isinstance(value, Path)}})
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
if __name__ == '__main__':
    main()
