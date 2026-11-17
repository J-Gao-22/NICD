from __future__ import annotations
from lifecycle_rec.paths import PROMPTS as PROMPT_ROOT
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from transformers import AutoTokenizer
from lifecycle_rec.teacher import transition_support as edgev3

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--state-candidates', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--schema', type=Path, required=True)
    args = parser.parse_args()
    SOURCE = args.state_candidates
    OUT = args.output_dir
    TOKENIZER = args.tokenizer
    VERSION = 'amazon_book_edge_transition_v2_c1_complete_phrases'
    PROMPTS = {key: PROMPT_ROOT / ('transitions_' + key.lower() + '.txt') for key in ('E1', 'E2', 'E3')}
    SCHEMA = args.schema

    def read(path):
        opener = gzip.open if str(path).endswith('.gz') else open
        with opener(path, 'rt', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    def write(path, rows):
        with open(path, 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')

    def file_hash(path):
        digest = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()

    def segnum(value):
        return int(''.join((x for x in str(value) if x.isdigit())) or 0)

    def stable(values, limit):
        out, seen = ([], set())
        for value in values:
            value = str(value).strip()
            key = value.casefold()
            if value and key != 'none' and (key not in seen):
                out.append(value)
                seen.add(key)
            if len(out) >= limit:
                break
        return out

    def authors(state):
        cues = (state.get('decision_cues') or {}).get('author_cues') or {}
        return (stable(cues.get('liked_authors') or [], 6), stable(cues.get('rejected_authors') or [], 6))

    def compare(left, right, limit=6):
        l = {x.casefold(): x for x in left}
        r = {x.casefold(): x for x in right}
        return {'retained': [x for x in left if x.casefold() in r][:limit], 'emerging': [x for x in right if x.casefold() not in l][:limit], 'fading': [x for x in left if x.casefold() not in r][:limit]}

    def object_terms(rows, pattern_key):
        values = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            if row.get(pattern_key):
                values.append(row[pattern_key])
            values.extend(row.get('support_terms') or [])
        return values

    def polarity_terms(state, polarity):
        cues = state.get('decision_cues') or {}
        if polarity == 'positive':
            analysis = state.get('positive_analysis') or {}
            values = list(cues.get('positive_match_cues') or [])
            values += object_terms(analysis.get('liked_patterns') or analysis.get('positive_patterns') or [], 'pattern')
        else:
            analysis = state.get('negative_analysis') or {}
            values = list(cues.get('negative_match_cues') or [])
            values += object_terms(analysis.get('rejected_or_risky_patterns') or analysis.get('negative_patterns') or [], 'pattern')
        return stable(values, 40)

    def transition_analysis(from_state, to_state):
        fp, tp = (polarity_terms(from_state, 'positive'), polarity_terms(to_state, 'positive'))
        fn, tn = (polarity_terms(from_state, 'negative'), polarity_terms(to_state, 'negative'))
        pos = compare(fp, tp, 8)
        neg = compare(fn, tn, 6)
        pos_keys = {x.casefold() for x in fp + tp}
        neg_keys = {x.casefold() for x in fn + tn}
        conflicts = stable([x for x in fp + tp + fn + tn if x.casefold() in pos_keys & neg_keys], 8)
        return {'temporal_direction': 'FROM is earlier and TO is later. Set differences describe state representation only, not preference strength or causal change.', 'from_state_type': from_state.get('state_type') or '', 'to_state_type': to_state.get('state_type') or '', 'program_suggested_transition_type': edgev3.suggest_type(pos, neg, conflicts), 'positive_signal_comparison': {'retained_positive_terms': pos['retained'], 'emerging_positive_terms': pos['emerging'], 'fading_positive_terms': pos['fading']}, 'negative_caution_comparison': {'retained_negative_terms': neg['retained'], 'emerging_negative_terms': neg['emerging'], 'fading_negative_terms': neg['fading']}, 'polarity_conflict_terms': conflicts}
    selected = []
    by_sample = defaultdict(list)
    for row in read(SOURCE):
        if row.get('candidate_id') != 'C1':
            continue
        cand = (row.get('model_output') or {}).get('candidate') or {}
        if cand.get('candidate_status') != 'proposed':
            raise RuntimeError('C1 is not proposed: ' + row['record_id'])
        compact = {'sample_id': row['sample_id'], 'user_id': row['user_id'], 'item_id': row['item_id'], 'segment_id': row['segment_id'], 'selected_candidate_id': 'C1', 'selected_candidate': cand, 'source_pack_hash': row['pack_hash']}
        selected.append(compact)
        by_sample[row['sample_id']].append(compact)
    assert len(selected) == 10066, len(selected)
    assert len(by_sample) == 6000, len(by_sample)
    for values in by_sample.values():
        values.sort(key=lambda r: segnum(r['segment_id']))
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
    source_hash = file_hash(SOURCE)
    prompt_hashes = {key: file_hash(path) for key, path in PROMPTS.items()}
    schema_hash = file_hash(SCHEMA)
    packs = []
    for sample_id, states in sorted(by_sample.items()):
        for left, right in zip(states, states[1:]):
            from_state = edgev3.state_core_payload(left)
            to_state = edgev3.state_core_payload(right)
            analysis = transition_analysis(from_state, to_state)
            fl, fn = authors(from_state)
            tl, tn = authors(to_state)
            analysis['author_signal_comparison'] = {'positive_authors': compare(fl, tl), 'negative_caution_authors': compare(fn, tn), 'evidence_rule': 'Authors appear only when repeated same-polarity state evidence produced an author cue.'}
            analysis['program_notes'] = ['Broad category or tone terms must not become hard negative evidence.', 'Fading means present in FROM and absent from TO state output. It does not prove weaker interest.', 'Emerging means absent from FROM and present in TO state output. It does not prove stronger interest.', 'Below-5 interactions provide non-causal tentative local caution only.', 'Do not infer author preference from an author absent from the state author cues.']
            fs, ts = (left['segment_id'], right['segment_id'])
            row = {'sample_id': sample_id, 'user_id': left['user_id'], 'item_id': left['item_id'], 'from_segment': fs, 'to_segment': ts, 'prompt_version': VERSION, 'edge_task': {'edge_id': f'{fs}->{ts}', 'from_state_id': fs, 'to_state_id': ts, 'expected_output': 'Analyze positive and tentative negative-caution updates between adjacent book-interest states.'}, 'from_state_node': from_state, 'to_state_node': to_state, 'transition_analysis_data': analysis, 'source_file_hashes': {'selected_state_candidates': source_hash, 'edge_prompts': prompt_hashes, 'edge_schema': schema_hash}}
            row['pack_text'] = edgev3.build_pack_text(row)
            row['pack_token_count'] = len(tokenizer.encode(row['pack_text']))
            row['pack_hash'] = hashlib.sha256(row['pack_text'].encode()).hexdigest()
            packs.append(row)
    assert len(packs) == 4066, len(packs)
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / 'selected_state_nodes.jsonl', selected)
    write(OUT / 'transition_packs.jsonl', packs)
    chosen, seen = ([], set())
    strata = [lambda p: any((p['transition_analysis_data']['author_signal_comparison'][k][x] for k in ('positive_authors', 'negative_caution_authors') for x in ('retained', 'emerging', 'fading'))), lambda p: bool(p['transition_analysis_data'].get('polarity_conflict_terms')), lambda p: p['from_segment'] == 'S2' and p['to_segment'] == 'S3', lambda p: True]
    for predicate, quota in zip(strata, (3, 3, 2, 4)):
        added = 0
        for pack in packs:
            key = (pack['sample_id'], pack['from_segment'], pack['to_segment'])
            if key not in seen and predicate(pack):
                chosen.append(pack)
                seen.add(key)
                added += 1
                if added >= quota:
                    break
    assert len(chosen) == 12, len(chosen)
    write(OUT / 'edge_generation_packs_book_v2_smoke12.jsonl', chosen)
    forbidden = ('target_title', 'target_semantic', 'target_profile', 'target_label', 'gold_item', 'recommendation_judge')
    leaks = []
    for p in packs:
        low = p['pack_text'].lower()
        hit = [x for x in forbidden if x in low]
        if hit:
            leaks.append({'sample_id': p['sample_id'], 'from': p['from_segment'], 'to': p['to_segment'], 'markers': hit})
    report = {'version': VERSION, 'samples': len(by_sample), 'selected_c1_states': len(selected), 'transition_packs': len(packs), 'smoke_packs': len(chosen), 'leakage_marker_failures': len(leaks), 'segment_count_distribution': dict(Counter((len(x) for x in by_sample.values()))), 'token_counts': {'min': min((p['pack_token_count'] for p in packs)), 'max': max((p['pack_token_count'] for p in packs)), 'mean': sum((p['pack_token_count'] for p in packs)) / len(packs)}, 'movie_people_fields_in_transition_analysis': sum(('people_signal_comparison' in p['transition_analysis_data'] for p in packs)), 'author_signal_field_coverage': sum((bool(p['transition_analysis_data']['author_signal_comparison']) for p in packs)), 'target_data_in_pack': False, 'source_hashes': {'selected_state_candidates': source_hash, 'edge_prompts': prompt_hashes, 'edge_schema': schema_hash}}
    write(OUT / 'edge_pack_leakage_failures.jsonl', leaks)
    (OUT / 'edge_pack_build_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
