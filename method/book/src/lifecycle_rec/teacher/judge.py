from __future__ import annotations
import argparse
import concurrent.futures as cf
import hashlib
import importlib.util
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any
SYSTEM_PROMPT = "You are a book recommendation classifier.\n\nYou will receive a user's ordered lifecycle interest chain before a target book and the target book's semantic profile.\nThe lifecycle chain is chronological: S1 is earlier, S2 is later, and S3 is the most recent when present.\nEach state summarizes one segment of prior user interactions:\n- positive analysis means patterns observed in books with positive feedback in that segment.\n- negative analysis means non-causal tentative caution patterns observed in books with below-5 feedback in that segment.\n- decision cues summarize positive cues, negative cues, and supported author cues that may help compare the target book.\n\nThe chain may also contain edge nodes between adjacent states:\n- EDGE S1->S2 describes representation differences from S1 to S2. An edge is transition evidence, not a standalone recommendation rule.\n- Edge nodes are rendered in a positive-focused way.\n- Retained positive signals are stable anchors across states.\n- Emerging positive signals are newer interests and can receive recency weight.\n- Fading positive signals receive less current representation weight and are not negative feedback.\n- Only specific, multi-word negative edge cautions are shown; broad category cautions are hidden to avoid over-penalization.\n- Polarity conflicts mean a broad term appears with mixed positive/caution meaning, so rely on more specific target semantics.\n- If state-level positive evidence and edge-level retained/emerging positive evidence both match the target, prefer Yes unless there is a concrete high-specificity caution match.\n\nDecision policy:\n1. First identify the strongest positive semantic matches between the target and the lifecycle chain.\n2. Use edge nodes to distinguish stable anchors, recent emerging interests, and fading interests.\n3. Then check whether the target matches a specific caution boundary; broad category overlap alone is weak evidence.\n4. Treat below-5 negative/caution evidence as soft, local, and non-causal, not as hard rejection.\n5. Do not answer No only because the target shares a broad category or common term with a caution signal.\n6. If the target strongly matches recent or repeated positive cues and only weakly overlaps with caution signals, prefer Yes.\n7. Answer No only when concrete caution matches are stronger than positive matches, or when the target lacks meaningful positive matches.\n\nOutput format:\nFirst token must be exactly Yes or No.\nThen write a reason grounded in the lifecycle chain, edge transitions, and target semantic profile.\nThe reason must be at most three sentences.\nDo not output JSON or markdown."
FORBIDDEN_FIELDS = 'label|rating|target_item_id|item_id|is_gold|gold|prediction|candidate_source'
FORBIDDEN_FIELD_RE = re.compile(f'(?im)^\\s*(?:{FORBIDDEN_FIELDS})\\s*:|"(?:{FORBIDDEN_FIELDS})"\\s*:')

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n')
    tmp.replace(path)

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    tmp.replace(path)

def clean(value: Any) -> str:
    return re.sub('\\s+', ' ', str(value or '')).strip()

def values(items: list[Any], key: str='signal', limit: int=6) -> list[str]:
    out, seen = ([], set())
    for item in items or []:
        value = clean(item.get(key) if isinstance(item, dict) else item)
        folded = value.casefold()
        if value and folded not in seen:
            out.append(value)
            seen.add(folded)
        if len(out) >= limit:
            break
    return out

def joined(items: list[Any], limit: int=10) -> str:
    return ', '.join(values(items, key='', limit=limit)) if items else 'none'

def plain_values(items: list[Any], limit: int=10) -> str:
    out, seen = ([], set())
    for item in items or []:
        value = clean(item)
        if value and value.casefold() not in seen:
            out.append(value)
            seen.add(value.casefold())
        if len(out) >= limit:
            break
    return ', '.join(out) or 'none'

def render_pattern(row: dict[str, Any]) -> str:
    terms = plain_values(row.get('support_terms') or [], 8)
    titles = plain_values(row.get('evidence_titles') or [], 3)
    return f"- {clean(row.get('pattern'))} ({terms}); evidence: {titles}"

def render_state(row: dict[str, Any]) -> str:
    segment = row['segment_id']
    candidate = row['selected_candidate']
    positive = candidate.get('positive_analysis') or {}
    negative = candidate.get('negative_analysis') or {}
    cues = candidate.get('decision_cues') or {}
    author_cues = cues.get('author_cues') or {}
    lines = [f'[STATE {segment}]', 'State label: ' + clean(candidate.get('state_label')), 'State summary: ' + clean(candidate.get('state_summary')), 'Positive analysis:']
    patterns = positive.get('liked_patterns') or []
    lines.extend((render_pattern(pattern) for pattern in patterns[:4]))
    if not patterns:
        lines.append('- none')
    lines.append('Negative / caution analysis:')
    caution = negative.get('rejected_or_risky_patterns') or []
    lines.extend((render_pattern(pattern) for pattern in caution[:3]))
    if not caution:
        lines.append('- none')
    lines.extend(['Decision cues:', 'Positive cues: ' + plain_values(cues.get('positive_match_cues') or [], 10), 'Negative cues: ' + plain_values(cues.get('negative_match_cues') or [], 8), 'Author cues: liked_authors: ' + plain_values(author_cues.get('liked_authors') or [], 5) + '; caution_authors: ' + plain_values(author_cues.get('rejected_authors') or [], 5)])
    return '\n'.join(lines)

def is_specific_caution(signal: str) -> bool:
    broad = {'books', 'fiction', 'literature', 'literature and fiction', 'romance', 'fantasy', 'mystery', 'thriller', 'horror', 'science fiction', 'young adult', 'children', 'genre', 'dark', 'emotional', 'humor', 'adventure'}
    signal = signal.strip().casefold()
    return bool(signal) and signal not in broad and (len(signal.split()) >= 2 or '-' in signal)

def render_edge(row: dict[str, Any]) -> str:
    node = row['model_output']['edge_node']
    positive = node.get('positive_transition') or {}
    negative = node.get('negative_caution_transition') or {}
    update = node.get('recommendation_update') or {}
    lines = [f"[EDGE {row['from_segment']} -> {row['to_segment']}]", 'Transition label: ' + clean(node.get('transition_label')), 'Transition type: ' + clean(node.get('transition_type')), 'Positive transition:', '- analysis: ' + clean(positive.get('analysis'))]
    for field, description in (('retained', 'stable anchors'), ('emerging', 'recent/emerging positive focus'), ('fading', 'less current representation weight, not negative feedback')):
        signals = values(positive.get(field) or [], limit=5)
        lines.append(f"- {field}: {', '.join(signals) or 'none'} ({description})")
    lines.extend(['Negative caution transition:', '- analysis: ' + clean(negative.get('analysis'))])
    for field in ('retained', 'emerging', 'fading'):
        signals = [value for value in values(negative.get(field) or [], limit=5) if is_specific_caution(value)]
        lines.append(f"- specific {field} cautions: {', '.join(signals) or 'none'} (soft non-causal caution only)")
    conflicts = values(node.get('polarity_conflicts') or [], limit=4)
    lines.extend(['Polarity conflicts: ' + (', '.join(conflicts) or 'none'), 'Recommendation update from edge:', '- long-term positive anchors: ' + (', '.join(values(update.get('long_term_positive_anchors') or [], key='', limit=5)) or 'none'), '- recent positive focus: ' + (', '.join(values(update.get('recent_positive_focus') or [], key='', limit=5)) or 'none')])
    caution = [value for value in values(update.get('soft_caution_signals') or [], key='', limit=5) if is_specific_caution(value)]
    lines.extend(['- soft caution signals: ' + (', '.join(caution) or 'none'), '- do not over-penalize: ' + (', '.join(values(update.get('do_not_over_penalize') or [], key='', limit=5)) or 'none'), 'Transition summary: ' + clean(node.get('transition_summary'))])
    return '\n'.join(lines)

def render_target(profile: dict[str, Any]) -> str:
    fields = ['Target book:', 'Title: ' + clean(profile.get('title')), 'Authors: ' + plain_values(profile.get('authors') or [], 6), 'Categories: ' + plain_values(profile.get('categories') or [], 8), 'Core themes: ' + plain_values(profile.get('core_themes') or [], 10), 'Story keywords: ' + plain_values(profile.get('story_keywords') or [], 10), 'Semantic tags: ' + plain_values(profile.get('semantic_tags') or [], 12), 'Semantic terms: ' + plain_values(profile.get('semantic_signal_terms') or [], 20), 'Narrative style: ' + plain_values(profile.get('narrative_style') or [], 8), 'Emotional tone: ' + plain_values(profile.get('emotional_tone') or [], 8), 'Audience facets: ' + plain_values(profile.get('audience_preference_facets') or [], 8), 'Named entities: ' + plain_values(profile.get('named_entities') or [], 8), 'Profile: ' + clean(profile.get('recommendation_profile'))[:1200], 'Description: ' + clean(profile.get('description'))[:1200]]
    return '\n'.join(fields)

def state_evidence_titles(states: list[dict[str, Any]]) -> set[str]:
    titles = set()
    for state in states:
        candidate = state.get('selected_candidate') or {}
        analyses = (candidate.get('positive_analysis') or {}, candidate.get('negative_analysis') or {})
        for analysis in analyses:
            patterns = analysis.get('liked_patterns') or analysis.get('rejected_or_risky_patterns') or []
            for pattern in patterns:
                for title in pattern.get('evidence_titles') or []:
                    normalized = clean(title).casefold()
                    if normalized:
                        titles.add(normalized)
    return titles

def build() -> list[dict[str, Any]]:
    OUT.mkdir(parents=True, exist_ok=True)
    samples = read_jsonl(SAMPLES)
    profiles = {int(row['item_id']): row for row in read_jsonl(PROFILES)}
    states: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(EDGE_ROOT / 'selected_state_nodes.jsonl'):
        states.setdefault(str(row['sample_id']), []).append(row)
    edges = {(str(row['sample_id']), str(row['from_segment']), str(row['to_segment'])): row for row in read_jsonl(EDGE_ROOT / 'E1/transition_candidates.jsonl')}
    prompts, skipped, leakage = ([], [], [])
    segment_dist, edge_dist = (Counter(), Counter())
    for sample in samples:
        sample_id = str(sample['sample_id'])
        ordered = sorted(states.get(sample_id, []), key=lambda row: int(re.sub('\\D', '', row['segment_id'])))
        if not ordered:
            skipped.append({'sample_id': sample_id, 'reason': 'missing_c1_states'})
            continue
        target_id = int(sample['item_id'])
        target = profiles.get(target_id)
        if target is None:
            skipped.append({'sample_id': sample_id, 'reason': 'missing_target_profile'})
            continue
        chain_parts = ['Lifecycle interest chain before the target:']
        missing_edges = []
        for index, state in enumerate(ordered):
            if index:
                previous = ordered[index - 1]['segment_id']
                current = state['segment_id']
                edge = edges.get((sample_id, previous, current))
                if edge is None:
                    missing_edges.append(f'{previous}->{current}')
                else:
                    chain_parts.append(render_edge(edge))
            chain_parts.append(render_state(state))
        if missing_edges:
            skipped.append({'sample_id': sample_id, 'reason': 'missing_e1_edges', 'edges': missing_edges})
            continue
        chain_text = '\n\n'.join(chain_parts)
        target_text = render_target(target)
        user_prompt = chain_text + '\n\n' + target_text + "\n\nQuestion:\nBased on the ordered lifecycle interest chain and the target book's semantic profile, would the user have a positive interaction with the target book?\n\nAnswer:"
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': user_prompt}]
        leaks = [match.group(0).strip() for match in FORBIDDEN_FIELD_RE.finditer(user_prompt)]
        target_title = clean(target.get('title')).casefold()
        target_in_chain = bool(target_title) and target_title in state_evidence_titles(ordered)
        if leaks or target_in_chain:
            leakage.append({'sample_id': sample_id, 'field_leaks': leaks, 'target_title_in_chain': target_in_chain})
            continue
        prompts.append({'sample_id': sample_id, 'user_id': sample['user_id'], 'target_item_id': target_id, 'label': int(sample['label']), 'rating': sample.get('rating'), 'num_segments': len(ordered), 'num_edges': max(0, len(ordered) - 1), 'messages': messages, 'prompt_variant': 'book_c1e1_unrepaired_posfocused_v1', 'edge_variant': 'E1_unrepaired', 'pack_hash': hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()})
        segment_dist[len(ordered)] += 1
        edge_dist[max(0, len(ordered) - 1)] += 1
    write_jsonl(OUT / 'judge_prompts.jsonl', prompts)
    write_jsonl(OUT / 'skipped_missing_state_or_edge.jsonl', skipped)
    write_jsonl(OUT / 'leakage_failures.jsonl', leakage)
    report = {'samples': len(samples), 'built': len(prompts), 'skipped': len(skipped), 'leakage_failures': len(leakage), 'label_counts': dict(Counter((str(row['label']) for row in prompts))), 'segment_distribution': {str(key): value for key, value in sorted(segment_dist.items())}, 'edge_distribution': {str(key): value for key, value in sorted(edge_dist.items())}, 'state_policy': 'C1 selected states, unrepaired', 'edge_policy': 'E1 72B edge candidates, unrepaired', 'entropy_repair_started': False}
    write_json(OUT / 'build_report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return prompts

def run(phase: str) -> None:
    rows = read_jsonl(OUT / 'judge_prompts.jsonl')
    if phase == 'smoke':
        rows = rows[:20]
    from lifecycle_rec.teacher import judge_client as judge
    judge.ENDPOINT = JUDGE_ENDPOINT
    output = OUT / ('smoke_results.jsonl' if phase == 'smoke' else 'judge_results.jsonl')
    existing = {row['sample_id']: row for row in read_jsonl(output)} if output.exists() else {}
    pending = [row for row in rows if row['sample_id'] not in existing]
    started = time.time()
    with output.open('a', encoding='utf-8') as handle, cf.ThreadPoolExecutor(max_workers=48) as executor:
        futures = [executor.submit(judge.call_model, row, index) for index, row in enumerate(pending)]
        for index, future in enumerate(cf.as_completed(futures), 1):
            result = future.result()
            handle.write(json.dumps(result, ensure_ascii=False) + '\n')
            handle.flush()
            existing[result['sample_id']] = result
            if index % 100 == 0:
                print(json.dumps({'completed_this_run': index, 'total_complete': len(existing)}), flush=True)
    results = [existing[row['sample_id']] for row in rows if row['sample_id'] in existing]
    valid = [row for row in results if row.get('prediction') is not None]
    labels = [int(row['label']) for row in valid]
    scores = [float(row['score']) for row in valid]
    report = {'requested': len(rows), 'completed': len(results), 'valid_predictions': len(valid), 'invalid_predictions': len(results) - len(valid), 'correct': sum((row['prediction'] == row['label'] for row in valid)), 'accuracy': sum((row['prediction'] == row['label'] for row in valid)) / len(valid) if valid else None, 'auc': judge.auc(labels, scores), 'label_counts': dict(Counter((str(row['label']) for row in results))), 'prediction_counts': dict(Counter((str(row['prediction']) for row in valid))), 'score_source_counts': dict(Counter((row['score_source'] for row in valid))), 'elapsed_sec': time.time() - started, 'model': judge.MODEL, 'temperature': 0.0, 'max_tokens': 192, 'top_logprobs': 20, 'state_policy': 'C1 unrepaired', 'edge_policy': 'E1 unrepaired', 'entropy_repair_started': False}
    write_json(OUT / f'{phase}_report.json', report)
    write_jsonl(OUT / f'{phase}_passed.jsonl', [row for row in valid if row['prediction'] == row['label']])
    write_jsonl(OUT / f'{phase}_failed.jsonl', [row for row in valid if row['prediction'] != row['label']])
    print(json.dumps(report, ensure_ascii=False, indent=2))

def main() -> None:
    global SAMPLES, PROFILES, EDGE_ROOT, OUT, JUDGE_ENDPOINT
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('build', 'smoke', 'run'), required=True)
    parser.add_argument('--sample-list', type=Path, required=True)
    parser.add_argument('--canonical-profile', type=Path, required=True)
    parser.add_argument('--transition-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--judge-endpoint', type=str, required=True)
    args = parser.parse_args()
    SAMPLES = args.sample_list
    PROFILES = args.canonical_profile
    EDGE_ROOT = args.transition_dir
    OUT = args.output_dir
    JUDGE_ENDPOINT = args.judge_endpoint
    if args.phase == 'build':
        build()
    else:
        run(args.phase)
