from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
CHAIN_SYSTEM = "You are a book recommendation classifier.\n\nYou will receive a user's ordered lifecycle interest chain before a target book and the target book's semantic profile.\nThe lifecycle chain is chronological: S1 is earlier, S2 is later, and S3 is the most recent when present.\nEach state summarizes one segment of prior user interactions:\n- positive analysis means patterns observed in books with positive feedback in that segment.\n- negative analysis means tentative, non-causal caution patterns observed in books with below-5 ratings in that segment.\n- decision cues summarize positive cues, negative cues, and author cues that may help compare the target book.\n\nDecision policy:\n1. First identify the strongest positive preference matches between the target and the lifecycle chain.\n2. Then check whether the target matches any specific caution boundary.\n3. Treat negative/caution boundaries as soft non-causal risk signals, not hard rejection rules.\n4. Do not answer No only because the target shares a broad category or common term with a caution signal.\n5. If the target strongly matches recent or repeated positive preference cues and only weakly overlaps with caution signals, prefer Yes.\n6. Answer No when specific caution matches are stronger than positive preference matches, or when the target lacks meaningful positive preference matches.\n\nOutput format:\nAnswer with exactly one token: Yes or No.\nDo not output a reason, JSON, markdown, or analysis."
PLAIN_SYSTEM = 'You are a book recommendation classifier. Answer with exactly one token: Yes or No.'

def read_jsonl(path):
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)

def clean(value):
    return re.sub('\\s+', ' ', str(value or '')).strip()

def words(value, maximum):
    return ' '.join(clean(value).split()[:maximum])

def terms(values, maximum):
    seen, result = (set(), [])
    for value in values or []:
        item = clean(value)
        if item and item.casefold() != 'none' and (item.casefold() not in seen):
            seen.add(item.casefold())
            result.append(item)
    return result[:maximum]

def render_target(item):
    lines = ['title: ' + clean(item.get('title'))]
    for key, name, limit in (('authors', 'authors', 4), ('categories', 'categories', 5)):
        values = terms(item.get(key), limit)
        if values:
            lines.append(name + ': ' + ', '.join(values))
    profile = words(item.get('recommendation_profile') or item.get('description'), 95)
    if profile:
        lines.append('profile: ' + profile)
    for key, name, limit in (('semantic_signal_terms', 'semantic_tags', 10), ('core_themes', 'themes', 5), ('story_keywords', 'story_keywords', 5), ('narrative_style', 'style', 3), ('emotional_tone', 'tone', 3), ('audience_preference_facets', 'audience', 3)):
        values = terms(item.get(key), limit)
        if values:
            lines.append(name + ': ' + ', '.join(values))
    return '\n'.join(lines)

def render_history(row, items):
    ids = row.get('history_item_ids') or []
    ratings = row.get('history_ratings') or []
    if not ids:
        return '- No substantial prior item history is available.'
    result = []
    for item_id, rating in list(zip(ids, ratings))[-10:]:
        item = items[int(item_id)]
        parts = [clean(item.get('title'))]
        categories = terms(item.get('categories'), 3)
        if categories:
            parts.append('categories=' + ', '.join(categories))
        semantic = terms(item.get('semantic_signal_terms'), 6)
        if semantic:
            parts.append('terms=' + ', '.join(semantic))
        profile = words(item.get('recommendation_profile') or item.get('description'), 25)
        if profile:
            parts.append('profile=' + profile)
        parts.append('feedback=' + ('positive' if float(rating) >= 5 else 'below-5'))
        result.append('- ' + ' | '.join(parts))
    return '\n'.join(result)

def render_state(state_row):
    node = state_row['model_output']['interest_state_node']
    lines = [f"[STATE {state_row['segment_id']}]", 'State label: ' + clean(node.get('state_label')), 'State summary: ' + words(node.get('state_summary'), 60), 'Positive preference signals:']
    liked = (node.get('positive_analysis') or {}).get('liked_patterns') or []
    if not liked:
        lines.append('- none')
    for pattern in liked[:2]:
        extras = []
        support = terms(pattern.get('support_terms'), 4)
        evidence = terms(pattern.get('evidence_titles'), 1)
        if support:
            extras.append(', '.join(support))
        if evidence:
            extras.append('evidence=' + ', '.join(evidence))
        lines.append('- ' + clean(pattern.get('pattern')) + (' (' + ' | '.join(extras) + ')' if extras else ''))
    lines.append('Negative / caution signals:')
    risky = (node.get('negative_analysis') or {}).get('rejected_or_risky_patterns') or []
    if not risky:
        lines.append('- none')
    for pattern in risky[:2]:
        support = terms(pattern.get('support_terms'), 4)
        evidence = terms(pattern.get('evidence_titles'), 1)
        extras = []
        if support:
            extras.append('terms=' + ', '.join(support))
        if evidence:
            extras.append('evidence=' + ', '.join(evidence))
        extras.append('caution_strength=' + ('medium' if len(support) >= 3 else 'weak'))
        extras.append('caution_specificity=' + ('concrete_style_or_theme' if any((' ' in x for x in support)) else 'semantic_pattern'))
        lines.append('- ' + clean(pattern.get('pattern')) + ' | ' + ' | '.join(extras))
    cues = node.get('decision_cues') or {}
    lines.append('Decision cues:')
    lines.append('Positive cues: ' + (', '.join(terms(cues.get('positive_match_cues'), 4)) or 'none'))
    lines.append('Negative cues: ' + (', '.join(terms(cues.get('negative_match_cues'), 4)) or 'none'))
    authors = cues.get('author_cues') or {}
    liked_authors = terms(authors.get('liked_authors'), 2)
    rejected_authors = terms(authors.get('rejected_authors'), 2)
    author_parts = []
    if liked_authors:
        author_parts.append('Liked authors: ' + ', '.join(liked_authors))
    if rejected_authors:
        author_parts.append('Risky authors: ' + ', '.join(rejected_authors))
    lines.append('Author cues: ' + ('; '.join(author_parts) or 'none'))
    return '\n'.join(lines)

def make_rows(row, items, states):
    split = row['split']
    sample_id = row['sample_id']
    target = items[int(row['item_id'])]
    target_text = render_target(target)
    plain = {'record_id': f'{split}::plain::{sample_id}', 'sample_id': sample_id, 'row_index': int(row['row_index']), 'uid': int(row['thinkrec_uid']), 'iid': int(row['thinkrec_iid']), 'label': int(row['label']), 'split': split, 'input_type': 'plain', 'system': PLAIN_SYSTEM, 'user': 'User history:\n' + render_history(row, items) + '\n\nTarget book:\n' + target_text + '\n\nWould the user enjoy the target book?'}
    if not states:
        return (plain, None, 'no_state')
    expected = states[0]['num_segments']
    if len(states) != expected or len({x['segment_id'] for x in states}) != expected:
        return (plain, None, 'incomplete_state')
    states = sorted(states, key=lambda x: int(str(x['segment_id'])[1:]))
    chain_text = '\n\n'.join((render_state(x) for x in states))
    title = clean(target.get('title'))
    if len(title) >= 15 and title.casefold() in chain_text.casefold():
        return (plain, None, 'target_title_in_chain')
    if re.search('"?(target_item_id|target_label|gold_item_id|is_gold)"?\\s*:', chain_text, flags=re.I):
        return (plain, None, 'target_marker_in_chain')
    target_section = 'Target book:\n' + target_text + '\n\nWould the user enjoy the target book?'
    user = 'Lifecycle interest chain before the target:\n\n' + chain_text + '\n\n' + target_section + "\n\nQuestion:\nBased on the ordered lifecycle interest chain and the target book's semantic profile, would the user have a positive interaction with the target book?\n\nAnswer:"
    chain = {**{key: plain[key] for key in ('sample_id', 'row_index', 'uid', 'iid', 'label', 'split')}, 'record_id': f'{split}::state::{sample_id}', 'input_type': 'chain', 'system': CHAIN_SYSTEM, 'user': user, 'chain_valid': True, 'chain_num_states': len(states), 'chain_num_edges': 0, 'chain_content_hash': hashlib.sha256(chain_text.encode()).hexdigest()}
    return (plain, chain, None)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all-interactions', type=Path, required=True)
    parser.add_argument('--canonical-profile', type=Path, required=True)
    parser.add_argument('--state-dirs', type=Path, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--limit-per-split', type=int, default=100)
    parser.add_argument('--tokenizer', type=Path)
    parser.add_argument('--gzip', action='store_true')
    parser.add_argument('--audit-token-sample-rate', type=float, default=1.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    items = {int(x['item_id']): x for x in read_jsonl(args.canonical_profile)}
    selected = defaultdict(list)
    for row in read_jsonl(args.all_interactions):
        if args.limit_per_split and len(selected[row['split']]) >= args.limit_per_split:
            continue
        selected[row['split']].append(row)
    wanted = {row['sample_id'] for rows in selected.values() for row in rows}
    states = defaultdict(list)
    for directory in args.state_dirs:
        for split in ('train', 'valid', 'test'):
            for row in read_jsonl(directory / f'{split}_state_nodes.jsonl'):
                if row['sample_id'] in wanted:
                    states[row['sample_id']].append(row)
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), trust_remote_code=True)
    report = {}
    for split, rows in selected.items():
        count = Counter()
        lengths = defaultdict(list)
        suffix = '.jsonl.gz' if args.gzip else '.jsonl'
        paths = {kind: args.output_dir / f'{split}_{kind}_yesno{suffix}' for kind in ('plain', 'chain')}
        open_file = gzip.open if args.gzip else open
        rng = random.Random(20260921)
        with open_file(paths['plain'], 'wt', encoding='utf-8') as plain_file, open_file(paths['chain'], 'wt', encoding='utf-8') as chain_file:
            for row in rows:
                plain, chain, reason = make_rows(row, items, states.get(row['sample_id'], []))
                plain_file.write(json.dumps(plain, ensure_ascii=False, separators=(',', ':')) + '\n')
                count['plain'] += 1
                if chain:
                    chain_file.write(json.dumps(chain, ensure_ascii=False, separators=(',', ':')) + '\n')
                    count['chain'] += 1
                    count[f"states_{chain['chain_num_states']}"] += 1
                else:
                    count[f'excluded_{reason}'] += 1
                if tokenizer and rng.random() < args.audit_token_sample_rate:
                    for kind, record in (('plain', plain), ('chain', chain)):
                        if record:
                            prompt = tokenizer.apply_chat_template([{'role': 'system', 'content': record['system']}, {'role': 'user', 'content': record['user']}], tokenize=False, add_generation_prompt=True)
                            length = len(tokenizer.encode(prompt, add_special_tokens=False))
                            lengths[kind].append(length)
                            count[f'{kind}_over2048'] += int(length > 2048)
        report[split] = {'counts': dict(count), 'token_lengths': {kind: {'min': min(vals), 'median': sorted(vals)[len(vals) // 2], 'max': max(vals)} for kind, vals in lengths.items() if vals}, 'bytes': {kind: path.stat().st_size for kind, path in paths.items()}}
    (args.output_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
