import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

def read_jsonl(path):
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')

def norm(value):
    value = unicodedata.normalize('NFKC', str(value)).casefold()
    return re.sub('[^a-z0-9]+', '', value)

def section(text, start, end):
    tail = text.split(start, 1)
    if len(tail) != 2:
        return ''
    return tail[1].split(end, 1)[0]

def parse_list(value):
    try:
        parsed = json.loads(value)
        return [str(item) for item in parsed] if isinstance(parsed, list) else []
    except Exception:
        return []

def parse_items(text):
    starts = list(re.finditer('(?m)^- title:\\s*(.*?)\\s*$', text))
    items = []
    for index, match in enumerate(starts):
        stop = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        block = text[match.start():stop]
        fields = {key: value.strip() for key, value in re.findall('(?m)^\\s*(?:-\\s*)?([a-z_]+):\\s*(.*?)\\s*$', block)}
        title = fields.get('title', '')
        categories = parse_list(fields.get('categories', '[]'))
        terms = parse_list(fields.get('semantic_terms', '[]'))
        author = fields.get('authors', '')
        items.append({'title': title, 'terms': categories + terms, 'author': author})
    return items

def unique_map(values):
    grouped = defaultdict(list)
    for value in values:
        if value:
            grouped[norm(value)].append(value)
    return {key: vals[0] for key, vals in grouped.items() if key and len(set(vals)) == 1}

def canonical(value, mapping):
    if value == 'none':
        return 'none'
    return mapping.get(norm(value))

def repair_polarity(node, items, text, analysis_key, patterns_key, cues_key, repeated_authors, author_key):
    stats = Counter()
    title_map = unique_map((item['title'] for item in items))
    item_by_title = {item['title']: item for item in items}
    all_term_map = unique_map((term for item in items for term in item['terms']))
    analysis = node.get(analysis_key) or {}
    repaired_patterns = []
    for pattern in analysis.get(patterns_key) or []:
        titles = pattern.get('evidence_titles') or []
        title = titles[0] if len(titles) == 1 and titles[0] in text else canonical(titles[0], title_map) if len(titles) == 1 else None
        if not title:
            stats['dropped_pattern_bad_title'] += 1
            continue
        cited_item = item_by_title.get(title)
        term_map = unique_map(cited_item['terms'] if cited_item else (term for item in items for term in item['terms']))
        terms = []
        for value in pattern.get('support_terms') or []:
            if value == 'none':
                stats['dropped_none_support_term'] += 1
                continue
            fixed = value if value in text else canonical(value, term_map)
            if fixed and fixed not in terms:
                terms.append(fixed)
                stats['canonicalized_term'] += int(fixed != value)
            elif not fixed:
                stats['dropped_ungrounded_term'] += 1
        if not terms:
            stats['dropped_pattern_no_terms'] += 1
            continue
        repaired = dict(pattern)
        repaired['evidence_titles'] = [title]
        repaired['support_terms'] = terms[:4]
        stats['canonicalized_title'] += int(title != titles[0])
        repaired_patterns.append(repaired)
    analysis[patterns_key] = repaired_patterns[:2]
    node[analysis_key] = analysis
    cues = node.setdefault('decision_cues', {})
    repaired_cues = []
    for value in cues.get(cues_key) or []:
        if value == 'none':
            continue
        fixed = value if value in text else canonical(value, all_term_map)
        if fixed and fixed not in repaired_cues:
            repaired_cues.append(fixed)
            stats['canonicalized_cue'] += int(fixed != value)
        elif not fixed:
            stats['dropped_ungrounded_cue'] += 1
    cues[cues_key] = repaired_cues[:4] or ['none']
    author_map = unique_map(repeated_authors)
    repeated_exact = set(repeated_authors)
    author_cues = cues.setdefault('author_cues', {})
    repaired_authors = []
    for value in author_cues.get(author_key) or []:
        if value == 'none':
            continue
        fixed = value if value in repeated_exact else canonical(value, author_map)
        if fixed and fixed not in repaired_authors:
            repaired_authors.append(fixed)
            stats['canonicalized_author'] += int(fixed != value)
        elif not fixed:
            stats['dropped_ungrounded_author'] += 1
    author_cues[author_key] = repaired_authors or ['none']
    return stats

def repeated_authors(items):
    counts = Counter((norm(item['author']) for item in items if item['author']))
    return [item['author'] for item in items if item['author'] and counts[norm(item['author'])] >= 2]

def repair_output(raw_output, user):
    output = json.loads(raw_output)
    node = output['interest_state_node']
    positive_text = section(user, '[POSITIVE INTERACTIONS]', '[NEGATIVE INTERACTIONS]')
    negative_text = section(user, '[NEGATIVE INTERACTIONS]', '[POSITIVE COMMON TERMS]')
    positive_items = parse_items(positive_text)
    negative_items = parse_items(negative_text)
    stats = Counter()
    stats.update(repair_polarity(node, positive_items, positive_text, 'positive_analysis', 'liked_patterns', 'positive_match_cues', repeated_authors(positive_items), 'liked_authors'))
    stats.update(repair_polarity(node, negative_items, negative_text, 'negative_analysis', 'rejected_or_risky_patterns', 'negative_match_cues', repeated_authors(negative_items), 'rejected_authors'))
    for text, analysis_key, pattern_key, cue_key in ((positive_text, 'positive_analysis', 'liked_patterns', 'positive_match_cues'), (negative_text, 'negative_analysis', 'rejected_or_risky_patterns', 'negative_match_cues')):
        analysis = node[analysis_key]
        grounded = []
        for pattern in analysis[pattern_key]:
            terms = [term for term in pattern['support_terms'] if term in text]
            stats['dropped_nonexact_term'] += len(pattern['support_terms']) - len(terms)
            if not terms:
                stats['dropped_pattern_no_exact_terms'] += 1
                continue
            pattern['support_terms'] = terms[:4]
            grounded.append(pattern)
        analysis[pattern_key] = grounded[:2]
        cues = node['decision_cues'].get(cue_key) or []
        exact_cues = [cue for cue in cues if cue == 'none' or cue in text]
        stats['dropped_nonexact_cue'] += len(cues) - len(exact_cues)
        node['decision_cues'][cue_key] = exact_cues[:4] or ['none']
    return (json.dumps(output, ensure_ascii=False, separators=(',', ':')), stats)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus-dir', type=Path, required=True)
    parser.add_argument('--input-model-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    sources = {}
    for split in ('valid', 'test'):
        for row in read_jsonl(args.corpus_dir / f'segment_interest_node_sft_{split}.jsonl'):
            sources[row['record_id']] = row['messages'][1]['content']
    totals = Counter()
    records = 0
    changed = 0
    for split in ('valid', 'test'):
        rows = []
        source_path = args.input_model_dir / f'generation_eval_{split}_outputs.jsonl'
        for row in read_jsonl(source_path):
            records += 1
            repaired, stats = repair_output(row['raw_output'], sources[row['record_id']])
            if repaired != row['raw_output']:
                changed += 1
                row = dict(row)
                row['raw_output_original'] = row['raw_output']
                row['raw_output'] = repaired
                row['grounding_repair'] = dict(stats)
            totals.update(stats)
            rows.append(row)
        write_jsonl(args.output_dir / f'generation_eval_{split}_outputs.jsonl', rows)
    report = {'records': records, 'changed_records': changed, 'actions': dict(totals)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'grounding_repair_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
if __name__ == '__main__':
    main()
