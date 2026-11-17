import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path

def normalize_text(value):
    value = '' if value is None else str(value)
    value = value.lower().strip()
    value = value.replace('&', 'and')
    value = value.translate(str.maketrans('', '', string.punctuation))
    value = re.sub('\\s+', ' ', value).strip()
    return value

def iter_terms(value):
    if value is None:
        return
    if isinstance(value, list):
        for item in value:
            yield item
    else:
        yield value

def merge_terms(tags, semantic_keywords):
    out = []
    seen = set()
    for values in (tags, semantic_keywords):
        for source in iter_terms(values):
            if not isinstance(source, str):
                source = str(source)
            term = re.sub('\\s+', ' ', source.strip())
            key = normalize_text(term)
            if not term or not key or key in seen:
                continue
            seen.add(key)
            out.append(term)
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    empty_term_ids = []
    with input_path.open(encoding='utf-8') as src, output_path.open('w', encoding='utf-8') as out:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            stats['rows'] += 1
            llm_profile = dict(row.get('llm_profile') or {})
            tags = llm_profile.pop('tags', [])
            semantic_keywords = llm_profile.pop('semantic_keywords', [])
            terms = merge_terms(tags, semantic_keywords)
            if terms:
                llm_profile['semantic_terms_for_embedding'] = terms
            else:
                empty_term_ids.append(row['item_id'])
            tag_count = len(list(iter_terms(tags) or []))
            keyword_count = len(list(iter_terms(semantic_keywords) or []))
            stats['tag_terms_input'] += tag_count
            stats['semantic_keyword_terms_input'] += keyword_count
            stats['semantic_terms_output'] += len(terms)
            stats['terms_removed_by_dedupe'] += tag_count + keyword_count - len(terms)
            row['llm_profile'] = llm_profile
            out.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    report = {'input': args.input, 'output': args.output, 'rows': stats['rows'], 'tag_terms_input': stats['tag_terms_input'], 'semantic_keyword_terms_input': stats['semantic_keyword_terms_input'], 'semantic_terms_output': stats['semantic_terms_output'], 'terms_removed_by_dedupe': stats['terms_removed_by_dedupe'], 'empty_semantic_terms_for_embedding_count': len(empty_term_ids), 'empty_semantic_terms_for_embedding_ids': empty_term_ids[:50], 'replaced_fields': ['llm_profile.tags', 'llm_profile.semantic_keywords'], 'new_field': 'llm_profile.semantic_terms_for_embedding'}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
