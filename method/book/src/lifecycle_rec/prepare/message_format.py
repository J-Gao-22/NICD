import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer
FOOTER = '\n\n[OUTPUT CHECKLIST]\nReturn a compact JSON candidate for the requested lens.\nFor each polarity: choose at most ONE pattern, ONE or TWO representative book titles, and TWO or THREE exact semantic phrases from those cited books. Do not try to cover all books. Do not use COMMON TERMS unless the same complete phrase occurs in a cited item\'s categories or semantic_terms.\nEach cited book must contribute at least one support phrase. Match cues copy support_terms. Do not invent synonyms, shorten titles, or output edition numbers and generic title words. Use fewer terms when necessary.\nIn prose, write only local positive signal or tentative local caution; never write strong preference, rejected, rejection, dislike, avoidance, rated poorly, or low-rated. A 4-star rating is not poor. Never infer why the rating occurred. Use ["none"] author cues unless the same author appears twice in that polarity.\nAn absent polarity has [] patterns and ["none"] cues. A single interaction supports only a single-book observation. For C3, use insufficient_distinct_evidence if no distinct specific signal is grounded.\nDo not put a phrase that occurs in BOTH polarities into either match_cues list. Shared phrases describe context but do not distinguish feedback; if nothing distinguishes a polarity, use ["none"] cues and explicitly state no distinct semantic contrast is supported.\nOutput only the JSON object, without markdown fences.'
REPLACEMENTS = {'Each polarity needs at most ONE compact pattern. Use 1-3 exact evidence_titles.': 'Each polarity permits at most TWO compact patterns. Each pattern cites exactly ONE verbatim book title.', "Match cues must be copied from that polarity's support_terms.": 'Match cues must be exact complete terms from any supplied book of the same polarity; they need not repeat the selected pattern support_terms.', 'For each polarity: choose at most ONE pattern, ONE or TWO representative book titles, and TWO or THREE exact semantic phrases from those cited books.': 'For each polarity: choose at most TWO compact patterns, each citing exactly ONE book and 1-4 exact complete phrases from that book.', 'Match cues copy support_terms.': 'Match cues may use exact complete terms from any input book of the same polarity, excluding shared terms.'}
SYSTEM_SUFFIX = '\nUse local positive signal, not strong/clear/persistent positive preference. No claim of preference intensity or permanence is justified by these observations. Remove equivalent duplicate phrases before returning JSON.\n'
STANDALONE_NAMES = {"alzheimer's", "parkinson's", "crohn's", "mcdonald's", "macy's", "rubik's"}

def read_jsonl(path):
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)

def write_jsonl(path, rows):
    with path.open('w', encoding='utf-8', newline='\n') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')

def encoded_len(value):
    if isinstance(value, dict):
        value = value.get('input_ids', value)
    elif hasattr(value, 'input_ids'):
        value = value.input_ids
    if value and isinstance(value[0], (list, tuple)):
        return len(value[0])
    return len(value)

def clean_pack(pack):

    def replace(match):
        terms = json.loads(match.group(2))
        keep = []
        for term in terms:
            normalized = term.casefold().replace('’', "'")
            isolated = re.fullmatch("[\\w]+['\\u2019](?:s|re|ve|m|d|ll|t)", term.strip(), re.I)
            if normalized not in STANDALONE_NAMES and isolated:
                continue
            keep.append(term)
        return match.group(1) + json.dumps(keep, ensure_ascii=False)
    before = re.findall('(?m)^\\s*semantic_terms: (\\[[^\\n]*\\])', pack)
    pack = re.sub('(^\\s*semantic_terms: )(\\[[^\\n]*\\])', replace, pack, flags=re.M)
    after = re.findall('(?m)^\\s*semantic_terms: (\\[[^\\n]*\\])', pack)
    removed = set()
    for old, new in zip(before, after):
        removed.update(json.loads(old))
        removed.difference_update(json.loads(new))
    normalized = {re.sub('[^a-z0-9]+', ' ', term.lower()).strip() for term in removed}
    lines = [line for line in pack.splitlines() if not (line.startswith('- ') and ' | support=' in line and (line[2:].split(' | support=')[0] in normalized))]
    return '\n'.join(lines)

def revise_row(row, tokenizer):
    row = json.loads(json.dumps(row))
    row['messages'][1]['content'] += FOOTER
    for message in row['messages']:
        for old, new in REPLACEMENTS.items():
            message['content'] = message['content'].replace(old, new)
    row['messages'][0]['content'] += SYSTEM_SUFFIX
    row['messages'][1]['content'] = clean_pack(row['messages'][1]['content'])
    system = row['messages'][0]['content'].replace('9. Author cues require the exact author in at least two books of the same polarity. Otherwise ["none"]. All director/actor lists must be ["none"].', '9. Author cues require the exact author in at least two books of the same polarity. Otherwise ["none"]. Do not output director or actor fields for books.')
    row['messages'][0]['content'] = '\n'.join((line for line in system.splitlines() if '"director_actor_cues"' not in line))
    row['pack_hash'] = hashlib.sha256(row['messages'][1]['content'].encode()).hexdigest()
    token_ids = tokenizer.apply_chat_template(row['messages'], tokenize=True, add_generation_prompt=True)
    row['prompt_tokens'] = encoded_len(token_ids)
    if row['prompt_tokens'] + 832 + 700 > 5120:
        raise RuntimeError(f"context overflow: {row['record_id']} {row['prompt_tokens']}")
    if '"director_actor_cues"' in row['messages'][0]['content']:
        raise RuntimeError(f"MovieLens-only schema remains: {row['record_id']}")
    return row

def load_by_record(path):
    return {row['record_id']: row for row in read_jsonl(path)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--dest', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--append-dest', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    args = parser.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    args.append_dest.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    reference_samples = {row['sample_id'] for row in read_jsonl(args.reference / 'teacher_sample_list.jsonl')}
    all_samples = list(read_jsonl(args.source / 'teacher_sample_list.jsonl'))
    append_samples = [row for row in all_samples if row['sample_id'] not in reference_samples]
    if len(all_samples) != 6000 or len(reference_samples) != 2000 or len(append_samples) != 4000:
        raise RuntimeError((len(all_samples), len(reference_samples), len(append_samples)))
    final_rows = [revise_row(row, tokenizer) for row in read_jsonl(args.source / 'state_messages.jsonl')]
    write_jsonl(args.dest / 'state_messages.jsonl', final_rows)
    write_jsonl(args.dest / 'state_messages_smoke20.jsonl', final_rows[:60])
    append_rows = [row for row in final_rows if row['sample_id'] not in reference_samples]
    write_jsonl(args.append_dest / 'state_messages.jsonl', append_rows)
    write_jsonl(args.append_dest / 'state_messages_smoke20.jsonl', append_rows[:60])
    reference = load_by_record(args.reference / 'state_messages.jsonl')
    prefix = {row['record_id']: row for row in final_rows if row['sample_id'] in reference_samples}
    if prefix != reference:
        missing = sorted(set(reference) - set(prefix))[:5]
        extra = sorted(set(prefix) - set(reference))[:5]
        changed = sorted((key for key in set(prefix) & set(reference) if prefix[key] != reference[key]))[:5]
        raise RuntimeError({'prefix_not_identical': True, 'missing': missing, 'extra': extra, 'changed': changed})
    all_segments = list(read_jsonl(args.source / 'state_segments.jsonl'))
    append_segments = [row for row in all_segments if row['sample_id'] not in reference_samples]
    write_jsonl(args.dest / 'teacher_sample_list.jsonl', all_samples)
    write_jsonl(args.dest / 'state_segments.jsonl', all_segments)
    write_jsonl(args.append_dest / 'teacher_sample_list.jsonl', append_samples)
    write_jsonl(args.append_dest / 'state_segments.jsonl', append_segments)
    ids = [row['record_id'] for row in append_rows]
    lens = Counter((row['candidate_id'] for row in append_rows))
    report = {'version': 'amazon_book_state_messages_v3_15_scaled_append_only', 'all_samples': len(all_samples), 'reference_samples': len(reference_samples), 'append_samples': len(append_samples), 'all_segments': len(all_segments), 'append_segments': len(append_segments), 'all_messages': len(final_rows), 'append_messages': len(append_rows), 'append_lens': dict(lens), 'unique_append_record_ids': len(set(ids)), 'overlap_with_reference_record_ids': len(set(ids) & set(reference)), 'reference_prefix_exact': True, 'prompt_tokens_min': min((row['prompt_tokens'] for row in append_rows)), 'prompt_tokens_max': max((row['prompt_tokens'] for row in append_rows)), 'messages_sha256': hashlib.sha256((args.append_dest / 'state_messages.jsonl').read_bytes()).hexdigest()}
    if report['unique_append_record_ids'] != len(ids) or report['overlap_with_reference_record_ids']:
        raise RuntimeError(report)
    for directory in (args.dest, args.append_dest):
        shutil.copy2(args.source / 'leakage_audit.json', directory / 'base_leakage_audit.json')
        (directory / 'scaled_build_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
if __name__ == '__main__':
    main()
