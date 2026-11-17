import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer
from lifecycle_rec.prepare import evidence as book_compat
from lifecycle_rec.prepare.message_format import FOOTER, REPLACEMENTS, clean_pack
base = book_compat.base

def read_jsonl(path):
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)

def first_row(path):
    return next(read_jsonl(path))

def finalize_user(pack):
    content = pack + FOOTER
    for old, new in REPLACEMENTS.items():
        content = content.replace(old, new)
    return clean_pack(content)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all-interactions', type=Path, required=True)
    parser.add_argument('--canonical-profile', type=Path, required=True)
    parser.add_argument('--embedding-dir', type=Path, required=True)
    parser.add_argument('--sft-reference', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--limit-per-split', type=int, default=0)
    parser.add_argument('--max-model-len', type=int, default=4096)
    parser.add_argument('--reserved-output-tokens', type=int, default=768)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    items = {int(row['item_id']): row for row in read_jsonl(args.canonical_profile)}
    embeddings = base.load_embeddings(args.embedding_dir)
    missing = sorted(set(items) - set(embeddings))
    if missing:
        raise RuntimeError(f'missing embeddings: {len(missing)} {missing[:10]}')
    reference = first_row(args.sft_reference)
    system_prompt = reference['messages'][0]['content']
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    handles = {split: (args.output_dir / f'{split}_state_segment_messages.jsonl').open('w', encoding='utf-8', newline='\n') for split in ('train', 'valid', 'test')}
    counts = Counter()
    segment_dist = Counter()
    prompt_lengths = []
    errors = []
    try:
        for row in read_jsonl(args.all_interactions):
            split = str(row.get('split'))
            if split not in handles or not row.get('history_item_ids'):
                continue
            if args.limit_per_split and counts[f'samples_{split}'] >= args.limit_per_split:
                continue
            target_id = int(row['item_id'])
            history_ids = [int(value) for value in row['history_item_ids']]
            target = items[target_id]
            if target_id in history_ids:
                errors.append({'sample_id': row['sample_id'], 'type': 'target_item_id_in_history'})
                continue
            segments, scores = base.build_segments(row, embeddings)
            counts[f'samples_{split}'] += 1
            segment_dist[f'{split}:{len(segments)}'] += 1
            target_title = base.normalize_title(target.get('title'))
            for segment in segments:
                start, end = (segment['start'], segment['end'])
                segment_ids = history_ids[start:end]
                pack = finalize_user(base.render_pack(row, segment, items, base.clean(target.get('title'))))
                pack_hash = hashlib.sha256(pack.encode('utf-8')).hexdigest()
                local_errors = []
                if target_id in segment_ids:
                    local_errors.append('target_item_id_in_segment')
                if target_title and len(target_title) >= 8 and (target_title in base.normalize_title(pack)):
                    local_errors.append('target_title_phrase_in_pack')
                if base.clean(target.get('asin')) and base.clean(target.get('asin')) in pack:
                    local_errors.append('target_asin_in_pack')
                if any((marker in pack for marker in ('target_iid:', 'target_item_id:', 'target_label:', 'sample_id:'))):
                    local_errors.append('target_control_marker_in_pack')
                messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': pack}]
                prompt_tokens = base.encoded_token_count(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
                prompt_lengths.append(prompt_tokens)
                if prompt_tokens + args.reserved_output_tokens > args.max_model_len:
                    local_errors.append('context_length_overflow')
                if local_errors:
                    errors.append({'sample_id': row['sample_id'], 'segment_id': segment['segment_id'], 'types': local_errors, 'prompt_tokens': prompt_tokens})
                    continue
                record = {'record_id': f"{row['sample_id']}::{segment['segment_id']}::C1", 'sample_id': row['sample_id'], 'split': split, 'row_index': int(row['row_index']), 'user_id': int(row['user_id']), 'item_id': target_id, 'source_label': int(row['label']), 'rating': float(row['rating']), 'timestamp': int(row['timestamp']), 'history_recent30_len': len(history_ids), 'num_segments': len(segments), 'segment_id': segment['segment_id'], 'segment_position': segment['position'], 'segment_start': start, 'segment_end': end, 'segment_item_ids': segment_ids, 'boundary_scores': scores, 'candidate_id': 'C1', 'pack_hash': pack_hash, 'prompt_tokens': prompt_tokens, 'messages': messages}
                handles[split].write(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n')
                counts[f'segments_{split}'] += 1
    finally:
        for handle in handles.values():
            handle.close()
    report = {'version': 'amazon_book_full_state_messages_v1', 'counts': dict(counts), 'segment_distribution': dict(segment_dist), 'prompt_tokens': {'min': min(prompt_lengths) if prompt_lengths else None, 'max': max(prompt_lengths) if prompt_lengths else None}, 'hard_error_count': len(errors), 'hard_error_preview': errors[:100], 'limit_per_split': args.limit_per_split, 'system_prompt_source': str(args.sft_reference)}
    (args.output_dir / 'build_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)
if __name__ == '__main__':
    main()
