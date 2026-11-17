from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List
LIST_KEYS = ['surface_genres', 'core_themes', 'narrative_style', 'emotional_tone', 'story_keywords', 'audience_preference_facets', 'semantic_tags', 'named_entities', 'evidence_used']
SCHEMA_KEYS = LIST_KEYS[:-1] + ['recommendation_profile', 'evidence_used', 'profile_confidence']
ALLOWED_EVIDENCE = {'overview', 'genres', 'title', 'year', 'director', 'actors', 'writer', 'production_countries', 'original_language'}
BANNED_GENERIC_PHRASES = {'movie', 'film', 'story', 'popular', 'classic', 'entertaining', 'american film', 'english-language film', 'high-rated', 'high rated', 'family fun', 'childhood wonder', 'fantasy escapade', 'adventure-seeking', 'relatable characters', 'humorous situations'}
GENERIC_TAGS = {'adventure', 'fantasy', 'comedy', 'romance', 'animation', "children's", 'drama', 'tense', 'funny', 'warm', 'growth', 'trapped', 'family-friendly', 'family fun', 'childhood wonder', 'fantasy escapade', 'adventure-seeking', 'child-friendly', 'family entertainment', 'relatable characters', 'humorous situations'}
NON_ENTITY_TERMS = {'rhinoceros', 'rhinoceroses', 'monkey', 'monkeys', 'creature', 'creatures', 'divorcée', 'divorc', 'divorcee', 'newcomer', 'shop', 'restaurant', 'room', 'bedroom', 'toys', 'toy', 'game', 'board', 'adult', 'siblings', 'neighbor', 'neighbors', 'wedding'}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt-jsonl', type=str, required=True)
    parser.add_argument('--output-jsonl', type=str, required=True)
    parser.add_argument('--final-jsonl', type=str, required=True)
    parser.add_argument('--model-dir', type=str, default='models/Qwen2.5-7B-Instruct')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--torch-dtype', type=str, default='bfloat16', choices=['auto', 'float16', 'bfloat16', 'float32'])
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--max-input-tokens', type=int, default=4096)
    parser.add_argument('--max-new-tokens', type=int, default=384)
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()

def iter_jsonl(path: Path, limit: int=0) -> Iterable[Dict[str, Any]]:
    emitted = 0
    with path.open('r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            emitted += 1
            if int(limit) > 0 and emitted >= int(limit):
                break

def batches(rows: Iterable[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    batch: List[Dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def dtype_from_name(name: str):
    import torch
    if name == 'float16':
        return torch.float16
    if name == 'bfloat16':
        return torch.bfloat16
    if name == 'float32':
        return torch.float32
    return 'auto'

def existing_item_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    seen: set[int] = set()
    with path.open('r', encoding='utf-8') as file:
        for line in file:
            try:
                row = json.loads(line)
                seen.add(int(row['item_id']))
            except Exception:
                continue
    return seen

def extract_json(text: str) -> Dict[str, Any]:
    chunks: List[str] = []
    fenced = re.search('```(?:json)?\\s*(\\{.*?\\})\\s*```', text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        chunks.append(fenced.group(1))
    first = text.find('{')
    last = text.rfind('}')
    if first >= 0 and last > first:
        chunks.append(text[first:last + 1])
    for chunk in chunks:
        try:
            parsed = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}

def clean_list(value: Any, limit: int=12) -> List[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out: List[str] = []
    seen = set()
    for item in value:
        text = re.sub('\\s+', ' ', str(item).strip())
        if not text:
            continue
        lowered = text.lower()
        if any((banned in lowered for banned in BANNED_GENERIC_PHRASES)):
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text[:120])
        if len(out) >= limit:
            break
    return out

def _flatten_generated_text(parsed: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key, value in parsed.items():
        if key in {'evidence_used', 'profile_confidence'}:
            continue
        if isinstance(value, list):
            parts.extend((str(item) for item in value))
        elif isinstance(value, str):
            parts.append(value)
    return ' '.join(parts).lower()

def normalize_evidence_used(row: Dict[str, Any], parsed: Dict[str, Any], values: List[str]) -> List[str]:
    payload = row.get('payload', {}) if isinstance(row.get('payload'), dict) else {}
    generated = _flatten_generated_text(parsed)
    kept: List[str] = []
    for field in values:
        if field in {'overview', 'genres'}:
            kept.append(field)
            continue
        if field not in {'title'}:
            continue
        source = payload.get(field)
        source_values = source if isinstance(source, list) else [source]
        normalized_sources = [str(item).strip().lower() for item in source_values if str(item or '').strip()]
        if field == 'title':
            title = normalized_sources[0] if normalized_sources else ''
            title_core = re.sub('\\s*\\(\\d{4}\\)\\s*$', '', title).strip()
            if title_core and title_core in generated:
                kept.append(field)
            continue
    out: List[str] = []
    seen = set()
    for field in kept:
        if field in ALLOWED_EVIDENCE and field not in seen:
            seen.add(field)
            out.append(field)
    return out

def _evidence_text(row: Dict[str, Any]) -> str:
    payload = row.get('payload', {}) if isinstance(row.get('payload'), dict) else {}
    parts: List[str] = []
    for field in ['overview', 'genres', 'title', 'year']:
        value = payload.get(field)
        if isinstance(value, list):
            parts.extend((str(item) for item in value))
        elif value is not None:
            parts.append(str(value))
    return ' '.join(parts).lower()

def supported_by_evidence(row: Dict[str, Any], phrase: str) -> bool:
    text = _evidence_text(row)
    lowered = phrase.lower()
    if any((term in lowered for term in ['magical', 'magic', 'enchanted'])):
        if not any((term in text for term in ['magic', 'magical', 'enchanted', 'fantasy', 'supernatural'])):
            return False
    if 'domestic conflict' in lowered:
        if not any((term in text for term in ['family', 'wedding', 'neighbor', 'next-door', 'marriage', 'home', 'household'])):
            return False
    if any((term in lowered for term in ['late-life', 'aging', 'elderly', 'retirement'])):
        if not any((term in text for term in ['old', 'older', 'aging', 'elderly', 'retirement', 'late-life'])):
            return False
    if 'neighbor' in lowered or 'neighborhood' in lowered:
        if not any((term in text for term in ['neighbor', 'neighborhood', 'next-door'])):
            return False
    return True

def _phrase_tokens(text: str) -> set[str]:
    return set(re.findall("[a-z0-9']+", text.lower()))

def dedupe_phrases(values: List[str], limit: int=12) -> List[str]:
    out: List[str] = []
    token_sets: List[set[str]] = []
    for value in values:
        tokens = _phrase_tokens(value)
        if not tokens:
            continue
        skip = False
        replace_at = -1
        for idx, existing in enumerate(token_sets):
            if tokens <= existing:
                skip = True
                break
            if existing < tokens:
                replace_at = idx
                break
        if skip:
            continue
        if replace_at >= 0:
            out[replace_at] = value
            token_sets[replace_at] = tokens
        else:
            out.append(value)
            token_sets.append(tokens)
        if len(out) >= limit:
            break
    return out

def is_real_named_entity(text: str) -> bool:
    cleaned = re.sub('\\s+', ' ', str(text or '').strip())
    if not cleaned:
        return False
    lowered = cleaned.lower()
    if any((term in lowered for term in NON_ENTITY_TERMS)):
        return False
    lowered_tokens = _phrase_tokens(cleaned)
    if lowered_tokens & NON_ENTITY_TERMS:
        return False
    tokens = re.findall("[A-Za-zÀ-ÖØ-öø-ÿ0-9][\\w'À-ÖØ-öø-ÿ-]*", cleaned)
    if not tokens:
        return False
    properish = sum((1 for tok in tokens if tok[:1].isupper() or tok.isupper() or any((ch.isdigit() for ch in tok))))
    if properish == 0:
        return False
    if len(tokens) > 3:
        return False
    return True

def normalize(row: Dict[str, Any], parsed: Dict[str, Any], raw_response: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {'item_id': int(row.get('item_id')), 'title': str(row.get('title', ''))}
    demoted_entities: List[str] = []
    for key in LIST_KEYS:
        values = clean_list(parsed.get(key, []), limit=12)
        if key == 'evidence_used':
            values = normalize_evidence_used(row, parsed, [v for v in values if v in ALLOWED_EVIDENCE])
        if key == 'semantic_tags':
            values = [v for v in values if v.lower() not in GENERIC_TAGS and len(v.split()) >= 2]
            values = dedupe_phrases(values, limit=8)
        if key == 'named_entities':
            kept = []
            for value in values:
                if is_real_named_entity(value):
                    kept.append(value)
                else:
                    demoted_entities.append(value)
            values = kept
        if key in {'semantic_tags', 'audience_preference_facets', 'story_keywords'}:
            values = [v for v in values if supported_by_evidence(row, v)]
        out[key] = values
    if demoted_entities:
        story_values = out.get('story_keywords', [])
        for value in demoted_entities:
            if supported_by_evidence(row, value) and value.lower() not in {x.lower() for x in story_values}:
                story_values.append(value)
        out['story_keywords'] = dedupe_phrases(story_values, limit=12)
    out['recommendation_profile'] = re.sub('\\s+', ' ', str(parsed.get('recommendation_profile', '')).strip())[:500]
    confidence = str(parsed.get('profile_confidence', '')).strip().lower()
    out['profile_confidence'] = confidence if confidence in {'high', 'medium', 'low'} else ''
    out['raw_response'] = raw_response
    out['parse_ok'] = all((key in parsed for key in SCHEMA_KEYS))
    return out

def prompt_to_text(tokenizer: Any, row: Dict[str, Any]) -> str:
    content = str(row.get('prompt', ''))
    messages = [{'role': 'user', 'content': content}]
    if hasattr(tokenizer, 'apply_chat_template'):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return 'user: ' + content + '\nassistant:'

def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    root = Path.cwd()
    prompt_path = root / args.prompt_jsonl
    output_path = root / args.output_jsonl
    final_path = root / args.final_jsonl
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    skip = existing_item_ids(output_path) if args.resume else set()
    rows = [row for row in iter_jsonl(prompt_path, int(args.limit)) if int(row.get('item_id')) not in skip]
    tokenizer = AutoTokenizer.from_pretrained(root / args.model_dir, local_files_only=True, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(root / args.model_dir, local_files_only=True, trust_remote_code=True, torch_dtype=dtype_from_name(str(args.torch_dtype))).to(str(args.device))
    model.eval()
    raw_mode = 'a' if args.resume else 'w'
    final_mode = 'a' if args.resume else 'w'
    written = 0
    with torch.inference_mode(), output_path.open(raw_mode, encoding='utf-8') as raw_out, final_path.open(final_mode, encoding='utf-8') as final_out:
        for batch in batches(rows, max(1, int(args.batch_size))):
            texts = [prompt_to_text(tokenizer, row) for row in batch]
            encode_kwargs: Dict[str, Any] = {'return_tensors': 'pt', 'padding': True}
            if int(args.max_input_tokens) > 0:
                encode_kwargs.update({'truncation': True, 'max_length': int(args.max_input_tokens)})
            inputs = tokenizer(texts, **encode_kwargs).to(model.device)
            output_ids = model.generate(**inputs, max_new_tokens=int(args.max_new_tokens), do_sample=False, pad_token_id=tokenizer.eos_token_id)
            responses = tokenizer.batch_decode(output_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            for row, response in zip(batch, responses):
                parsed = extract_json(response)
                final_row = normalize(row, parsed, response)
                raw_out.write(json.dumps({**row, 'parsed': parsed, 'raw_response': response}, ensure_ascii=False) + '\n')
                final_out.write(json.dumps(final_row, ensure_ascii=False) + '\n')
                written += 1
                if written % 50 == 0:
                    print(f'[ml1m-item-profile] generated={written}', flush=True)
            raw_out.flush()
            final_out.flush()
    print(json.dumps({'raw_output': str(output_path), 'final_output': str(final_path), 'written': written}, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    main()
