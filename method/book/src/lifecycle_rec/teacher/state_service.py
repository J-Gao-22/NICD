from __future__ import annotations
import argparse
import gzip
import hashlib
import itertools
import json
import math
import re
import time
import urllib.request
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Iterator

def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == '.gz' else Path.open
    with opener(path, 'rt' if path.suffix == '.gz' else 'r', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)

def append_jsonl(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
    handle.flush()

def parse_output(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith('```'):
        value = re.sub('^```(?:json)?\\s*', '', value)
        value = re.sub('\\s*```$', '', value)
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError('top-level output is not an object')
    return parsed

def normalize_text(value: Any) -> str:
    return ' '.join(re.findall('[a-z0-9]+', str(value).casefold()))

def extract_section(text: str, heading: str, next_headings: tuple[str, ...]) -> str:
    if not next_headings:
        match = re.search(f'\\[{re.escape(heading)}\\]\\n(.*)\\Z', text, flags=re.S)
        return match.group(1) if match else ''
    alternatives = '|'.join((re.escape(f'[{value}]') for value in next_headings))
    match = re.search(f'\\[{re.escape(heading)}\\]\\n(.*?)(?=\\n(?:{alternatives})|\\Z)', text, flags=re.S)
    return match.group(1) if match else ''

def segment_evidence(user_prompt: str) -> dict[str, Any]:
    positive = extract_section(user_prompt, 'POSITIVE INTERACTIONS', ('NEGATIVE INTERACTIONS',))
    negative = extract_section(user_prompt, 'NEGATIVE INTERACTIONS', ('POSITIVE COMMON TERMS', 'NEGATIVE COMMON TERMS', 'ELIGIBLE AUTHOR CUES'))
    positive_common = extract_section(user_prompt, 'POSITIVE COMMON TERMS', ('NEGATIVE COMMON TERMS',))
    negative_common = extract_section(user_prompt, 'NEGATIVE COMMON TERMS', ('ELIGIBLE AUTHOR CUES',))

    def extract_titles(section: str) -> list[str]:
        titles = []
        for value in re.findall('^- title: (.+)$', section, flags=re.M):
            titles.append(value.split(' | rating:', 1)[0].strip())
        return titles

    def repeated_authors(section: str) -> list[str]:
        counts: Counter[str] = Counter()
        canonical: dict[str, str] = {}
        for value in re.findall('^  authors: (.+)$', section, flags=re.M):
            author = value.strip()
            key = normalize_text(author)
            if not key or key == 'none':
                continue
            counts[key] += 1
            canonical.setdefault(key, author)
        return [canonical[key] for key, count in counts.most_common() if count >= 2]
    hint = re.search('^feedback_state_hint: (\\S+)$', user_prompt, flags=re.M)
    return {'positive_titles': extract_titles(positive), 'negative_titles': extract_titles(negative), 'liked_authors': repeated_authors(positive), 'rejected_authors': repeated_authors(negative), 'positive_ground_text': normalize_text(positive + '\n' + positive_common), 'negative_ground_text': normalize_text(negative + '\n' + negative_common), 'state_hint': hint.group(1) if hint else None, 'has_positive': bool(re.search('^- title:', positive, flags=re.M)), 'has_negative': bool(re.search('^- title:', negative, flags=re.M))}

def legacy_canonicalize_evidence(obj: dict[str, Any], user_prompt: str) -> int:
    candidate = obj.get('candidate')
    if not isinstance(candidate, dict) or candidate.get('candidate_status') != 'proposed':
        return 0
    evidence = segment_evidence(user_prompt)
    changes = 0
    for section_key, pattern_key, title_key in (('positive_analysis', 'liked_patterns', 'positive_titles'), ('negative_analysis', 'rejected_or_risky_patterns', 'negative_titles')):
        patterns = (candidate.get(section_key) or {}).get(pattern_key) or []
        canonical_titles = evidence[title_key]
        exact = {normalize_text(value): value for value in canonical_titles}
        for pattern in patterns:
            values = pattern.get('evidence_titles') if isinstance(pattern, dict) else None
            if not isinstance(values, list):
                continue
            for index, value in enumerate(values):
                key = normalize_text(value)
                replacement = exact.get(key)
                if replacement is None and len(key) >= 8:
                    matches = [title for title in canonical_titles if key in normalize_text(title)]
                    if len(matches) == 1:
                        replacement = matches[0]
                if replacement is not None and replacement != value:
                    values[index] = replacement
                    changes += 1
            if isinstance(values, list) and len(values) > 3:
                del values[3:]
                changes += 1
            support_terms = pattern.get('support_terms') if isinstance(pattern, dict) else None
            if isinstance(support_terms, list) and len(support_terms) > 6:
                del support_terms[6:]
                changes += 1
    cues = candidate.get('decision_cues') or {}
    for key in ('positive_match_cues', 'negative_match_cues'):
        values = cues.get(key)
        if isinstance(values, list) and len(values) > 5:
            del values[5:]
            changes += 1
    people = cues.get('director_actor_cues')
    if isinstance(people, dict):
        for key in ('liked_directors', 'liked_actors', 'rejected_directors', 'rejected_actors'):
            if people.get(key) != ['none']:
                people[key] = ['none']
                changes += 1
    authors = cues.get('author_cues') or {}
    for key in ('liked_authors', 'rejected_authors'):
        canonical = {normalize_text(value): value for value in evidence[key]}
        values = authors.get(key)
        if not isinstance(values, list):
            continue
        grounded = []
        for value in values:
            replacement = canonical.get(normalize_text(value))
            if replacement is not None and replacement not in grounded:
                grounded.append(replacement)
        grounded = grounded[:2] or ['none']
        if grounded != values:
            authors[key] = grounded
            changes += 1
    return changes

def legacy_validate_output(obj: dict[str, Any], candidate_id: str, lens: str, user_prompt: str) -> list[str]:
    errors = []
    candidate = obj.get('candidate')
    if not isinstance(candidate, dict):
        return ['missing_candidate']
    if candidate.get('candidate_id') != candidate_id:
        errors.append('candidate_id_mismatch')
    if candidate.get('analysis_lens') != lens:
        errors.append('analysis_lens_mismatch')
    status = candidate.get('candidate_status')
    if status not in {'proposed', 'insufficient_distinct_evidence'}:
        errors.append('bad_candidate_status')
    if candidate_id != 'C3' and status != 'proposed':
        errors.append('only_c3_can_be_insufficient')
    if status == 'insufficient_distinct_evidence':
        if not isinstance(candidate.get('insufficient_reason'), str):
            errors.append('missing_insufficient_reason')
        return errors
    if status != 'proposed':
        return errors
    for key in ('state_type', 'state_label', 'state_summary', 'positive_analysis', 'negative_analysis', 'decision_cues'):
        if key not in candidate:
            errors.append('missing_' + key)
    if candidate.get('state_type') not in {'positive_preference_state', 'negative_rejection_state', 'mixed_feedback_state'}:
        errors.append('bad_state_type')
    evidence = segment_evidence(user_prompt)
    if evidence['state_hint'] and candidate.get('state_type') != evidence['state_hint']:
        errors.append('state_type_hint_mismatch')
    cues = candidate.get('decision_cues') if isinstance(candidate.get('decision_cues'), dict) else {}
    for key in ('positive_match_cues', 'negative_match_cues'):
        if not isinstance(cues.get(key), list):
            errors.append('bad_' + key)
        elif len(cues[key]) > 5:
            errors.append('too_many_' + key)
    people = cues.get('director_actor_cues') if isinstance(cues.get('director_actor_cues'), dict) else {}
    for key in ('liked_directors', 'liked_actors', 'rejected_directors', 'rejected_actors'):
        if people.get(key) != ['none']:
            errors.append('bad_' + key)
    authors = cues.get('author_cues') if isinstance(cues.get('author_cues'), dict) else {}
    for key in ('liked_authors', 'rejected_authors'):
        if not isinstance(authors.get(key), list) or len(authors.get(key, [])) > 2:
            errors.append('bad_' + key)
            continue
        allowed = {normalize_text(value) for value in evidence[key]}
        provided = {normalize_text(value) for value in authors[key]}
        if provided == {'none'}:
            provided = set()
        if 'none' in provided or not provided.issubset(allowed):
            errors.append('ungrounded_' + key)
    for section, array_key, title_key, polarity in (('positive_analysis', 'liked_patterns', 'positive_titles', 'positive'), ('negative_analysis', 'rejected_or_risky_patterns', 'negative_titles', 'negative')):
        value = candidate.get(section) if isinstance(candidate.get(section), dict) else {}
        patterns = value.get(array_key)
        if not isinstance(patterns, list):
            errors.append('bad_' + array_key)
            continue
        for pattern in patterns:
            if not isinstance(pattern, dict):
                errors.append('bad_pattern_object')
                continue
            if len(pattern.get('evidence_titles', []) or []) > 3:
                errors.append('too_many_evidence_titles')
            if len(pattern.get('support_terms', []) or []) > 6:
                errors.append('too_many_support_terms')
            support_terms = pattern.get('support_terms') or []
            if not isinstance(support_terms, list) or any((not normalize_text(value) or normalize_text(value) not in evidence[polarity + '_ground_text'] for value in support_terms)):
                errors.append('ungrounded_' + polarity + '_support_term')
            allowed_titles = {normalize_text(value) for value in evidence[title_key]}
            provided_titles = pattern.get('evidence_titles') or []
            if not isinstance(provided_titles, list) or any((normalize_text(value) not in allowed_titles for value in provided_titles)):
                errors.append('ungrounded_' + polarity + '_evidence_title')
        if not evidence['has_' + polarity] and patterns:
            errors.append('invented_' + polarity + '_pattern')
    for polarity, cue_key in (('positive', 'positive_match_cues'), ('negative', 'negative_match_cues')):
        values = cues.get(cue_key) or []
        normalized = {normalize_text(value) for value in values}
        if not evidence['has_' + polarity] and normalized not in (set(), {'none'}):
            errors.append('invented_' + polarity + '_cues')
        if evidence['has_' + polarity] and any((value != 'none' and value not in evidence[polarity + '_ground_text'] for value in normalized)):
            errors.append('ungrounded_' + polarity + '_cue')
    audited_text = '\n'.join([str(candidate.get('state_label', '')), str(candidate.get('state_summary', '')), str((candidate.get('positive_analysis') or {}).get('reasoning', '')), str((candidate.get('negative_analysis') or {}).get('reasoning', '')), *[str(pattern.get('pattern', '')) for analysis_key, patterns_key in (('positive_analysis', 'liked_patterns'), ('negative_analysis', 'rejected_or_risky_patterns')) for pattern in (candidate.get(analysis_key) or {}).get(patterns_key) or [] if isinstance(pattern, dict)]])
    if re.search('\\b(?:dislike|dislikes|hates?|avoid|avoids|avoidance|reject|rejects|rejection|strong preference)\\b', audited_text, flags=re.I):
        errors.append('overstated_feedback_wording')
    if re.search('\\b(?:adequate|weak|external supported|external_supported|evidence quality|metadata status|semantic evidence)\\b', audited_text, flags=re.I):
        errors.append('semantic_evidence_status_leak')
    return errors

def canonicalize_evidence(obj, user_prompt):
    candidate = obj.get('candidate', {})
    if candidate.get('candidate_status') != 'proposed':
        return 0
    lists = []
    for analysis, field in (('positive_analysis', 'liked_patterns'), ('negative_analysis', 'rejected_or_risky_patterns')):
        for pattern in (candidate.get(analysis) or {}).get(field, []):
            lists.append(pattern.get('support_terms'))
    for field in ('positive_match_cues', 'negative_match_cues'):
        lists.append((candidate.get('decision_cues') or {}).get(field))
    changes = 0
    for values in lists:
        if not isinstance(values, list):
            continue
        kept, seen = ([], set())
        for value in values:
            key = ' '.join((x for x in normalize_text(value).split() if x != 'and'))
            if key in seen:
                changes += 1
            else:
                seen.add(key)
                kept.append(value)
        values[:] = kept
    return changes

def evidence_items(user_prompt):
    result = {}
    for polarity, heading, following in (('positive', 'POSITIVE INTERACTIONS', ('NEGATIVE INTERACTIONS',)), ('negative', 'NEGATIVE INTERACTIONS', ('POSITIVE COMMON TERMS',))):
        section = extract_section(user_prompt, heading, following)
        items = {}
        for block in re.split('(?m)^- title: ', section)[1:]:
            title = block.splitlines()[0].strip()
            terms = set()
            for field in ('categories', 'semantic_terms'):
                match = re.search(f'(?m)^  {field}: (.+)$', block)
                if match:
                    terms.update(json.loads(match.group(1)))
            items[title] = terms
        result[polarity] = items
    return result

def validate_output(obj, candidate_id, lens, user_prompt):
    errors = legacy_validate_output(obj, candidate_id, lens, user_prompt)
    errors = [e for e in errors if e not in {'ungrounded_positive_support_term', 'ungrounded_negative_support_term', 'ungrounded_positive_cue', 'ungrounded_negative_cue'}]
    candidate = obj.get('candidate', {})
    if candidate.get('candidate_status') != 'proposed':
        reason = candidate.get('insufficient_reason', '')
        if not isinstance(reason, str) or not reason.strip():
            errors.append('empty_insufficient_reason')
        if re.search('\\b(?:adequate|weak|external_supported|metadata status|evidence quality)\\b', str(reason), re.I):
            errors.append('semantic_evidence_status_leak')
        return errors
    items = evidence_items(user_prompt)

    def overlap_key(term):
        return ' '.join((x for x in normalize_text(term).split() if x != 'and'))
    term_sets = {polarity: {overlap_key(t) for terms in books.values() for t in terms} for polarity, books in items.items()}
    shared_terms = term_sets['positive'] & term_sets['negative']
    cues = candidate.get('decision_cues') or {}
    prose = [candidate.get('state_label', ''), candidate.get('state_summary', '')]
    for polarity, analysis, key in (('positive', 'positive_analysis', 'liked_patterns'), ('negative', 'negative_analysis', 'rejected_or_risky_patterns')):
        value = candidate.get(analysis) or {}
        if not isinstance(value.get('reasoning'), str):
            errors.append('missing_reasoning')
        prose.append(value.get('reasoning', ''))
        patterns = value.get(key) or []
        if len(patterns) > 2:
            errors.append('too_many_patterns')
        all_support = set()
        for pattern in patterns:
            titles = pattern.get('evidence_titles') or []
            terms = pattern.get('support_terms') or []
            prose.append(pattern.get('pattern', ''))
            if not titles or not terms or len(terms) > 4:
                errors.append('empty_or_excess_pattern_evidence')
            allowed = set().union(*(items[polarity].get(title, set()) for title in titles))
            allowed_normalized = {overlap_key(t) for t in allowed}
            if len({overlap_key(t) for t in terms}) != len(terms):
                errors.append('duplicate_semantic_support_terms')
            unsupported = [term for term in terms if overlap_key(term) not in allowed_normalized]
            if unsupported:
                errors.append('support_not_exactly_in_cited_books')
                errors.append('unsupported_terms=' + json.dumps(unsupported) + '; cited_books_allow=' + json.dumps(sorted(allowed)[:20]))
            if any((title not in items[polarity] for title in titles)):
                errors.append('title_not_verbatim')
            unsupported_titles = [title for title in titles if not {overlap_key(t) for t in terms}.intersection({overlap_key(t) for t in items[polarity].get(title, set())})]
            if unsupported_titles:
                errors.append('cited_book_contributes_no_support')
                errors.append('remove_unsupported_citations=' + json.dumps(unsupported_titles))
            all_support.update((overlap_key(t) for t in terms))
        match_cues = cues.get(polarity + '_match_cues', [])
        shared_cues = [t for t in match_cues if overlap_key(t) in shared_terms]
        if len({overlap_key(t) for t in match_cues}) != len(match_cues):
            errors.append('duplicate_match_cues')
        if shared_cues:
            errors.append('shared_term_used_as_discriminative_cue=' + json.dumps(shared_cues))
        if not match_cues or len(match_cues) > 4 or any((term != 'none' and overlap_key(term) not in term_sets[polarity] for term in match_cues)):
            errors.append('cue_not_in_same_polarity_evidence')
        if 'none' in match_cues and match_cues != ['none']:
            errors.append('none_must_be_the_only_cue')
        if not items[polarity] and match_cues != ['none']:
            errors.append('absent_polarity_requires_none')
    text = '\n'.join((str(x) for x in prose))
    if re.search('\\b(?:dislik\\w*|hat(?:e|es|ed|ing)|avoid\\w*|reject\\w*|strong preference|rated\\s+poorly|poorly\\s+rated|low[ -]rated)\\b', text, re.I):
        errors.append('overstated_feedback_morphology')
    if any((not isinstance(candidate.get(k), str) or not candidate[k].strip() for k in ('state_label', 'state_summary'))):
        errors.append('empty_state_prose')
    return sorted(set(errors))

def logprob_summary(choice: dict[str, Any]) -> dict[str, Any]:
    content = (choice.get('logprobs') or {}).get('content') or []
    sampled = []
    entropies = []
    low_confidence = 0
    for token in content:
        value = token.get('logprob') if isinstance(token, dict) else None
        if isinstance(value, (int, float)):
            sampled.append(float(value))
            low_confidence += value < -2.0
        probabilities = [math.exp(float(row['logprob'])) for row in token.get('top_logprobs') or [] if isinstance(row, dict) and isinstance(row.get('logprob'), (int, float))] if isinstance(token, dict) else []
        total = sum(probabilities)
        if total > 0:
            normalized = [probability / total for probability in probabilities]
            entropies.append(-sum((p * math.log(max(p, 1e-12)) for p in normalized)))
    mean_logprob = sum(sampled) / len(sampled) if sampled else None
    return {'token_logprobs': content, 'token_count_for_scoring': len(content), 'sampled_logprob_count': len(sampled), 'mean_sampled_token_logprob': mean_logprob, 'avg_token_nll': -mean_logprob if mean_logprob is not None else None, 'approx_topk_entropy_mean': sum(entropies) / len(entropies) if entropies else None, 'topk_entropy_count': len(entropies), 'low_conf_token_rate': low_confidence / len(sampled) if sampled else None}

def output_schema(messages):
    system, pack = (messages[0]['content'], messages[1]['content'])
    candidate_id = re.search('"candidate_id":\\s*"(C[123])"', system).group(1)
    lens = re.search('"analysis_lens":\\s*"([^"]+)"', system).group(1)
    items = evidence_items(pack)
    evidence = segment_evidence(pack)
    key = lambda value: ' '.join((x for x in normalize_text(value).split() if x != 'and'))
    terms = {p: sorted(set().union(*books.values())) if books else [] for p, books in items.items()}
    shared = {key(x) for x in terms['positive']} & {key(x) for x in terms['negative']}

    def obj(properties):
        return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}

    def strings(values, minimum=1, maximum=3):
        values = list({key(v): v for v in reversed(values)}.values())
        if 'none' in values and len(values) > 1:
            return {'anyOf': [strings(['none'], 1, 1), strings([v for v in values if v != 'none'], minimum, maximum)]}
        return {'type': 'array', 'items': {'type': 'string', 'enum': values or ['none']}, 'minItems': minimum, 'maxItems': min(maximum, len(values) or 1)}
    text = {'type': 'string'}
    analyses = {}
    for polarity, field, pattern_field in (('positive', 'positive_analysis', 'liked_patterns'), ('negative', 'negative_analysis', 'rejected_or_risky_patterns')):
        variants = []
        for size in (1,):
            for titles in itertools.combinations(items[polarity], size):
                maps = [{key(t): t for t in sorted(items[polarity][title])} for title in titles]
                common = set(maps[0]).intersection(*(set(m) for m in maps[1:]))
                if not common:
                    continue
                allowed = [maps[0][k] for k in sorted(common)]
                variants.append(obj({'evidence_titles': {'type': 'array', 'const': list(titles)}, 'support_terms': strings(allowed, 1, min(3, len(allowed))), 'pattern': text}))
        pattern = {'anyOf': variants} if variants else obj({'pattern': text, 'evidence_titles': strings(['none']), 'support_terms': strings(['none'])})
        analyses[field] = obj({pattern_field: {'type': 'array', 'items': pattern, 'minItems': 1 if terms[polarity] else 0, 'maxItems': 2 if terms[polarity] else 0}, 'reasoning': text})
    proposed = obj({'candidate_id': {'type': 'string', 'const': candidate_id}, 'analysis_lens': {'type': 'string', 'const': lens}, 'candidate_status': {'type': 'string', 'const': 'proposed'}, 'state_type': {'type': 'string', 'const': evidence['state_hint']}, **analyses, 'state_label': text, 'state_summary': text, 'decision_cues': obj({'positive_match_cues': strings([t for t in terms['positive'] if key(t) not in shared] + ['none']), 'negative_match_cues': strings([t for t in terms['negative'] if key(t) not in shared] + ['none']), 'director_actor_cues': obj({k: strings(['none'], 1, 1) for k in ('liked_directors', 'liked_actors', 'rejected_directors', 'rejected_actors')}), 'author_cues': obj({k: strings(evidence[k] + ['none'], 1, 2) for k in ('liked_authors', 'rejected_authors')})})})
    if candidate_id == 'C3':
        insufficient = obj({'candidate_id': {'const': candidate_id}, 'analysis_lens': {'const': lens}, 'candidate_status': {'const': 'insufficient_distinct_evidence'}, 'insufficient_reason': text})
        return obj({'candidate': {'anyOf': [proposed, insufficient]}})
    return obj({'candidate': proposed})

def call_chat(args: argparse.Namespace, messages: list[dict[str, str]]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    payload = {'model': args.model, 'messages': messages, 'temperature': getattr(args, 'temperature', 0.0), 'seed': getattr(args, 'seed', 0), 'max_tokens': args.max_output_tokens, 'logprobs': True, 'top_logprobs': args.top_logprobs, 'response_format': {'type': 'json_schema', 'json_schema': {'name': 'book_state_candidate', 'strict': True, 'schema': output_schema(messages)}}}
    request = urllib.request.Request(args.endpoint.rstrip('/') + '/chat/completions', data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        data = json.loads(response.read().decode('utf-8'))
    choice = data['choices'][0]
    if choice.get('finish_reason') != 'stop':
        raise ValueError('incomplete_generation:' + str(choice.get('finish_reason')))
    if not (choice.get('logprobs') or {}).get('content'):
        raise ValueError('missing_token_logprobs')
    return (choice['message']['content'], logprob_summary(choice), data.get('usage') or {})

def retry_instruction(errors: list[str], candidate_id: str) -> str:
    instruction = 'Your previous JSON failed validation: ' + ', '.join(errors)[:1000] + '. Regenerate the full JSON only. Keep evidence_titles <= 3, support_terms <= 6, positive/negative_match_cues <= 5, and all director/actor lists as ["none"]. Copy state_type from feedback_state_hint. Copy evidence titles verbatim from the matching polarity. Copy every support term and match cue from the matching polarity\'s categories, semantic_terms, or common terms. Use only author names appearing in at least two items of the matching polarity; otherwise ["none"]. Use local positive signal and tentative local caution wording; no strong preference, reject/rejected/rejection, dislike variants, rated poorly, or low-rated. Use at most ONE pattern per polarity. Choose just ONE or TWO representative evidence titles and TWO or THREE support terms from those books. Do not list every book. Each support term must match a complete categories/semantic_terms JSON-array element of a CITED book. Every cited book must contribute a term. Match cues must be copied from support_terms. Fewer terms are fine. Terms occurring on BOTH polarities cannot be match cues. If no distinguishing term remains, use ["none"] match cues and say no distinct semantic contrast is supported. Shared terms may support a cautious descriptive pattern. When there are no negative interactions, write exactly: No negative interactions are present, so no local caution pattern is inferred. When there are no positive interactions, write exactly: No positive interactions are present, so no local positive pattern is inferred.'
    if candidate_id == 'C3' and any(('ungrounded' in error for error in errors)):
        instruction += ' For C3, return candidate_status insufficient_distinct_evidence when no fully grounded fine-grained signal remains.'
    return instruction

def validate_input_records(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    errors = []
    ids = set()
    sample_ids = set()
    candidate_counts: Counter[str] = Counter()
    token_lengths = []
    for index, row in enumerate(rows):
        record_id = row.get('record_id')
        if not isinstance(record_id, str) or record_id in ids:
            errors.append({'index': index, 'type': 'missing_or_duplicate_record_id', 'record_id': record_id})
        ids.add(record_id)
        sample_ids.add(row.get('sample_id'))
        candidate_counts[str(row.get('candidate_id'))] += 1
        messages = row.get('messages')
        if not isinstance(messages, list) or [message.get('role') for message in messages] != ['system', 'user']:
            errors.append({'index': index, 'type': 'bad_messages_shape', 'record_id': record_id})
            continue
        serialized = json.dumps(messages, ensure_ascii=False)
        user_prompt = messages[1]['content']
        evidence = segment_evidence(user_prompt)
        positive = len(evidence['positive_titles'])
        negative = len(evidence['negative_titles'])
        expected = 'positive_preference_state' if positive and (not negative) else 'negative_rejection_state' if negative and (not positive) else 'positive_preference_state' if positive >= 2 and negative <= 1 else 'negative_rejection_state' if negative >= 2 and positive <= 1 else 'mixed_feedback_state'
        if evidence['state_hint'] != expected:
            errors.append({'index': index, 'type': 'feedback_hint_wrong', 'record_id': record_id})
        orders = [int(x) for x in re.findall('(?m)^  order: (\\d+)$', user_prompt)]
        if sorted(orders) != list(range(1, positive + negative + 1)):
            errors.append({'index': index, 'type': 'bad_chronological_positions', 'record_id': record_id})
        for polarity, count in (('positive', positive), ('negative', negative)):
            match = re.search(f'(?m)^{polarity}_count: (\\d+)$', user_prompt)
            if not match or int(match.group(1)) != count:
                errors.append({'index': index, 'type': 'feedback_count_mismatch', 'record_id': record_id})
        evidence_items(user_prompt)
        if any((marker in serialized for marker in ('target_iid:', 'target_item_id:', 'target_label:', 'sample_id:'))):
            errors.append({'index': index, 'type': 'target_marker_in_messages', 'record_id': record_id})
        prompt_tokens = row.get('prompt_tokens')
        if not isinstance(prompt_tokens, int):
            errors.append({'index': index, 'type': 'missing_prompt_tokens', 'record_id': record_id})
        else:
            token_lengths.append(prompt_tokens)
            if prompt_tokens + args.max_output_tokens > args.max_model_len:
                errors.append({'index': index, 'type': 'context_overflow', 'record_id': record_id})
    return {'records': len(rows), 'samples': len(sample_ids), 'unique_record_ids': len(ids), 'candidate_counts': dict(candidate_counts), 'prompt_tokens_min': min(token_lengths) if token_lengths else None, 'prompt_tokens_max': max(token_lengths) if token_lengths else None, 'errors': errors[:100], 'error_count': len(errors)}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--messages-file', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--endpoint', default='http://127.0.0.1:8010/v1')
    parser.add_argument('--model', default='Qwen2.5-72B-Instruct-AWQ')
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--max-attempts', type=int, default=2)
    parser.add_argument('--max-output-tokens', type=int, default=768)
    parser.add_argument('--max-model-len', type=int, default=4096)
    parser.add_argument('--top-logprobs', type=int, default=5)
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--resume-skip-existing', action='store_true')
    parser.add_argument('--gzip-output', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    started = time.time()
    rows = list(read_jsonl(args.messages_file))
    if args.limit:
        rows = rows[:args.limit]
    input_audit = validate_input_records(rows, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if input_audit['error_count']:
        (args.output_dir / 'input_dry_run_report.json').write_text(json.dumps(input_audit, indent=2) + '\n', encoding='utf-8')
        raise RuntimeError(f"Input validation failed: {input_audit['error_count']} errors")
    if args.dry_run:
        input_audit['elapsed_sec'] = time.time() - started
        (args.output_dir / 'input_dry_run_report.json').write_text(json.dumps(input_audit, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(input_audit, indent=2), flush=True)
        return
    suffix = '.jsonl.gz' if args.gzip_output else '.jsonl'
    candidate_path = args.output_dir / ('state_candidates' + suffix)
    existing = set()
    if args.resume_skip_existing and candidate_path.exists():
        requested = {row['record_id']: row for row in rows}
        for saved in read_jsonl(candidate_path):
            source = requested.get(saved['record_id'])
            if source is None or saved['pack_hash'] != source['pack_hash']:
                raise RuntimeError('Resume input mismatch: ' + saved['record_id'])
            if validate_output(saved['model_output'], source['candidate_id'], source['analysis_lens'], source['messages'][1]['content']):
                raise RuntimeError('Resume output fails current validator: ' + saved['record_id'])
            existing.add(saved['record_id'])
    rows = [row for row in rows if row['record_id'] not in existing]
    attempts_path = args.output_dir / ('state_generation_attempts_v3' + suffix)
    failures_path = args.output_dir / ('state_generation_failures_v3' + suffix)
    logprobs_path = args.output_dir / ('state_candidate_logprobs_v3' + suffix)

    def run_one(row: dict[str, Any]) -> dict[str, Any]:
        base_messages = row['messages']
        messages = base_messages
        attempts = []
        output = None
        final_errors = []
        final_summary = {}
        final_usage = {}
        for attempt_index in range(1, args.max_attempts + 1):
            attempt_started = time.time()
            raw = ''
            status = 'success'
            errors = []
            summary = {}
            usage = {}
            try:
                request_args = argparse.Namespace(**vars(args))
                request_args.temperature = 0.0 if attempt_index == 1 else 0.2
                request_args.seed = (int(hashlib.sha256(row['record_id'].encode()).hexdigest()[:8], 16) + attempt_index) % 2 ** 31
                raw, summary, usage = call_chat(request_args, messages)
                output = parse_output(raw)
                canonicalization_count = canonicalize_evidence(output, base_messages[1]['content'])
                errors = validate_output(output, row['candidate_id'], row['analysis_lens'], base_messages[1]['content'])
                if errors:
                    status = 'validator_error'
            except Exception as error:
                status = 'request_or_parse_error'
                errors = [repr(error)]
            attempts.append({'record_id': row['record_id'], 'attempt_index': attempt_index, 'status': status, 'errors': errors, 'latency_sec': round(time.time() - attempt_started, 3), 'request_messages': messages, 'temperature': 0.0 if attempt_index == 1 else 0.2, 'seed': request_args.seed, 'usage': usage, 'logprob_summary': summary, 'raw_output': raw})
            final_errors, final_summary, final_usage = (errors, summary, usage)
            if status == 'success':
                return {'row': row, 'attempts': attempts, 'output': output, 'errors': [], 'summary': summary, 'usage': usage, 'canonicalization_count': canonicalization_count}
            if attempt_index < args.max_attempts:
                correction = retry_instruction(errors, row['candidate_id'])
                if attempt_index >= 2:
                    correction += " Change your previous evidence selection. For each polarity choose ONE representative book and just ONE or TWO of that exact book's semantic phrases. For mixed feedback, prefer phrases absent from the other polarity. Keep all observations explicitly local."
                messages = [base_messages[0], {'role': 'user', 'content': base_messages[1]['content'] + '\n\n[VALIDATION CORRECTION]\n' + correction}]
        return {'row': row, 'attempts': attempts, 'output': None, 'errors': final_errors, 'summary': final_summary, 'usage': final_usage, 'canonicalization_count': 0}
    success = 0
    failures = 0
    canonicalizations = 0

    def open_output(path):
        if args.gzip_output:
            return gzip.open(path, 'at', encoding='utf-8', compresslevel=6)
        return path.open('a', encoding='utf-8')
    with open_output(attempts_path) as attempts_out, open_output(candidate_path) as candidates_out, open_output(failures_path) as failures_out, open_output(logprobs_path) as logprobs_out, ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = {executor.submit(run_one, row) for row in rows[:args.workers]}
        next_index = len(pending)
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                row = result['row']
                for attempt in result['attempts']:
                    append_jsonl(attempts_out, attempt)
                append_jsonl(logprobs_out, {'record_id': row['record_id'], 'status': 'success' if result['output'] else 'failed', 'logprob_summary': result['summary'], 'usage': result['usage'], 'logprob_config': {'logprobs': True, 'top_logprobs': args.top_logprobs}})
                if result['output']:
                    append_jsonl(candidates_out, {**{key: row[key] for key in ('record_id', 'sample_id', 'user_id', 'item_id', 'segment_id', 'candidate_id', 'analysis_lens', 'pack_hash')}, 'model_output': result['output'], 'logprob_summary': result['summary'], 'usage': result['usage'], 'canonicalization_count': result['canonicalization_count']})
                    success += 1
                    canonicalizations += result['canonicalization_count']
                else:
                    append_jsonl(failures_out, {'record_id': row['record_id'], 'errors': result['errors']})
                    failures += 1
                if next_index < len(rows):
                    pending.add(executor.submit(run_one, rows[next_index]))
                    next_index += 1
    report = {'version': 'amazon_book_state_72b_message_runner_v3_12_lossless_dedup', 'normalization_policy': 'remove semantically identical duplicate support terms/cues only; no prose, title, or author changes', 'structured_decoding': 'json_schema_source_term_enums', 'logprobs_mode': 'raw_logprobs (server default verified from installed vLLM config)', 'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'messages_sha256': hashlib.sha256(args.messages_file.read_bytes()).hexdigest(), 'messages_file': str(args.messages_file), 'endpoint': args.endpoint, 'model': args.model, 'max_model_len': args.max_model_len, 'tasks': len(rows), 'success': success, 'failures': failures, 'workers': args.workers, 'max_attempts': args.max_attempts, 'temperature': 0.0, 'max_output_tokens': args.max_output_tokens, 'repair_temperature': 0.2, 'seed_policy': 'sha256(record_id) first 32 bits plus attempt index modulo 2**31', 'logprobs': True, 'top_logprobs': args.top_logprobs, 'output_compression': 'gzip' if args.gzip_output else 'none', 'duplicate_signal_normalizations': canonicalizations, 'input_audit': input_audit, 'elapsed_sec': time.time() - started}
    (args.output_dir / 'state_generation_report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
