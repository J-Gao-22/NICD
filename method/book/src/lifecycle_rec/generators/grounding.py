import argparse
import json
import re
from collections import Counter
from pathlib import Path

def read_jsonl(path):
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)

def validate(record_id, raw_output, source):
    failures = []
    try:
        output = json.loads(raw_output)
    except Exception:
        return ['json_parse_failed']
    if set(output) != {'interest_state_node'}:
        return ['bad_outer_schema']
    node = output['interest_state_node']
    user = source['messages'][1]['content']
    hint = re.search('^feedback_state_hint:\\s*(\\S+)\\s*$', user, flags=re.M)
    if not hint or node.get('state_type') != hint.group(1):
        failures.append('state_type_not_grounded')
    positive = user.split('[POSITIVE INTERACTIONS]', 1)[-1].split('[NEGATIVE INTERACTIONS]', 1)[0]
    negative_tail = user.split('[NEGATIVE INTERACTIONS]', 1)
    negative = negative_tail[-1].split('[POSITIVE COMMON TERMS]', 1)[0] if len(negative_tail) == 2 else ''
    for analysis_key, pattern_key, text, polarity in (('positive_analysis', 'liked_patterns', positive, 'positive'), ('negative_analysis', 'rejected_or_risky_patterns', negative, 'negative')):
        patterns = (node.get(analysis_key) or {}).get(pattern_key) or []
        if len(patterns) > 2:
            failures.append(f'too_many_{polarity}_patterns')
        for pattern in patterns:
            titles = pattern.get('evidence_titles') or []
            terms = pattern.get('support_terms') or []
            if len(titles) != 1:
                failures.append(f'bad_{polarity}_title_count')
            if not 1 <= len(terms) <= 4:
                failures.append(f'bad_{polarity}_support_count')
            for title in titles:
                if title not in text:
                    failures.append(f'ungrounded_{polarity}_title')
            for term in terms:
                if term not in text:
                    failures.append(f'ungrounded_{polarity}_term')
    cues = node.get('decision_cues') or {}
    for key, text, polarity in (('positive_match_cues', positive, 'positive'), ('negative_match_cues', negative, 'negative')):
        values = cues.get(key) or []
        if not 1 <= len(values) <= 4:
            failures.append(f'bad_{polarity}_cue_count')
        for value in values:
            if value != 'none' and value not in text:
                failures.append(f'ungrounded_{polarity}_cue')
    positive_authors = re.findall('^\\s*authors:\\s*(.+?)\\s*$', positive, flags=re.M)
    negative_authors = re.findall('^\\s*authors:\\s*(.+?)\\s*$', negative, flags=re.M)
    author_cues = cues.get('author_cues') or {}
    for key, authors, polarity in (('liked_authors', positive_authors, 'positive'), ('rejected_authors', negative_authors, 'negative')):
        for author in author_cues.get(key) or []:
            if author == 'none':
                continue
            if author not in authors:
                failures.append(f'{polarity}_author_not_exact_input')
            if sum((normalized_author(value) == normalized_author(author) for value in authors)) < 2:
                failures.append(f'ungrounded_{polarity}_author')
    return sorted(set(failures))
