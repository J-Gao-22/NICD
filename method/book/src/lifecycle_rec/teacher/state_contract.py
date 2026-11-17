import re
import hashlib
import json
import sys
from pathlib import Path
from lifecycle_rec.teacher import state_service as base
original_validate = base.validate_output

def validate_output(obj, candidate_id, lens, user_prompt):
    errors = original_validate(obj, candidate_id, lens, user_prompt)
    candidate = obj.get('candidate', {})
    prose = [candidate.get('state_label', ''), candidate.get('state_summary', '')]
    for key, patterns in [('positive_analysis', 'liked_patterns'), ('negative_analysis', 'rejected_or_risky_patterns')]:
        analysis = candidate.get(key, {})
        prose.append(analysis.get('reasoning', ''))
        prose.extend((p.get('pattern', '') for p in analysis.get(patterns, [])))
    if re.search('\\b(?:strong|clear|persistent|permanent|consistent|stable|definite|definitive)\\s+(?:(?:positive|negative|personal|reading|user)\\s+){0,2}preference\\b', '\n'.join(prose), re.I):
        errors.append('overstated_preference_intensity')
    return sorted(set(errors))
base.validate_output = validate_output
