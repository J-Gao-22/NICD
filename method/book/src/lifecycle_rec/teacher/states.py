import hashlib
import json
import sys
from pathlib import Path
from lifecycle_rec.teacher import state_contract as v13
base = v13.base
v13_schema = base.output_schema
v13_retry = base.retry_instruction

def _remove_people(schema):
    candidate = schema['properties']['candidate']
    variants = candidate.get('anyOf', [candidate])
    for variant in variants:
        cues = variant.get('properties', {}).get('decision_cues')
        if not cues:
            continue
        cues['properties'].pop('director_actor_cues', None)
        cues['required'] = [key for key in cues['required'] if key != 'director_actor_cues']
    return schema

def output_schema(messages):
    return _remove_people(v13_schema(messages))

def validate_output(obj, candidate_id, lens, user_prompt):
    errors = v13.validate_output(obj, candidate_id, lens, user_prompt)
    errors = [error for error in errors if error not in {'bad_liked_directors', 'bad_liked_actors', 'bad_rejected_directors', 'bad_rejected_actors'}]
    cues = (obj.get('candidate') or {}).get('decision_cues') or {}
    if 'director_actor_cues' in cues:
        errors.append('book_schema_forbids_director_actor_cues')
    return sorted(set(errors))

def retry_instruction(errors, candidate_id):
    return v13_retry(errors, candidate_id).replace('positive/negative_match_cues <= 5, and all director/actor lists as ["none"]. ', 'positive/negative_match_cues <= 5. Do not output director or actor fields for books. ')
base.output_schema = output_schema
base.validate_output = validate_output
base.retry_instruction = retry_instruction

def main():
    base.main()
