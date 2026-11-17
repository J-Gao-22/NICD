from pathlib import Path
import json
import re
import collections

def seg_ord(s):
    s = str(s)
    if s.startswith('S'):
        try:
            return int(s[1:])
        except Exception:
            pass
    return 999

def clean_list(xs):
    out = []
    for x in xs or []:
        if isinstance(x, dict):
            sig = x.get('signal') or x.get('term') or x.get('value') or x.get('label') or ''
            why = x.get('why') or x.get('reason') or ''
            if sig and why:
                out.append(f'{sig} ({why})')
            elif sig:
                out.append(str(sig))
        elif x:
            out.append(str(x))
    return ', '.join(out) if out else 'none'

def state_obj(row):
    return row.get('state_node') or row.get('selected_candidate') or row

def render_state(row):
    s = state_obj(row)
    lines = [f"[STATE {row.get('segment_id')}]"]
    lines.append('State label: ' + str(s.get('state_label') or ''))
    lines.append('State summary: ' + str(s.get('state_summary') or ''))
    pos = s.get('positive_analysis') if isinstance(s.get('positive_analysis'), dict) else {}
    neg = s.get('negative_analysis') if isinstance(s.get('negative_analysis'), dict) else {}
    cues = s.get('decision_cues') if isinstance(s.get('decision_cues'), dict) else {}
    lines.append('Positive analysis:')
    pats = pos.get('liked_patterns') or pos.get('positive_patterns') or []
    if pats:
        for p in pats[:3]:
            if isinstance(p, dict):
                pattern = p.get('pattern') or p.get('factor') or ''
                terms = ', '.join((str(x) for x in (p.get('support_terms') or [])[:6]))
                titles = ', '.join((str(x) for x in (p.get('evidence_titles') or [])[:3]))
                tail = f' ({terms})' if terms else ''
                ev = f'; evidence: {titles}' if titles else ''
                lines.append(f'- {pattern}{tail}{ev}')
    else:
        lines.append('- none')
    lines.append('Negative / caution analysis:')
    npats = neg.get('rejected_or_risky_patterns') or neg.get('negative_patterns') or []
    if npats:
        for p in npats[:3]:
            if isinstance(p, dict):
                pattern = p.get('pattern') or p.get('boundary') or ''
                terms = ', '.join((str(x) for x in (p.get('support_terms') or [])[:6]))
                titles = ', '.join((str(x) for x in (p.get('evidence_titles') or [])[:3]))
                tail = f' ({terms})' if terms else ''
                ev = f'; evidence: {titles}' if titles else ''
                lines.append(f'- {pattern}{tail}{ev}')
    else:
        lines.append('- none')
    lines.append('Decision cues:')
    pos_cues = cues.get('positive_match_cues') or cues.get('positive_cues') or []
    neg_cues = cues.get('negative_match_cues') or cues.get('negative_cues') or []
    lines.append('Positive cues: ' + (', '.join(map(str, pos_cues[:8])) if pos_cues else 'none'))
    lines.append('Negative cues: ' + (', '.join(map(str, neg_cues[:8])) if neg_cues else 'none'))
    da = cues.get('director_actor_cues') if isinstance(cues.get('director_actor_cues'), dict) else {}
    people = []
    for k in ['liked_directors', 'liked_actors', 'rejected_directors', 'rejected_actors']:
        vals = [str(x) for x in da.get(k) or [] if str(x).lower() != 'none']
        if vals:
            people.append(k + ': ' + ', '.join(vals[:2]))
    lines.append('Director / actor cues: ' + ('; '.join(people) if people else 'none'))
    return '\n'.join(lines)

def render_edge(row):
    node = (row.get('model_output') or {}).get('edge_node') or row.get('edge_node') or {}
    fs = node.get('from_state_id') or row.get('from_segment')
    ts = node.get('to_state_id') or row.get('to_segment')
    lines = [f'[EDGE {fs}->{ts}]']
    lines.append('Transition label: ' + str(node.get('transition_label') or ''))
    lines.append('Transition type: ' + str(node.get('transition_type') or ''))
    pos = node.get('positive_transition') if isinstance(node.get('positive_transition'), dict) else {}
    neg = node.get('negative_caution_transition') if isinstance(node.get('negative_caution_transition'), dict) else {}
    lines.append('Positive transition:')
    lines.append('- analysis: ' + str(pos.get('analysis') or ''))
    lines.append('- retained: ' + clean_list(pos.get('retained')))
    lines.append('- emerging: ' + clean_list(pos.get('emerging')))
    lines.append('- fading context: ' + clean_list(pos.get('fading')))
    lines.append('Negative caution transition:')
    lines.append('- analysis: ' + str(neg.get('analysis') or ''))
    lines.append('- retained: ' + clean_list(neg.get('retained')))
    lines.append('- emerging: ' + clean_list(neg.get('emerging')))
    lines.append('- fading: ' + clean_list(neg.get('fading')))
    conflicts = node.get('polarity_conflicts') or []
    lines.append('Polarity conflicts: ' + clean_list(conflicts))
    lines.append('Transition summary: ' + str(node.get('transition_summary') or ''))
    return '\n'.join(lines)
