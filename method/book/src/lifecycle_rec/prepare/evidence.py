import json
import re
from lifecycle_rec.prepare import inputs as base
original_hint = base.state_hint
original_render = base.render_pack

def state_hint(labels):
    if labels and all((x == 1 for x in labels)):
        return 'positive_preference_state'
    if labels and all((x == 0 for x in labels)):
        return 'negative_rejection_state'
    return original_hint(labels)

def semantic_terms(item):
    categories = base.compact_categories(item)
    category_words = set(base.norm(' '.join(categories)).split())
    title_words = set(base.norm(item.get('title', '')).split())
    values = base.as_list(item.get('semantic_signal_terms'))
    result = []
    for term in base.dedupe(values):
        key = base.norm(term)
        if key in base.GENERIC_TERMS or key in {'volume', 'series', 'page turner', 'can t put down', 'entertaining read', 'exciting read', 'fun story', 'funny book'}:
            continue
        if re.fullmatch('(?:book|volume|vol|part)\\s*\\d+', key):
            continue
        if base.semantic_evidence(item) == 'weak' and len(key.split()) == 1 and (key in title_words) and (key not in category_words):
            continue
        if key in base.CATALOG_NOISE_TERMS or key in {'she', 'her', 'he', 'his', 'you', 'your', 'our', 'they', 'their'}:
            continue
        result.append(term)
    return result[:10]

def item_terms(item):
    return [base.norm(x) for x in base.compact_categories(item) + semantic_terms(item) if base.norm(x)]

def render_item(iid, order, rating, timestamp, item, target_title_key):
    title = base.clean(item.get('title')) or 'unknown'
    if target_title_key and target_title_key in base.norm(title):
        title = 'none'
    categories = base.without_target_phrase(base.compact_categories(item), target_title_key, 2)
    terms = base.without_target_phrase(semantic_terms(item), target_title_key, 10)
    authors = base.without_target_phrase([base.clean_author(x) for x in base.as_list(item.get('authors'))], target_title_key, 1)
    return '\n'.join([f'- title: {title}', f'  rating: {rating:g}', f'  order: {order}', '  categories: ' + json.dumps(categories, ensure_ascii=False), '  semantic_terms: ' + json.dumps(terms, ensure_ascii=False), '  authors: ' + (', '.join(authors) or 'none'), '  semantic_evidence: ' + base.semantic_evidence(item)])

def render_pack(row, segment, items, target_title):
    pack = original_render(row, segment, items, target_title)
    labels = row['his_label'][segment['start']:segment['end']]
    orders = iter([i + 1 for i, x in enumerate(labels) if x == 1] + [i + 1 for i, x in enumerate(labels) if x == 0])
    pack = re.sub('^  order: \\d+$', lambda _: f'  order: {next(orders)}', pack, flags=re.M)
    return pack.replace('negative_rejection_signal', 'local_caution_signal').replace('positive_discriminative', 'positive_observed_only')
base.state_hint = state_hint
base.render_item = render_item
base.render_pack = render_pack
base.item_terms = item_terms
base.LENSES = (('C1', 'dominant_feedback_pattern', 'Describe the main locally observed positive signal and local caution, where supported.'), ('C2', 'positive_negative_contrast', 'Describe specific semantic differences between polarities. With only one polarity, explicitly limit the observation to it.'), ('C3', 'fine_grained_decision_signal', 'Find a specific supported signal beyond broad categories. Return insufficient_distinct_evidence if only broad categories or title fragments are available.'))
if __name__ == '__main__':
    base.main()
