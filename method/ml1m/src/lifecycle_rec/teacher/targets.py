import argparse
import concurrent.futures as cf
import json
import math
import os
import random
import re
import time
import urllib.request
from collections import defaultdict

def words(s, max_words):
    toks = str(s or '').split()
    return ' '.join(toks[:max_words])

def norm_list(x, n=10):
    if x is None:
        return []
    if isinstance(x, str):
        parts = re.split('[,;|]', x)
        return [p.strip() for p in parts if p.strip()][:n]
    if isinstance(x, list):
        vals = []
        for v in x:
            if isinstance(v, dict):
                vals.append(v.get('term') or v.get('name') or v.get('tag') or v.get('keyword') or v.get('value') or '')
            else:
                vals.append(str(v))
        return [v.strip() for v in vals if v and v.strip()][:n]
    return []

def item_title(item, sample=None):
    for k in ['title', 'movie_title', 'name']:
        if item.get(k):
            return item[k]
    if sample:
        return sample.get('target_title') or f"item_{sample.get('target_item_id')}"
    return 'unknown'

def render_item(iid, item, sample):
    title = item_title(item, sample)
    year = item.get('year') or ''
    genres = item.get('genres') or item.get('genre') or sample.get('genres') or ''
    if isinstance(genres, list):
        genres = ', '.join(map(str, genres))
    director = item.get('director') or item.get('directors') or ''
    if isinstance(director, list):
        director = '; '.join(map(str, director[:3]))
    actors = item.get('actors') or item.get('actor') or item.get('cast') or []
    if isinstance(actors, str):
        actors_txt = actors
    else:
        actors_txt = '; '.join(map(str, actors))
    terms = []
    for k in ['semantic_terms', 'semantic_tags', 'keywords', 'tags', 'key_terms']:
        terms += norm_list(item.get(k), 30)
    seen = set()
    terms2 = []
    for t in terms:
        lt = t.lower()
        if lt not in seen:
            seen.add(lt)
            terms2.append(t)
        if len(terms2) >= 15:
            break
    profile = item.get('profile') or item.get('semantic_profile') or item.get('short_profile') or item.get('text') or item.get('description') or ''
    return '\n'.join([f'Title: {title}' + (f' ({year})' if year else ''), f'Genres: {genres}', f'Director: {director}', f'Actors: {actors_txt}', 'Semantic terms: ' + (', '.join(terms2) if terms2 else 'unknown'), 'Profile: ' + words(profile, 140)])
