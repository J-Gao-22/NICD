from __future__ import annotations
import argparse
import json
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any
from transformers import AutoTokenizer
from lifecycle_rec.teacher import io as stategen
BAD_MAIN_RELATIONS = {'actor', 'director', 'writer', 'country', 'language'}
GENERIC_GENRES = {'action', 'adventure', 'animation', "children's", 'comedy', 'crime', 'documentary', 'drama', 'fantasy', 'film-noir', 'horror', 'musical', 'mystery', 'romance', 'sci-fi', 'thriller', 'war', 'western'}

def norm_term(term: Any) -> str:
    return re.sub('\\s+', ' ', str(term or '').strip()).lower()

def is_bad_main_term(row: dict[str, Any] | None, term: str) -> bool:
    relation = norm_term((row or {}).get('relation'))
    if relation in BAD_MAIN_RELATIONS:
        return True
    return relation == 'genre' and norm_term(term) in GENERIC_GENRES

def pick_transition_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    picked = []
    for row in rows or []:
        term = str(row.get('value') or '').strip()
        if not term or is_bad_main_term(row, term):
            continue
        picked.append(row)
        if len(picked) >= limit:
            break
    return picked
