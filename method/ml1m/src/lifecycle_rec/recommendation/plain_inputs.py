from __future__ import annotations
import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any
import pandas as pd
SYSTEM_PROMPT = 'You are a movie recommendation classifier. Answer with exactly one token: Yes or No.'

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='\n') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n')
    tmp.replace(path)

def words(text: str) -> list[str]:
    return str(text or '').split()

def truncate_words(text: str, max_words: int) -> str:
    ws = words(text)
    if len(ws) <= max_words:
        return ' '.join(ws)
    return ' '.join(ws[:max_words]).rstrip(' ,.;:') + '...'

def clean_one_line(text: str) -> str:
    return re.sub('\\s+', ' ', str(text or '').replace('\n', ' ')).strip()

def load_id2title(path: Path) -> dict[int, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding='utf-8'))
    return {int(k): v for k, v in raw.items() if str(k).isdigit()}

def load_semantic_cache(path: Path) -> dict[int, dict[str, Any]]:
    items: dict[int, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if 'item_id' in row:
            items[int(row['item_id'])] = row
    return items

def load_text_variant(path: Path) -> dict[int, dict[str, Any]]:
    items: dict[int, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if 'item_id' in row:
            items[int(row['item_id'])] = row
    return items

class ItemRenderer:

    def __init__(self, id2title: dict[int, dict[str, Any]], cache: dict[int, dict[str, Any]], compact: dict[int, dict[str, Any]], balanced: dict[int, dict[str, Any]]):
        self.id2title = id2title
        self.cache = cache
        self.compact = compact
        self.balanced = balanced
        self.cache_by_title = self._build_title_map(cache)
        self.compact_by_title = self._build_title_map(compact)
        self.balanced_by_title = self._build_title_map(balanced)

    @staticmethod
    def _norm_title(title: str) -> str:
        t = clean_one_line(title).lower()
        t = re.sub('\\s+', ' ', t)
        for suffix, prefix in [(', the', 'the '), (', a', 'a '), (', an', 'an ')]:
            if t.endswith(suffix):
                t = prefix + t[:-len(suffix)]
        return t.strip()

    def _build_title_map(self, items: dict[int, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for row in items.values():
            title = row.get('title') or (row.get('metadata') or {}).get('title')
            if title:
                out.setdefault(self._norm_title(str(title)), row)
        return out

    def _by_title_or_id(self, item_id: int, source: str) -> dict[str, Any]:
        title = self._fallback_title(item_id)
        norm = self._norm_title(title)
        if source == 'cache':
            return self.cache_by_title.get(norm) or self.cache.get(int(item_id), {})
        if source == 'compact':
            return self.compact_by_title.get(norm) or self.compact.get(int(item_id), {})
        if source == 'balanced':
            return self.balanced_by_title.get(norm) or self.balanced.get(int(item_id), {})
        return {}

    def _fallback_title(self, item_id: int) -> str:
        return clean_one_line((self.id2title.get(item_id) or {}).get('title') or (self.compact.get(item_id) or {}).get('title') or (self.balanced.get(item_id) or {}).get('title') or f'item_{item_id}')

    def _fallback_genres(self, item_id: int) -> list[str]:
        kw = (self.id2title.get(item_id) or {}).get('keywords') or ''
        return [x.strip() for x in str(kw).split(',') if x.strip()][:6]

    def history_item(self, item_id: int) -> str:
        item = self._by_title_or_id(int(item_id), 'cache')
        meta = item.get('metadata') or {}
        prof = item.get('llm_profile') or {}
        title = clean_one_line(self._fallback_title(item_id))
        genres = (meta.get('ml1m_genres') or meta.get('tmdb_genres') or self._fallback_genres(item_id))[:4]
        rec_profile = clean_one_line(prof.get('recommendation_profile') or '')
        terms = (prof.get('semantic_terms_for_embedding') or prof.get('core_themes') or [])[:6]
        if not rec_profile:
            text = clean_one_line((self._by_title_or_id(int(item_id), 'compact') or self._by_title_or_id(int(item_id), 'balanced') or {}).get('text') or '')
            rec_profile = text
        parts = [title]
        if genres:
            parts.append('genres=' + ', '.join(map(str, genres)))
        if terms:
            parts.append('terms=' + ', '.join(map(str, terms)))
        if rec_profile:
            parts.append('profile=' + truncate_words(rec_profile, 28))
        return '- ' + ' | '.join(parts)

    def target_item(self, item_id: int) -> str:
        item = self._by_title_or_id(int(item_id), 'cache')
        meta = item.get('metadata') or {}
        prof = item.get('llm_profile') or {}
        title = clean_one_line(self._fallback_title(item_id))
        year = clean_one_line(meta.get('year') or '')
        genres = (meta.get('ml1m_genres') or meta.get('tmdb_genres') or self._fallback_genres(item_id))[:5]
        rec_profile = clean_one_line(prof.get('recommendation_profile') or '')
        if not rec_profile:
            text = clean_one_line((self._by_title_or_id(int(item_id), 'balanced') or self._by_title_or_id(int(item_id), 'compact') or {}).get('text') or '')
            rec_profile = text
        terms = (prof.get('semantic_terms_for_embedding') or [])[:8]
        themes = (prof.get('core_themes') or [])[:4]
        tone = (prof.get('emotional_tone') or [])[:4]
        style = (prof.get('narrative_style') or [])[:3]
        people: list[str] = []
        if meta.get('director'):
            people.append(str(meta['director']))
        people.extend([str(x) for x in (meta.get('actors') or [])[:4]])
        lines = [f'title: {title}' + (f' ({year})' if year else '')]
        if genres:
            lines.append('genres: ' + ', '.join(map(str, genres)))
        if rec_profile:
            lines.append('profile: ' + truncate_words(rec_profile, 120))
        if terms:
            lines.append('semantic_tags: ' + ', '.join(map(str, terms)))
        if themes:
            lines.append('themes: ' + ', '.join(map(str, themes)))
        if style or tone:
            lines.append('style_tone: ' + ', '.join(map(str, style + tone)))
        if people:
            lines.append('people: ' + ', '.join(people[:5]))
        return '\n'.join(lines)

def render_history_semantics(his: Any, renderer: ItemRenderer, max_items: int=15) -> str:
    ids = [int(x) for x in list(his) if int(x) != 0]
    ids = ids[-max_items:]
    if not ids:
        return '- No substantial prior item history is available.'
    return '\n'.join((renderer.history_item(i) for i in ids))

def make_plain_user_prompt(history_text: str, target_text: str) -> str:
    return f'User history:\n{history_text}\n\nTarget movie:\n{target_text}\n\nWould the user enjoy the target movie?'

def build_record(record_id: str, split: str, input_type: str, uid: int, iid: int, label: int, user_prompt: str, extra: dict[str, Any] | None=None) -> dict[str, Any]:
    rec = {'record_id': record_id, 'split': split, 'input_type': input_type, 'uid': int(uid), 'iid': int(iid), 'label': int(label), 'system': SYSTEM_PROMPT, 'user': user_prompt}
    if extra:
        rec.update(extra)
    return rec

def build_plain_records(df: pd.DataFrame, split: str, renderer: ItemRenderer) -> tuple[list[dict[str, Any]], Counter]:
    rows: list[dict[str, Any]] = []
    stats: Counter = Counter()
    for idx, row in df.iterrows():
        iid = int(row['iid'])
        hist = render_history_semantics(row['his'], renderer)
        target = renderer.target_item(iid)
        if 'title:' in target:
            stats['target_semantic_available'] += 1
        user = make_plain_user_prompt(hist, target)
        rows.append(build_record(record_id=f"{split}::plain::{int(idx)}::u{int(row['uid'])}_i{iid}", split=split, input_type='plain', uid=int(row['uid']), iid=iid, label=int(row['label']), user_prompt=user, extra={'row_index': int(idx), 'timestamp': int(row.get('timestamp', -1)) if 'timestamp' in row else None}))
    return (rows, stats)
