import re
PATTERN = re.compile('::row_(\\d+)::u([^_]+)_i([^_]+)_t(\\d+)$')

def sample_id(r):
    return str(r.get('sample_id') or r['record_id'].split('::state_v3_chain::', 1)[-1])

def hybrid_explicit(chain, plain, frame):
    assert len(plain) == len(frame), 'PLAIN_POPULATION_MISMATCH'
    index = {}
    for r in plain:
        row = int(r['row_index'])
        assert row not in index, 'DUPLICATE_SOURCE_ROW_INDEX'
        assert 0 <= row < len(frame), 'SOURCE_ROW_INDEX_OUT_OF_RANGE'
        canonical = frame.iloc[row]
        assert tuple((int(r[k]) for k in ['uid', 'iid', 'label'])) == tuple((int(canonical[k]) for k in ['uid', 'iid', 'label'])), 'CANONICAL_TARGET_MISMATCH'
        m = PATTERN.search(sample_id(r))
        assert m is not None, 'MISSING_EXPLICIT_EVENT_ID'
        assert int(m.group(1)) == row and int(m.group(4)) == int(canonical['timestamp']), 'EVENT_TIMESTAMP_OR_ROW_MISMATCH'
        index[row] = r
    assert set(index) == set(range(len(frame))), 'INCOMPLETE_SOURCE_ROWS'
    seen = set()
    for r in chain:
        row = int(r['row_index'])
        assert row not in seen, 'DUPLICATE_CHAIN_ROW'
        seen.add(row)
        assert row in index, 'CHAIN_ROW_MISSING_PLAIN'
        assert sample_id(r) == sample_id(index[row]), 'CHAIN_EVENT_ID_MISMATCH'
        assert tuple((int(r[k]) for k in ['uid', 'iid', 'label'])) == tuple((int(index[row][k]) for k in ['uid', 'iid', 'label'])), 'CHAIN_TARGET_MISMATCH'
        index[row] = r
    return [index[row] for row in range(len(frame))]
