import random

def mainline_schedule(chain_count, plain_count):

    def order(seed):
        rng = random.Random(seed)
        c = list(range(chain_count))
        p = list(range(plain_count))
        rng.shuffle(c)
        rng.shuffle(p)
        result = []
        pp = 0
        for pos, index in enumerate(c, 1):
            result.append(('chain', index))
            if pos % 4 == 0:
                result.append(('plain', p[pp % len(p)]))
                pp += 1
        return result
    primary = order(20260710)
    continuation = order(20260709)[:1600]
    assert len(primary) == 41080 and len(continuation) == 1600
    return primary + continuation
