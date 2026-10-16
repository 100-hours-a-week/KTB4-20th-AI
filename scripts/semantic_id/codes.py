"""Semantic ID 코드 묶음을 다루는 순수 파이썬 도구 (M2 T2-4, T2-5, ADR-0003).

RQ-VAE가 낸 코드 묶음(3단계)이 같은 장소끼리 같은지, 서로 다른 장소가 같은 묶음을 받는지(충돌)를 다룬다.
충돌한 장소는 구분 번호를 하나 더 붙인다(TIGER 방식): (c1, c2, c3) → (c1, c2, c3, n).
"""

import collections
import random

Code = tuple[int, ...]


def assign_collision_ids(codes_by_key: dict[str, Code]) -> dict[str, Code]:
    """같은 코드 묶음을 가진 장소마다 0부터 구분 번호를 붙인다. 번호는 키 순서로 정해져 항상 같다."""
    groups: dict[Code, list[str]] = collections.defaultdict(list)
    for key in sorted(codes_by_key):
        groups[tuple(codes_by_key[key])].append(key)
    out: dict[str, Code] = {}
    for code, keys in groups.items():
        for n, key in enumerate(keys):
            out[key] = (*code, n)
    return out


def collision_stats(codes_by_key: dict[str, Code]) -> dict:
    groups = collections.Counter(tuple(c) for c in codes_by_key.values())
    n = len(codes_by_key)
    colliding = sum(size for size in groups.values() if size > 1)
    return {
        "places": n,
        "distinct_codes": len(groups),
        "colliding_places": colliding,
        "colliding_ratio": colliding / n if n else 0.0,
        "max_group": max(groups.values()) if groups else 0,
    }


def usage_stats(codes: list[Code], levels: int, codebook_size: int) -> list[dict]:
    # 단계별 사용 코드 수와 균형(엔트로피 기반 퍼플렉시티)
    import math

    out = []
    n = len(codes)
    for lvl in range(levels):
        counts = collections.Counter(c[lvl] for c in codes)
        probs = [v / n for v in counts.values()]
        entropy = -sum(p * math.log(p) for p in probs)
        out.append({"level": lvl + 1, "used": len(counts), "of": codebook_size, "perplexity": round(math.exp(entropy), 1)})
    return out


def prefix_agreement(pairs_codes: list[tuple[Code, Code]], depths: tuple[int, ...] = (1, 2, 3)) -> dict[int, float]:
    # 쌍마다 앞 d개 코드가 같은 비율
    if not pairs_codes:
        return {d: 0.0 for d in depths}
    return {d: sum(a[:d] == b[:d] for a, b in pairs_codes) / len(pairs_codes) for d in depths}


def random_pair_agreement(
    codes_by_key: dict[str, Code], samples: int = 100_000, seed: int = 20261016, depths: tuple[int, ...] = (1, 2, 3)
) -> dict[int, float]:
    # 우연 수준: 서로 다른 DB 장소 두 개를 무작위로 뽑았을 때 앞 d개 코드가 같은 비율
    keys = sorted(codes_by_key)
    rng = random.Random(seed)
    pairs = []
    for _ in range(samples):
        a, b = rng.sample(keys, 2)
        pairs.append((tuple(codes_by_key[a]), tuple(codes_by_key[b])))
    return prefix_agreement(pairs, depths)
