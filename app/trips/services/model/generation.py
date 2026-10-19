"""학습한 모델로 하루 일정을 슬롯 순서대로 만든다 (ADR-0003).

슬롯마다 후보 장소(지역, 카테고리, 품질 필터, 회피조건이 이미 적용된 google_place_id 목록)를 받아,
앞 슬롯에서 고른 장소를 문맥에 넣고 후보 제한 빔 서치의 1순위를 고른다. 이미 쓴 장소는 후보에서 뺀다.
첫 슬롯만 예외다(ADR-0007): first_k가 2 이상이면 빔 서치 상위 first_k곳 중 시드로 고정한 무작위 한 곳을 고른다.
CPU 경로(컨테이너)와 RunPod 핸들러가 이 함수를 똑같이 쓴다. 모델 호출만 next_logprobs로 바꿔 끼운다.
"""

import random
from collections.abc import Callable, Sequence

from scripts.semantic_id.decode import beam_search, build_trie
from scripts.semantic_id.tokens import header_tokens, place_tokens

NextLogprobs = Callable[[list[list[int]]], list[Sequence[float]]]

BEAM_WIDTH = 10
MAX_FIRST_K = 100  # 첫 장소 후보 폭 상한(워커가 빔 폭으로 쓰므로 계산량을 제한한다)
MAX_TRIP_DAYS = 9  # 최대 8박 9일 (ADR-0008)
MAX_DAY_SLOTS = 8  # 하루 슬롯 상한. 모델은 하루의 앞 6곳까지만 학습했다


def generate_itinerary(
    header: list[int],
    slot_candidates: list[list[str]],
    codes: dict[str, Sequence[int]],
    next_logprobs: NextLogprobs,
    width: int = BEAM_WIDTH,
    first_k: int = 1,
    seed: int | None = None,
    used: set[str] | None = None,
) -> list[str | None]:
    """슬롯마다 고른 google_place_id를 돌려준다. 후보가 없는 슬롯은 None.

    first_k >= 2이면 첫 슬롯은 모델 점수 상위 first_k곳 중 하나를 random.Random(seed)로 고른다. 시드가 같으면 결과가 같다.
    used를 넘기면 그 집합에 있는 장소는 후보에서 빼고, 고른 장소를 그 집합에 더한다(여러 날에 걸친 중복 방지).
    """
    first_k = max(1, min(int(first_k), MAX_FIRST_K))
    rng = random.Random(seed if seed is not None else 0)
    context = list(header)
    used = set() if used is None else used
    chosen: list[str | None] = []
    for index, candidates in enumerate(slot_candidates):
        pool = {g: codes[g] for g in dict.fromkeys(candidates) if g in codes and g not in used}
        randomize = index == 0 and first_k > 1
        ranking = beam_search(context, build_trie(pool), next_logprobs, first_k if randomize else width) if pool else []
        if not ranking:
            chosen.append(None)
            continue
        gid = rng.choice([g for g, _ in ranking[:first_k]]) if randomize else ranking[0][0]
        used.add(gid)
        context += place_tokens(codes[gid])
        chosen.append(gid)
    return chosen


def generate_trip(
    header: list[int],
    day_candidates: list[list[list[str]]],
    codes: dict[str, Sequence[int]],
    next_logprobs: NextLogprobs,
    width: int = BEAM_WIDTH,
    first_k: int = 1,
    seed: int | None = None,
) -> list[list[str | None]]:
    """여러 날의 일정을 일자 순서대로 만든다 (ADR-0008). 일자마다 문맥을 처음부터 시작하고(모델은 하루 단위로 학습함),
    이미 쓴 장소는 여행 전체에서 후보에서 뺀다. 첫 장소 무작위의 시드는 일자마다 seed + 일자 번호(0부터)다."""
    used: set[str] = set()
    return [
        generate_itinerary(
            header, slots, codes, next_logprobs, width, first_k, None if seed is None else seed + day, used,
        )
        for day, slots in enumerate(day_candidates)
    ]


def resolve_days(pools: object, days: object) -> list[list[list[str]]]:
    """요청의 pools(이름 → 후보 목록)와 days(일자마다 슬롯별 풀 이름)를 일자별 슬롯 후보 목록으로 풀어낸다.
    풀을 한 번만 보내서 요청 크기가 일수에 비례하지 않게 한다. 형식이 틀리면 ValueError."""
    if not isinstance(pools, dict) or not all(
        isinstance(name, str) and isinstance(ids, list) and all(isinstance(g, str) for g in ids) for name, ids in pools.items()
    ):
        raise ValueError("pools는 이름 → google_place_id 목록이어야 함")
    if not isinstance(days, list) or not 1 <= len(days) <= MAX_TRIP_DAYS:
        raise ValueError(f"days는 1~{MAX_TRIP_DAYS}일이어야 함")
    resolved: list[list[list[str]]] = []
    for day in days:
        if not isinstance(day, list) or not 1 <= len(day) <= MAX_DAY_SLOTS:
            raise ValueError(f"하루 슬롯은 1~{MAX_DAY_SLOTS}개여야 함")
        if not all(isinstance(name, str) and name in pools for name in day):
            raise ValueError("days의 슬롯은 pools에 있는 풀 이름이어야 함")
        resolved.append([pools[name] for name in day])
    return resolved


def run_trip_request(payload: dict, codes: dict[str, Sequence[int]], next_logprobs: NextLogprobs) -> dict:
    """RunPod 핸들러의 days 요청을 처리한다. 입력이 틀리면 {"error": ...}를 돌려준다(예외를 던지지 않음)."""
    header = payload.get("header")
    if not isinstance(header, list):
        return {"error": "header가 필요함"}
    try:
        day_candidates = resolve_days(payload.get("pools"), payload.get("days"))
    except ValueError as error:
        return {"error": str(error)}
    seed = payload.get("seed")
    if seed is not None and not isinstance(seed, int):
        return {"error": "seed는 정수여야 함"}
    width = min(int(payload.get("width", BEAM_WIDTH)), 20)
    first_k = max(1, min(int(payload.get("first_k") or 1), MAX_FIRST_K))
    return {"days": generate_trip(header, day_candidates, codes, next_logprobs, width, first_k, seed)}


def preference_header(top_categories: Sequence[str], region: str) -> list[int]:
    """서비스 취향(그룹 점수 상위 2개 카테고리)을 학습 때와 같은 0/1 토큰으로 바꾼다."""
    from scripts.semantic_id.tokens import CATEGORIES

    preference = {c: int(c in top_categories) for c in CATEGORIES}
    return header_tokens(preference, region)
