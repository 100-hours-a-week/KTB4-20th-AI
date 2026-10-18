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


def generate_itinerary(
    header: list[int],
    slot_candidates: list[list[str]],
    codes: dict[str, Sequence[int]],
    next_logprobs: NextLogprobs,
    width: int = BEAM_WIDTH,
    first_k: int = 1,
    seed: int | None = None,
) -> list[str | None]:
    """슬롯마다 고른 google_place_id를 돌려준다. 후보가 없는 슬롯은 None.

    first_k >= 2이면 첫 슬롯은 모델 점수 상위 first_k곳 중 하나를 random.Random(seed)로 고른다. 시드가 같으면 결과가 같다.
    """
    first_k = max(1, min(int(first_k), MAX_FIRST_K))
    rng = random.Random(seed if seed is not None else 0)
    context = list(header)
    used: set[str] = set()
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


def preference_header(top_categories: Sequence[str], region: str) -> list[int]:
    """서비스 취향(그룹 점수 상위 2개 카테고리)을 학습 때와 같은 0/1 토큰으로 바꾼다."""
    from scripts.semantic_id.tokens import CATEGORIES

    preference = {c: int(c in top_categories) for c in CATEGORIES}
    return header_tokens(preference, region)
