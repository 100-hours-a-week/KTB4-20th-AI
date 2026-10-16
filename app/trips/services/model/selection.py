"""학습한 모델로 장소를 고른다 (ADR-0003). 규칙 경로(place_selection.py)와 같은 입력·출력 모양이다.

슬롯 구조(점심·저녁 FOOD, 저녁 이후 ACTIVITY)와 회피조건, 품질 필터는 규칙 경로와 같은 DB 조회를 그대로 쓴다.
모델은 슬롯마다 후보 중 무엇을 고를지만 정한다. 후보 밖 장소는 낼 수 없다.
"""

import random

from app.trips.schemas.schemas import (
    E_Breaker,
    E_Preference,
    E_Region,
    Member_Survey,
    Place,
)
from app.trips.services.model.generation import BEAM_WIDTH, preference_header
from app.trips.services.model.inference import InferenceRouter
from app.trips.services.place_selection import (
    DIRECT_EXCLUDE_MAP,
    NIGHT_ACTIVITY_TYPE_VALUES,
    get_DB_places_by_category,
)
from app.trips.services.preference import find_matched_members
from app.trips.services.slot_plan import E_Time_Slot, Slot_Plan

POOL_LIMIT = 3000  # 지역·카테고리 후보를 사실상 전부 가져온다
TIE_SEED = 0  # 취향 동점일 때 카테고리 순서를 고정한다(같은 입력은 같은 결과)


def model_slot_plan_seed() -> random.Random:
    return random.Random(TIE_SEED)


def select_places_by_model(
    slot_plan: Slot_Plan, deal_breakers: list[E_Breaker], region: E_Region, members: list[Member_Survey],
    router: InferenceRouter, width: int = BEAM_WIDTH,
) -> tuple[list[Place], str]:
    direct_excluded: set[str] = set()
    for breaker in deal_breakers:
        direct_excluded |= DIRECT_EXCLUDE_MAP.get(breaker, set())

    pools: dict[tuple[str, bool], list[Place]] = {}

    def pool_for(category: E_Preference, evening: bool) -> list[Place]:
        key = (category.value, evening)
        if key not in pools:
            if evening:
                pools[key] = get_DB_places_by_category(
                    E_Preference.ACTIVITY, POOL_LIMIT, region, excluded_types=direct_excluded, required_types=NIGHT_ACTIVITY_TYPE_VALUES,
                )
            else:
                excluded = direct_excluded | NIGHT_ACTIVITY_TYPE_VALUES if category == E_Preference.ACTIVITY else direct_excluded
                pools[key] = get_DB_places_by_category(category, POOL_LIMIT, region, excluded_types=excluded)
        return pools[key]

    slot_categories: list[tuple[E_Preference, bool]] = [
        (item.category, item.time_slot == E_Time_Slot.EVENING) for item in slot_plan.items
    ]
    slot_pools = [pool_for(category, evening) for category, evening in slot_categories]
    top_two = [c.value for c in slot_plan.daytime_category_ranking[:2]]
    header = preference_header(top_two, region.value)

    chosen, route = router.generate(header, [[p.id for p in pool] for pool in slot_pools], width)

    places: list[Place] = []
    matched: dict[E_Preference, list[str]] = {}
    for gid, (category, _), pool in zip(chosen, slot_categories, slot_pools, strict=True):
        if gid is None:
            continue
        place = next(p for p in pool if p.id == gid)
        if category not in matched:
            matched[category] = find_matched_members(members, category)
        place.matched_preferences = [category]
        place.selected_for = matched[category]
        places.append(place)
    return places, route
