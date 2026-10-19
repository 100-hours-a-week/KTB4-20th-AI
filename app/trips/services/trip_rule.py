"""여러 날 일정의 규칙 경로 (ADR-0008). 하루 규칙 경로(place_selection.select_places_by_slot_plan)와 같은 DB 조회·보충 규칙을 쓴다.

- 이미 쓴 장소는 여행 전체에서 뺀다. 한 안(호출 한 번) 안에서만 막고 안 사이는 막지 않는다.
- 날씨 신호가 있는 날(outdoor_excluded_days)은 그날의 모든 후보에서 야외 type을 뺀다. 야외 목록에는 NATURE_HEALING의 type이 모두 들어 있어서
  그날은 NATURE_HEALING 후보가 비고, 주간 슬롯은 하루 규칙 경로와 같이 다음 순위 카테고리로 보충된다.
- variant_seed가 없으면 후보 앞에서부터 고른다(안 [0], 평점순). 있으면 앞에서부터 top_k곳 중 시드로 고른다(안 [1], [2]의 규칙 변형).
"""

import random
from collections import Counter
from collections.abc import Collection

from app.trips.schemas.schemas import (
    E_Breaker,
    E_Preference,
    E_Region,
    Member_Survey,
    Place,
)
from app.trips.services.place_selection import (
    DIRECT_EXCLUDE_MAP,
    NIGHT_ACTIVITY_TYPE_VALUES,
    POOL_SPARE,
    _build_fallback_order,
    get_DB_places_by_category,
)
from app.trips.services.preference import find_matched_members
from app.trips.services.slot_plan import E_Time_Slot, Trip_Slot_Plan

OUTDOOR_TYPE_VALUES: frozenset[str] = frozenset(DIRECT_EXCLUDE_MAP[E_Breaker.OUTDOOR_ACTIVITY])
RULE_VARIANT_TOP_K = 10  # 규칙 변형이 후보 앞에서부터 이만큼 중에서 시드로 고른다(잠정값, 안 사이 차이를 재서 조정)


def select_trip_by_slot_plan(
    trip_plan: Trip_Slot_Plan,
    deal_breakers: list[E_Breaker],
    region: E_Region,
    members: list[Member_Survey],
    outdoor_excluded_days: Collection[int] = frozenset(),
    variant_seed: int | None = None,
    top_k: int = RULE_VARIANT_TOP_K,
) -> list[list[Place]]:
    """일자별 장소 목록을 돌려준다. 후보가 없어 채우지 못한 슬롯은 건너뛴다(그 일자의 목록이 짧아진다)."""
    direct_excluded: set[str] = set()
    for breaker in deal_breakers:
        direct_excluded |= DIRECT_EXCLUDE_MAP.get(breaker, set())
    rng = random.Random(variant_seed) if variant_seed is not None else None
    pick_width = max(1, top_k) if rng is not None else 1

    slot_counts: Counter[str] = Counter()
    for day in trip_plan.days:
        for item in day.items:
            if item.time_slot == E_Time_Slot.EVENING:
                slot_counts["evening"] += 1
            elif item.category == E_Preference.FOOD:
                slot_counts["food"] += 1
            else:
                slot_counts["daytime"] += 1
    # 카테고리 하나가 여행 전체의 주간 슬롯을 모두 맡을 수 있고(보충 포함), 앞에서부터 고르는 폭과 중복 건너뛰기 여유를 더한다
    limit = {kind: count + pick_width + POOL_SPARE for kind, count in slot_counts.items()}

    pools: dict[tuple[str, bool], list[Place]] = {}

    def pool_for(kind: str, category: E_Preference, outdoor_excluded: bool) -> list[Place]:
        key = (category.value if kind == "daytime" else kind, outdoor_excluded)
        if key not in pools:
            excluded = set(direct_excluded)
            if outdoor_excluded:
                excluded |= OUTDOOR_TYPE_VALUES
            required = None
            if kind == "evening":
                required = NIGHT_ACTIVITY_TYPE_VALUES
            elif category == E_Preference.ACTIVITY:
                excluded |= NIGHT_ACTIVITY_TYPE_VALUES
            pools[key] = get_DB_places_by_category(
                category, limit.get(kind, pick_width + POOL_SPARE), region, excluded_types=excluded, required_types=required,
            )
        return pools[key]

    matched_members_by_category: dict[E_Preference, list[str]] = {}

    def assign(place: Place, category: E_Preference) -> Place:
        if category not in matched_members_by_category:
            matched_members_by_category[category] = find_matched_members(members, category)
        place.matched_preferences = [category]
        place.selected_for = matched_members_by_category[category]
        return place

    used_place_ids: set[str] = set()

    def take(pool: list[Place]) -> Place | None:
        # 앞에서부터 아직 안 쓴 장소를 pick_width곳 모아, 그중 하나를 고른다(변형이 아니면 첫 번째)
        candidates: list[Place] = []
        for place in pool:
            if place.id not in used_place_ids:
                candidates.append(place)
                if len(candidates) == pick_width:
                    break
        if not candidates:
            return None
        chosen = rng.choice(candidates) if rng is not None else candidates[0]
        used_place_ids.add(chosen.id)
        return chosen

    trip: list[list[Place]] = []
    for index, day in enumerate(trip_plan.days):
        outdoor_excluded = index in outdoor_excluded_days
        day_places: list[Place] = []
        for item in day.items:
            if item.time_slot == E_Time_Slot.EVENING:
                place = take(pool_for("evening", E_Preference.ACTIVITY, outdoor_excluded))
                if place:
                    day_places.append(assign(place, E_Preference.ACTIVITY))
            elif item.category == E_Preference.FOOD:
                place = take(pool_for("food", E_Preference.FOOD, outdoor_excluded))
                if place:
                    day_places.append(assign(place, E_Preference.FOOD))
            else:
                for category in [item.category, *_build_fallback_order(item.category, day.daytime_category_ranking)]:
                    place = take(pool_for("daytime", category, outdoor_excluded))
                    if place:
                        day_places.append(assign(place, category))
                        break
        trip.append(day_places)
    return trip
