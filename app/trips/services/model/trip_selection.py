"""학습한 모델로 여러 날 일정을 만든다 (ADR-0008). 하루 단위 모델 경로(selection.py)와 후보 제한, 회피조건, 품질 필터가 같다.

차이는 세 가지다.
  - GPU(또는 CPU) 호출 한 번에 여행 전체를 맡긴다. 후보 풀을 이름 붙여 한 번만 보내고 슬롯은 풀 이름으로 가리킨다.
  - 날씨 신호가 있는 날(outdoor_excluded_days)은 그날의 모든 풀에서 야외 type을 뺀 풀(이름 끝에 _indoor)을 쓴다.
  - diversity면 주간 풀에 3위, 4위 카테고리를 처음부터 더한다(다양성 안). 헤더는 취향 상위 두 카테고리 그대로다.
"""

from collections.abc import Collection

from app.trips.schemas.schemas import (
    E_Breaker,
    E_Preference,
    E_Region,
    Member_Survey,
    Place,
)
from app.trips.services.model.generation import BEAM_WIDTH, preference_header
from app.trips.services.model.inference import InferenceRouter
from app.trips.services.model.selection import DAYTIME_SLOT_COUNT, POOL_LIMIT
from app.trips.services.place_selection import (
    DIRECT_EXCLUDE_MAP,
    NIGHT_ACTIVITY_TYPE_VALUES,
    get_DB_places_by_category,
)
from app.trips.services.preference import find_matched_members
from app.trips.services.slot_plan import E_Time_Slot, Trip_Slot_Plan
from app.trips.services.trip_rule import OUTDOOR_TYPE_VALUES

DIVERSITY_EXTRA_RANKS = 2  # 다양성 안이 취향 상위 둘에 더하는 카테고리 수(3위, 4위)


def select_trip_by_model(
    trip_plan: Trip_Slot_Plan, deal_breakers: list[E_Breaker], region: E_Region, members: list[Member_Survey],
    router: InferenceRouter, outdoor_excluded_days: Collection[int] = frozenset(), diversity: bool = False,
    width: int = BEAM_WIDTH, first_k: int = 1, seed: int | None = None, trace: dict | None = None,
) -> tuple[list[list[Place]], str]:
    """일자별 장소 목록과 추론 경로(gpu·cpu)를 돌려준다. 채우지 못한 슬롯은 건너뛴다(그 일자의 목록이 짧아진다)."""
    direct_excluded: set[str] = set()
    for breaker in deal_breakers:
        direct_excluded |= DIRECT_EXCLUDE_MAP.get(breaker, set())
    ranking = trip_plan.days[0].daytime_category_ranking

    places_by_pool: dict[str, dict[str, Place]] = {}  # 풀 이름 → google_place_id → Place
    category_by_pool: dict[str, dict[str, E_Preference]] = {}  # 주간 풀에서 장소마다 취향 순위가 높은 소속 카테고리

    def excluded_for(extra: set[str], outdoor_excluded: bool) -> set[str]:
        return direct_excluded | extra | (OUTDOOR_TYPE_VALUES if outdoor_excluded else set())

    def load(name: str, outdoor_excluded: bool) -> None:
        if name in places_by_pool:
            return
        kind = name.removesuffix("_indoor")
        pool: dict[str, Place] = {}
        if kind == "evening":
            for place in get_DB_places_by_category(
                E_Preference.ACTIVITY, POOL_LIMIT, region,
                excluded_types=excluded_for(set(), outdoor_excluded), required_types=NIGHT_ACTIVITY_TYPE_VALUES,
            ):
                pool.setdefault(place.id, place)
        elif kind == "food":
            for place in get_DB_places_by_category(
                E_Preference.FOOD, POOL_LIMIT, region, excluded_types=excluded_for(set(), outdoor_excluded),
            ):
                pool.setdefault(place.id, place)
        else:  # daytime
            categories: dict[str, E_Preference] = {}
            used_categories: list[str] = []
            top = 2 + (DIVERSITY_EXTRA_RANKS if diversity else 0)
            for rank, category in enumerate(ranking):
                # 상위 둘(다양성 안은 상위 넷)을 합치고, 회피조건 때문에 주간 슬롯 수보다 적으면 다음 순위를 더한다
                if rank >= top and len(pool) >= DAYTIME_SLOT_COUNT:
                    break
                used_categories.append(category.value)
                extra = NIGHT_ACTIVITY_TYPE_VALUES if category == E_Preference.ACTIVITY else set()
                for place in get_DB_places_by_category(
                    category, POOL_LIMIT, region, excluded_types=excluded_for(extra, outdoor_excluded),
                ):
                    if place.id not in pool:
                        pool[place.id] = place
                        categories[place.id] = category
            category_by_pool[name] = categories
            if trace is not None:
                trace.setdefault("daytime_categories", {})[name] = used_categories
        places_by_pool[name] = pool

    day_pool_names: list[list[str]] = []
    for index, day in enumerate(trip_plan.days):
        outdoor_excluded = index in outdoor_excluded_days
        suffix = "_indoor" if outdoor_excluded else ""
        names: list[str] = []
        for item in day.items:
            if item.time_slot == E_Time_Slot.EVENING:
                names.append("evening" + suffix)
            elif item.category == E_Preference.FOOD:
                names.append("food" + suffix)
            else:
                names.append("daytime" + suffix)
            load(names[-1], outdoor_excluded)
        day_pool_names.append(names)

    top_two = [category.value for category in ranking[:2]]
    header = preference_header(top_two, region.value)
    pools = {name: list(pool) for name, pool in places_by_pool.items()}
    chosen_days, route = router.generate_trip(header, pools, day_pool_names, width, first_k, seed)

    matched: dict[E_Preference, list[str]] = {}
    trip: list[list[Place]] = []
    for names, chosen in zip(day_pool_names, chosen_days, strict=True):
        day_places: list[Place] = []
        for name, gid in zip(names, chosen, strict=True):
            if gid is None:
                continue
            place = places_by_pool[name][gid]
            kind = name.removesuffix("_indoor")
            if kind == "evening":
                category = E_Preference.ACTIVITY
            elif kind == "food":
                category = E_Preference.FOOD
            else:
                category = category_by_pool[name][gid]
            if category not in matched:
                matched[category] = find_matched_members(members, category)
            place.matched_preferences = [category]
            place.selected_for = matched[category]
            day_places.append(place)
        trip.append(day_places)
    return trip, route
