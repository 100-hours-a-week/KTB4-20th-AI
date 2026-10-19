import random
from dataclasses import dataclass
from enum import StrEnum

from app.trips.constants import (
    CATEGORY_INDEX_MAP_FOR_SLOT,
    EVENING_SLOT_THRESHOLD,
    NIGHT_ACTIVITY_INDEX_RANGE,
)
from app.trips.schemas.schemas import E_Preference, Member_Survey
from app.trips.services.group_score import (
    calculate_group_score,
    calculate_per_member_averages,
)


class E_Time_Slot(StrEnum):
    MORNING = "MORNING"
    LUNCH = "LUNCH"
    AFTERNOON_FIRST = "AFTERNOON_FIRST"
    AFTERNOON_SECOND = "AFTERNOON_SECOND"
    DINNER = "DINNER"
    EVENING = "EVENING"


# 주간 슬롯(오전, 오후1, 오후2) 후보 카테고리. FOOD는 점심·저녁 고정이므로 제외
DAYTIME_CANDIDATE_CATEGORIES: tuple[E_Preference, ...] = (
    E_Preference.HISTORY_CULTURE,
    E_Preference.NATURE_HEALING,
    E_Preference.ACTIVITY,
    E_Preference.CONVENIENCE_SHOPPING,
)


@dataclass(frozen=True)
class Slot_Plan_Item:
    time_slot: E_Time_Slot
    category: E_Preference

@dataclass(frozen=True)
class Slot_Plan:
    items: tuple[Slot_Plan_Item, ...]
    daytime_category_ranking: tuple[E_Preference, ...]  # 방식 A 보충 순서 계산용 주간 후보 전체 순위


def calculate_group_preference_for_slot(members: list[Member_Survey]) -> dict[E_Preference, float]:
    """슬롯 배치용: 카테고리별 그룹 점수. 주간형 ACTIVITY 문항(11~12번) 사용"""
    return {
        category: calculate_group_score([
            avg for _, avg in calculate_per_member_averages(members, CATEGORY_INDEX_MAP_FOR_SLOT[category])
        ])
        for category in E_Preference
    }


def calculate_night_activity_group_score(members: list[Member_Survey]) -> float:
    """슬롯 추가용: 야간형 ACTIVITY 문항(10번) 사용"""
    per_member = calculate_per_member_averages(members, NIGHT_ACTIVITY_INDEX_RANGE)
    return calculate_group_score([avg for _, avg in per_member])


def rank_daytime_categories(
    slot_preferences: dict[E_Preference, float], random_generator: random.Random
) -> list[E_Preference]:
    """주간 후보 카테고리를 점수 내림차순으로 정렬. 동점은 무작위 순서."""
    return sorted(
        DAYTIME_CANDIDATE_CATEGORIES,
        key=lambda category: (-slot_preferences[category], random_generator.random()),
    )


def build_slot_plan(
    members: list[Member_Survey], random_generator: random.Random | None = None
) -> Slot_Plan:
    """하루 슬롯 구조 결정: 주간 1위 2개, 2위 1개, FOOD 점심·저녁 고정, 조건 충족 시 저녁 이후 ACTIVITY."""
    random_generator = random_generator or random.Random()

    slot_preferences = calculate_group_preference_for_slot(members)
    daytime_category_ranking = rank_daytime_categories(slot_preferences, random_generator)
    first_category, second_category = daytime_category_ranking[:2]

    slot_plan_items = [
        Slot_Plan_Item(E_Time_Slot.MORNING, first_category),
        Slot_Plan_Item(E_Time_Slot.LUNCH, E_Preference.FOOD),
        Slot_Plan_Item(E_Time_Slot.AFTERNOON_FIRST, first_category),
        Slot_Plan_Item(E_Time_Slot.AFTERNOON_SECOND, second_category),
        Slot_Plan_Item(E_Time_Slot.DINNER, E_Preference.FOOD),
    ]

    if calculate_night_activity_group_score(members) > EVENING_SLOT_THRESHOLD:
        slot_plan_items.append(Slot_Plan_Item(E_Time_Slot.EVENING, E_Preference.ACTIVITY))

    return Slot_Plan(
        items=tuple(slot_plan_items),
        daytime_category_ranking=tuple(daytime_category_ranking),
    )

# 마지막 날은 점심까지다 (ADR-0008). 당일치기(1일)는 마지막 날 규칙을 적용하지 않고 하루 전체 템플릿을 쓴다
LAST_DAY_TIME_SLOTS: frozenset[E_Time_Slot] = frozenset({E_Time_Slot.MORNING, E_Time_Slot.LUNCH})


@dataclass(frozen=True)
class Trip_Slot_Plan:
    days: tuple[Slot_Plan, ...]  # 일자 순서. 마지막 날만 점심까지인 짧은 계획이다


def build_trip_slot_plan(
    members: list[Member_Survey], num_days: int, random_generator: random.Random | None = None, diversity: bool = False,
) -> Trip_Slot_Plan:
    """여행 전체 슬롯 구조 (ADR-0008). 카테고리 순위와 저녁 이후 슬롯 여부는 여행 전체에서 한 번만 정하고, 일자 유형만 다르다.

    마지막 날(2일 이상일 때)은 MORNING, LUNCH만 둔다. 저녁 이후 슬롯은 마지막 날을 뺀 날마다 시도한다(후보가 없으면 선택 단계에서 제거).
    diversity면 오후2(2위 카테고리) 슬롯을 3위, 4위 카테고리로 일자마다 번갈아 바꾼다(첫째 날 3위). 규칙 경로의 다양성 안이다.
    """
    if num_days < 1:
        raise ValueError("num_days는 1 이상이어야 함")
    base = build_slot_plan(members, random_generator)
    ranking = base.daytime_category_ranking
    days: list[Slot_Plan] = []
    for index in range(num_days):
        is_last_day = num_days >= 2 and index == num_days - 1
        items = tuple(
            item for item in base.items if not is_last_day or item.time_slot in LAST_DAY_TIME_SLOTS
        )
        if diversity:
            extra_category = ranking[2 + index % 2]
            items = tuple(
                Slot_Plan_Item(item.time_slot, extra_category) if item.time_slot == E_Time_Slot.AFTERNOON_SECOND else item
                for item in items
            )
        days.append(Slot_Plan(items=items, daytime_category_ranking=ranking))
    return Trip_Slot_Plan(days=tuple(days))
