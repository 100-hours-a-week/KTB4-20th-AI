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