from datetime import date

from app.trips.schemas.schemas import E_Preference


def _distribute_by_preference_rank(
    remaining: int, preferences: dict[E_Preference, float]
) -> dict[E_Preference, int]:
    """취향 순위(1위 많이, 순위 낮으면 적게)에 따라 remaining 슬롯을 배분. FOOD 제외."""
    result: dict[E_Preference, int] = {c: 0 for c in E_Preference}

    ranked = sorted(
        (c for c in preferences if c != E_Preference.FOOD),
        key=lambda c: preferences[c],
        reverse=True,
    )
    weights = [len(ranked) - i for i in range(len(ranked))]
    total_weight = sum(weights)

    allocated = 0
    for category, weight in zip(ranked, weights):
        count = (remaining * weight) // total_weight
        result[category] = count
        allocated += count

    leftover = remaining - allocated
    for category in ranked[:leftover]:
        result[category] += 1

    return result


def allocate_slots_by_category(
    total_days: int, preferences: dict[E_Preference, float]
) -> dict[E_Preference, int]:
    full_days = total_days - 1

    avg_score = sum(preferences.values()) / len(preferences)
    has_activity = preferences[E_Preference.ACTIVITY] >= avg_score
    slots_per_full_day = 6 if has_activity else 5
    total_slots = full_days * slots_per_full_day + 2

    food_slots = full_days * 2 + 1
    remaining = total_slots - food_slots

    result = _distribute_by_preference_rank(remaining, preferences)
    result[E_Preference.FOOD] = food_slots
    return result


def allocate_slots_for_single_day(preferences: dict[E_Preference, float]) -> dict[E_Preference, int]:
    avg_score = sum(preferences.values()) / len(preferences)
    has_activity = preferences[E_Preference.ACTIVITY] >= avg_score
    slots_per_day = 6 if has_activity else 5

    remaining = slots_per_day - 2  # FOOD(점심+저녁) 제외한 나머지

    result = _distribute_by_preference_rank(remaining, preferences)
    result[E_Preference.FOOD] = 2
    return result

def allocate_slots(
        start_date: str, end_date: str, preferences: dict[E_Preference, float]
) -> dict[E_Preference, int]:
    """여행 기간에 따라 하루 전용/다일 계산을 선택"""
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    total_days = (end - start).days + 1

    if total_days == 1:
        return allocate_slots_for_single_day(preferences)
    return allocate_slots_by_category(total_days, preferences)