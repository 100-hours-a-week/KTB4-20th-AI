from datetime import date
from app.trips.schema.schemas import E_Preference

def allocate_slots_by_category(start_date: str, end_date: str, preferences: dict[E_Preference, float]) -> dict[E_Preference, int]:
    # 시작일, 종료일로 여행 일수 계산
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    total_days = (end - start).days + 1
    full_days = total_days - 1 # 마지막 날은 점심까지만

    # 하루당 슬롯 수를 곱해 전체 슬롯 개수 계산("오전, 점심, 오후 1, 오후 2, 저녁", activity가 있을 경우 "저녁 이후 일정 1" 추가)
    # activity가 있을 경우 "저녁 이후 일정 1개" 추가
    # 있는지 여부 판단 기준: 평균 점수보다 activity가 높을 경우로 판단
    avg_score = sum(preferences.values()) / len(preferences)
    has_activity = preferences[E_Preference.ACTIVITY] >= avg_score
    slots_per_full_day = 6 if has_activity else 5
    total_slots = full_days * slots_per_full_day + 2

    # FOOD는 항상 하루 최소 몇 개(점심, 저녁) 고정 배정
    food_slots = full_days * 2 + 1
    remaining = total_slots - food_slots

    # 나머지 슬롯을 취향 점수 순위에 따라 배정(1위가 가장 많이, 순위가 낮을수록 적게)
    ranked = sorted(
        (c for c in preferences if c != E_Preference.FOOD),
        key=lambda c: preferences[c],
        reverse=True,
    )
    weights = [len(ranked) - i for i in range(len(ranked))]
    total_weight = sum(weights)

    # 카테고리별 배정 개수를 딕셔너리로 반환
    result: dict[E_Preference, int] = {c: 0 for c in E_Preference}
    result[E_Preference.FOOD] = food_slots

    allocated = 0
    for category, weight in zip(ranked, weights):
        count = (remaining * weight) // total_weight
        result[category] = count
        allocated += count

    leftover = remaining - allocated
    for category in ranked[:leftover]:
        result[category] += 1

    return result