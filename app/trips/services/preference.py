from math import sqrt

from app.trips.schemas.schemas import E_Preference, Member_Survey

# 카테고리별 문항 인덱스 매핑 (survey_result의 위치 기반 접근에 사용)
CATEGORY_INDEX_MAP: dict[E_Preference, tuple[int, int]] = {
    E_Preference.HISTORY_CULTURE: (0, 3),
    E_Preference.NATURE_HEALING: (3, 6),
    E_Preference.FOOD: (6, 9),
    E_Preference.ACTIVITY: (9, 12),
    E_Preference.CONVENIENCE_SHOPPING: (12, 15),
}

# 표준편차 페널티 가중치 (잠정)
WEIGHT: float = 0.5

def _calculate_category_score(members: list[Member_Survey], category: E_Preference) -> float:
    if not members: return 0.0

    start, end = CATEGORY_INDEX_MAP[category]

    per_member_avg: list[float] = []
    for member in members:
        scores = member.survey_result[start:end]
        per_member_avg.append(sum(scores) / len(scores))

    mean = sum(per_member_avg) / len(per_member_avg) # 평균
    std = 0.0 if len(per_member_avg) == 1 else sqrt(sum((x - mean) ** 2 for x in per_member_avg) / len(per_member_avg))  # 표준편차

    return mean - std * WEIGHT


def calculate_group_preference(members: list[Member_Survey]) -> dict[E_Preference, float]:
    return {category: _calculate_category_score(members, category) for category in E_Preference}
