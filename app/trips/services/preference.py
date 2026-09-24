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

def _calculate_per_member_averages(
    members: list[Member_Survey], category: E_Preference
) -> list[tuple[str, float]]:
    """카테고리 하나에 대해, 각 멤버의 (user_id, 개인 평균 점수) 쌍을 계산.
    calculate_category_score와 find_matched_members가 공유하는 내부 헬퍼."""
    start, end = CATEGORY_INDEX_MAP[category]

    per_member: list[tuple[str, float]] = []
    for member in members:
        scores = member.survey_result[start:end]
        avg = sum(scores) / len(scores)
        per_member.append((member.user.user_id, avg))

    return per_member


def _calculate_category_score(members: list[Member_Survey], category: E_Preference) -> float:
    """카테고리 하나의 최종 점수 = 평균 - (표준편차 × W)"""
    per_member = _calculate_per_member_averages(members, category)

    if not per_member:
        return 0.0

    per_member_avg = [avg for _, avg in per_member]
    mean = sum(per_member_avg) / len(per_member_avg)

    std = 0.0 if len(per_member_avg) == 1 else sqrt(
        sum((x - mean) ** 2 for x in per_member_avg) / len(per_member_avg)
    )

    return mean - (std * WEIGHT)


def find_matched_members(members: list[Member_Survey], category: E_Preference) -> list[str]:
    """카테고리 하나에서, 그룹 평균보다 개인 점수가 높은 멤버들의 user_id 반환."""
    per_member = _calculate_per_member_averages(members, category)

    if not per_member:
        return []

    per_member_avg = [avg for _, avg in per_member]
    mean = sum(per_member_avg) / len(per_member_avg)

    return [user_id for user_id, avg in per_member if avg > mean]


def calculate_group_preference(members: list[Member_Survey]) -> dict[E_Preference, float]:
    return {category: _calculate_category_score(members, category) for category in E_Preference}

