from math import sqrt

from app.trips.schemas.schemas import Member_Survey

# 표준편차 페널티 가중치 (잠정)
WEIGHT: float = 0.5


def calculate_per_member_averages(
    members: list[Member_Survey], index_range: tuple[int, int]
) -> list[tuple[str, float]]:
    """문항 인덱스 범위 하나에 대해, 각 멤버의 (user_id, 개인 평균 점수) 쌍을 계산."""
    start, end = index_range

    per_member: list[tuple[str, float]] = []
    for member in members:
        scores = member.survey_result[start:end]
        avg = sum(scores) / len(scores)
        per_member.append((member.user.user_id, avg))

    return per_member


def calculate_group_score(per_member_averages: list[float]) -> float:
    """그룹 점수 = 평균 - (표준편차 × W)"""
    if not per_member_averages:
        return 0.0

    mean = sum(per_member_averages) / len(per_member_averages)

    std = 0.0 if len(per_member_averages) == 1 else sqrt(
        sum((x - mean) ** 2 for x in per_member_averages) / len(per_member_averages)
    )

    return mean - (std * WEIGHT)