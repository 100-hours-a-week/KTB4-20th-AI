"""
설문 대표 요청값 생성 및 검증 보조 스크립트.

응답값 24개(주간 배치 12 x 저녁 이후 여부 2) 각각에 대해,
해당 응답값을 만들어내는 members(설문 목록)를 생성하고 그룹 점수 공식으로 검증한다.

- 점수 단위: HISTORY_CULTURE, NATURE_HEALING, FOOD, ACTIVITY_NIGHT(문항 10),
  ACTIVITY_DAY(문항 11~12), CONVENIENCE_SHOPPING
- 그룹 점수: 멤버 평균 - (모표준편차 x 0.5)
- 다수 멤버 요청값은 페널티가 없으면(평균만 쓰면) 주간 상위 2개 순위가 달라지는 입력만 채택한다.

실행: python -m tests.trips.tools.generate_representative_surveys
"""

import json
import random
from itertools import permutations
from math import sqrt
from pathlib import Path

# ---------------- 설문 구조 ----------------
HISTORY_CULTURE = "HISTORY_CULTURE"
NATURE_HEALING = "NATURE_HEALING"
FOOD = "FOOD"
ACTIVITY_NIGHT = "ACTIVITY_NIGHT"
ACTIVITY_DAY = "ACTIVITY_DAY"
CONVENIENCE_SHOPPING = "CONVENIENCE_SHOPPING"

# 점수 단위별 문항 인덱스 (survey_result 15개 기준)
QUESTION_INDEXES_BY_UNIT: dict[str, list[int]] = {
    HISTORY_CULTURE: [0, 1, 2],
    NATURE_HEALING: [3, 4, 5],
    FOOD: [6, 7, 8],
    ACTIVITY_NIGHT: [9],
    ACTIVITY_DAY: [10, 11],
    CONVENIENCE_SHOPPING: [12, 13, 14],
}

DAYTIME_CANDIDATES = [HISTORY_CULTURE, NATURE_HEALING, ACTIVITY_DAY, CONVENIENCE_SHOPPING]

STANDARD_DEVIATION_WEIGHT = 0.5
EVENING_THRESHOLD = 3.0
MINIMUM_SCORE = 1
MAXIMUM_SCORE = 5
MEMBER_COUNTS = list(range(1, 9))

RANDOM_SEED = 20260930
MAXIMUM_ATTEMPTS = 200_000

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "representative_surveys.json"


# ---------------- 그룹 점수 계산 (테스트 오라클, preference.py와 독립 구현) ----------------
def calculate_group_score(member_scores: list[float], apply_penalty: bool = True) -> float:
    mean = sum(member_scores) / len(member_scores)
    if not apply_penalty or len(member_scores) == 1:
        return mean
    standard_deviation = sqrt(sum((score - mean) ** 2 for score in member_scores) / len(member_scores))
    return mean - standard_deviation * STANDARD_DEVIATION_WEIGHT


def calculate_group_scores(
    members_unit_scores: list[dict[str, int]], apply_penalty: bool = True
) -> dict[str, float]:
    return {
        unit: calculate_group_score([member[unit] for member in members_unit_scores], apply_penalty)
        for unit in QUESTION_INDEXES_BY_UNIT
    }


def rank_daytime_candidates(group_scores: dict[str, float]) -> list[str]:
    return sorted(DAYTIME_CANDIDATES, key=lambda unit: group_scores[unit], reverse=True)


def has_strict_top_two(group_scores: dict[str, float]) -> bool:
    """1위 > 2위 > 3위가 엄격히 성립하는지. 동점이면 무작위 처리 대상이라 대표 입력에서 제외한다."""
    ranked = rank_daytime_candidates(group_scores)
    first, second, third = (group_scores[unit] for unit in ranked[:3])
    return first > second > third


# ---------------- 후보 생성 ----------------
def sample_member_unit_scores(
    random_generator: random.Random, first: str, second: str, has_evening: bool
) -> dict[str, int]:
    """목표 방향으로 편향된 범위에서 멤버 1명의 점수 단위를 샘플링한다."""
    unit_scores: dict[str, int] = {}
    for unit in QUESTION_INDEXES_BY_UNIT:
        if unit == first:
            low, high = 4, 5
        elif unit == second:
            low, high = 3, 5
        elif unit == ACTIVITY_NIGHT:
            low, high = (4, 5) if has_evening else (1, 3)
        else:
            low, high = MINIMUM_SCORE, MAXIMUM_SCORE
        unit_scores[unit] = random_generator.randint(low, high)
    return unit_scores


def to_survey_result(unit_scores: dict[str, int]) -> list[int]:
    survey_result = [0] * 15
    for unit, question_indexes in QUESTION_INDEXES_BY_UNIT.items():
        for question_index in question_indexes:
            survey_result[question_index] = unit_scores[unit]
    return survey_result


def is_valid_representative(
    members_unit_scores: list[dict[str, int]], first: str, second: str, has_evening: bool
) -> bool:
    group_scores = calculate_group_scores(members_unit_scores)

    if not has_strict_top_two(group_scores):
        return False
    if rank_daytime_candidates(group_scores)[:2] != [first, second]:
        return False
    if (group_scores[ACTIVITY_NIGHT] > EVENING_THRESHOLD) != has_evening:
        return False

    # 다수 멤버: 페널티 없이 계산하면 주간 상위 2개가 동점 없이 달라지는 입력만 채택
    # (평균만으로 동점이 되면 동점 무작위 처리에 따라 우연히 기대값과 같아질 수 있으므로 제외)
    if len(members_unit_scores) > 1:
        mean_only_scores = calculate_group_scores(members_unit_scores, apply_penalty=False)
        if not has_strict_top_two(mean_only_scores):
            return False
        if rank_daytime_candidates(mean_only_scores)[:2] == [first, second]:
            return False

    return True


def generate_representative(
    random_generator: random.Random, first: str, second: str, has_evening: bool, member_count: int
) -> list[dict[str, int]]:
    for _ in range(MAXIMUM_ATTEMPTS):
        members_unit_scores = [
            sample_member_unit_scores(random_generator, first, second, has_evening)
            for _ in range(member_count)
        ]
        if is_valid_representative(members_unit_scores, first, second, has_evening):
            return members_unit_scores
    raise RuntimeError(
        f"대표 입력 생성 실패: first={first}, second={second}, "
        f"has_evening={has_evening}, member_count={member_count}"
    )


def generate_representative_surveys() -> list[dict]:
    random_generator = random.Random(RANDOM_SEED)
    targets = [
        (first, second, has_evening)
        for has_evening in (False, True)
        for first, second in permutations(DAYTIME_CANDIDATES, 2)
    ]

    cases = []
    for case_index, (first, second, has_evening) in enumerate(targets):
        member_count = MEMBER_COUNTS[case_index % len(MEMBER_COUNTS)]
        members_unit_scores = generate_representative(
            random_generator, first, second, has_evening, member_count
        )
        cases.append({
            "case_id": f"{first}__{second}__{'evening' if has_evening else 'no_evening'}",
            "expected_first": first,
            "expected_second": second,
            "expected_has_evening": has_evening,
            "members": [
                {
                    "user": {"user_id": f"user_{member_index + 1}"},
                    "survey_result": to_survey_result(unit_scores),
                    "deal_breakers": [],
                }
                for member_index, unit_scores in enumerate(members_unit_scores)
            ],
        })
    return cases


if __name__ == "__main__":
    representative_surveys = generate_representative_surveys()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(representative_surveys, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(representative_surveys)}개 생성 완료: {OUTPUT_PATH}")