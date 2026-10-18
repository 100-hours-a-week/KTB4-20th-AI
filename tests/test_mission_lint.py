import pytest

from app.photomissions.prompts import MISSION_SYSTEM_PROMPT
from scripts.mission_lint import count_failed, summarize, warnings


@pytest.mark.parametrize(
    "description",
    [
        "기와지붕 크게 찍기",
        "석가탑 돌 가까이 찍기",
        "빨간 등대 앞에서 찍기",
        "각자 음료 컵 들고 찍기",
        "경주타워 앞에서 다 같이 찍기",
        "바다 배경으로 다 같이 찍기",
        "테이블 위 음식 찍기",
        "서점에 진열된 책 찍기",
    ],
)
def test_allowed_missions_have_no_warning(description):
    assert warnings(description) == []


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        # 평가셋 B에서 실제로 나온 문구
        ("각자 놀이기구 앞에서 점프하며 찍기", ["동작·과정"]),
        ("경주타워 앞에서 다 같이 만세하기", ["동작·과정"]),
        ("숟가락으로 디저트 뜨는 모습 찍기", ["동작·과정"]),
        ("동물원 벤치에 앉아서 찍기", ["동작·과정"]),
        ("각자 마음에 드는 물건 찍기", ["마음·선택"]),
        ("각자 고른 책 표지 찍기", ["마음·선택"]),
        ("길가에 핀 꽃 찍기", ["계절"]),
        ("다 같이 맥주잔 부딪히는 모습 찍기", ["동작·과정", "다 같이 형태"]),
        ("한복 입고 다 같이 찍기", ["다 같이 형태"]),
        ("다같이 파도 치는 바다 찍기", ["다 같이 형태"]),
        # 처음 버전이 놓친 흔한 변형
        ("붕어빵 먹으며 찍기", ["동작·과정"]),
        ("한복 착용한 모습 찍기", ["동작·과정"]),
        ("계단 뛰어오르며 찍기", ["동작·과정"]),
        ("벽화 앞에서 서서 찍기", ["동작·과정"]),
        ("각자 선택한 빵 찍기", ["마음·선택"]),
        ("직접 구매한 기념품 찍기", ["마음·선택"]),
    ],
)
def test_suspicious_missions_are_flagged(description, expected):
    assert warnings(description) == expected


@pytest.mark.parametrize("description", ["흔들다리 찍기", "자동차 앞모습 찍기"])
def test_object_words_are_not_actions(description):
    # "흔들"·"모습"만 보면 흔들다리, 앞모습도 동작으로 걸린다
    assert warnings(description) == []


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        # 알고 있는 오탐. 경고용이라 그대로 두고 사람이 걸러 낸다
        ("단풍나무 간판 찍기", ["계절"]),
        ("열차 지나가는 모습 찍기", ["동작·과정"]),
        ("고분 하나 골라서 찍기", ["마음·선택"]),
    ],
)
def test_known_false_positives_stay_visible(description, expected):
    assert warnings(description) == expected


def test_summary_counts_duplicates_inside_a_batch_separately():
    rows = [
        {"batch": "b0", "mission_description": "테이블 위 음식 찍기", "scope": "GROUP"},
        {"batch": "b0", "mission_description": "테이블 위 음식 찍기", "scope": "GROUP"},
        {"batch": "b1", "mission_description": "테이블 위 음식 찍기", "scope": "GROUP"},
        {"batch": "b1", "mission_description": "각자 마음에 드는 물건 찍기", "scope": "PERSONAL"},
    ]
    summary = summarize(rows, "mission_description")
    assert summary["any_warning"] == 1
    assert summary["by_warning"] == {"마음·선택": 1}
    assert summary["dup_in_batch"] == 1
    assert summary["dup_overall"] == 2
    assert summary["scope"] == {"GROUP": 3, "PERSONAL": 1}


# 프롬프트에 완성된 미션 문장으로 나오는 긍정 예시. 프롬프트를 고치면 이 목록도 맞춘다
PROMPT_POSITIVE_EXAMPLES = [
    "음료 컵 들고 찍기",
    "빨간 등대 앞에서 찍기",
    "석가탑 돌 가까이 찍기",
    "각자 음료 들고 찍기",
    "기와지붕 담기",
    "골목 사진 찍기",
]


def test_positive_prompt_examples_have_no_lint_warning():
    # 긍정 예시가 규칙에 걸리면 모델이 그 예시를 따라 한다. 검사기가 놓치는 위반은 이 테스트도 놓친다
    good = [
        description.strip()
        for line in MISSION_SYSTEM_PROMPT.splitlines()
        if line.strip().startswith("- 좋음:")
        for description in line.split(":", 1)[1].split("/")
    ]
    assert good
    for description in PROMPT_POSITIVE_EXAMPLES:
        assert description in MISSION_SYSTEM_PROMPT, description
    for description in good + PROMPT_POSITIVE_EXAMPLES:
        assert warnings(description) == [], description


def test_failed_count_ignores_batches_that_succeeded_on_rerun():
    rows = [
        {"photo": "a.jpg", "error": "502"},  # 처음 실패, 다시 실행해 성공
        {"photo": "b.jpg", "error": "502"},  # 끝내 실패
        {"photo": "a.jpg", "mission_description": "기와지붕 크게 찍기"},
    ]
    assert count_failed(rows) == 1
