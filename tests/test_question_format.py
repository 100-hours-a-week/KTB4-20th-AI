import json

import pytest

from scripts.question_format import QuestionAnswer, normalize, rule_grade, to_example


@pytest.mark.parametrize(
    ("answer", "grade"),
    [
        ({"judgeable": "no"}, "fail"),
        ({"judgeable": "yes", "target": "enough", "condition": "yes", "place": "no"}, "fail"),
        ({"judgeable": "yes", "target": "none", "place": "na"}, "fail"),
        ({"judgeable": "yes", "target": "short", "condition": "yes", "place": "yes"}, "retry"),
        ({"judgeable": "yes", "target": "enough", "condition": "no", "place": "na"}, "retry"),
        ({"judgeable": "yes", "target": "enough", "condition": "na", "place": "unknown"}, "success"),
    ],
)
def test_rule_grade_matches_labeling_screen(answer, grade):
    # 표시 화면(index.html)의 ruleGrade와 같은 규칙이어야 한다
    assert rule_grade(QuestionAnswer(**answer)) == grade


def test_normalize_blanks_questions_that_cannot_be_asked():
    # 판단 불가면 나머지를, 대상이 없으면 조건을 물을 수 없다. 1차 파일럿은 대상 없음에도 Q3 답이 갈려 있었다
    assert normalize({"q1": "no", "q2": "none", "q3": "no", "q4": "na"}) == QuestionAnswer(judgeable="no")
    assert normalize({"q1": "yes", "q2": "none", "q3": "no", "q4": "no"}) == QuestionAnswer(
        judgeable="yes", target="none", condition=None, place="no")


def test_to_example_is_sharegpt_with_image_and_service_style_text():
    mission = {"mission_id": "p.jpg#other", "photo": "p.jpg", "place": "청와대 본관", "target_place": "창덕궁",
               "mission": "인정전 건물 정면 찍기"}
    label = {"q1": "yes", "q2": "short", "q3": "no", "q4": "unknown"}
    ex = to_example(mission, label, "images/p.jpg")
    assert ex["images"] == ["images/p.jpg"]
    user, assistant = ex["messages"]
    # 모델은 사진을 찍은 실제 장소(청와대)가 아니라 미션의 목표 장소(창덕궁)만 받는다. 서비스와 같다
    assert "목표 장소: 창덕궁" in user["content"] and "청와대" not in user["content"]
    assert user["content"].count("<image>") == 1
    assert json.loads(assistant["content"]) == {"judgeable": "yes", "target": "short", "condition": "no", "place": "unknown"}


def test_normalize_refuses_missing_answers_instead_of_passing_as_success():
    # Q2가 비어 있으면 규칙상 아무 조건에도 안 걸려 성공이 된다. 틀린 정답이 조용히 들어가면 안 된다
    with pytest.raises(ValueError):
        normalize({"mission_id": "x#pw", "q1": "yes", "q2": None, "q3": None, "q4": "na"})
