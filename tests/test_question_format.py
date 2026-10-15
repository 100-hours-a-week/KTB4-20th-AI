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
    assert rule_grade(QuestionAnswer(**{"target": None, "condition": None, "place": None, **answer})) == grade


@pytest.mark.parametrize(
    "answer",
    [
        '{"judgeable": "yes"}',  # 칸을 빼먹음
        '{"judgeable": "yes", "target": null, "condition": null, "place": "yes"}',
        '{"judgeable": "yes", "target": "enough", "condition": "na", "place": null}',
        '{"judgeable": "yes", "target": "short", "condition": null, "place": "yes"}',
    ],
)
def test_missing_answers_are_format_errors_not_success(answer):
    # 측정 때 모델이 칸을 빼먹거나 null로 두면 규칙상 "성공"이 된다. 형식 오류로 막아 "판정 없음"으로 세게 한다
    with pytest.raises(ValueError):
        QuestionAnswer.model_validate_json(answer)


def test_schema_requires_all_four_fields():
    # 서버(vLLM)·Gemini가 스키마로 답을 막을 때 네 칸을 모두 쓰게 한다
    assert set(QuestionAnswer.model_json_schema()["required"]) == {"judgeable", "target", "condition", "place"}


def test_normalize_blanks_questions_that_cannot_be_asked():
    # 판단 불가면 나머지를, 대상이 없으면 조건을 물을 수 없다. 1차 파일럿은 대상 없음에도 Q3 답이 갈려 있었다
    assert normalize({"q1": "no", "q2": "none", "q3": "no", "q4": "na"}) == QuestionAnswer(
        judgeable="no", target=None, condition=None, place=None)
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


def test_v2_prompt_changes_only_the_place_question():
    # 한 번에 하나만 바꿔 재야 효과를 알 수 있다. 장소 질문 앞뒤는 v1과 같아야 한다
    from scripts.question_format import (
        QUESTION_SYSTEM_PROMPT,
        QUESTION_SYSTEM_PROMPT_V2,
    )

    head, tail = QUESTION_SYSTEM_PROMPT.split("[place]")
    head_v2, tail_v2 = QUESTION_SYSTEM_PROMPT_V2.split("[place]")
    assert head_v2 == head
    assert tail_v2 != tail
    assert tail.splitlines()[-1] == tail_v2.splitlines()[-1]  # "na" 줄은 그대로
    assert "확신" not in tail_v2
