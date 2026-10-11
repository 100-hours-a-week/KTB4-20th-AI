import json
import pathlib

import pytest

from scripts.eval_cases import (
    EvalCase,
    apply_relabel,
    apply_split,
    classify,
    dedupe,
    from_a,
    from_b,
    load_relabel,
    metrics,
    mixed_runs,
)
from scripts.question_format import QuestionAnswer
from scripts.split_eval import split_error


def _case(truth="success", score=95.0, case_type="target_place", **kw):
    base = {
        "eval_set": "B", "photo": "p.jpg", "place": "황리단길", "target": "황리단길", "case_type": case_type,
        "mission_source": "generate", "category": "음식", "truth": truth, "model": "m", "match_score": score,
    }
    return EvalCase(**{**base, **kw})


@pytest.mark.parametrize(
    ("truth", "score", "case_type", "expected"),
    [
        ("success", 95, "target_place", "TP"),
        ("success", 70, "target_place", "FN_LIGHT"),  # 재시도로 막음
        ("success", 20, "target_place", "FN_HEAVY"),  # 실패로 막음
        ("retry", 95, "target_place", "FP_LIGHT"),  # 가려진 사진을 통과
        ("fail", 95, "target_place", "FP"),  # 장소는 맞는데 미션 안 한 사진을 통과
        ("fail", 95, "other_place", "FP_CRITICAL"),
        ("fail", 95, "lookalike", "FP_CRITICAL"),
        ("retry", 20, "target_place", "TN"),  # 재시도·실패끼리 엇갈려도 통과하지 못한 것은 같다
        ("fail", 70, "target_place", "TN"),
        ("fail", 70, "lookalike", "TN"),
    ],
)
def test_classify_each_outcome(truth, score, case_type, expected):
    assert classify(_case(truth, score, case_type)) == expected


def test_threshold_can_be_changed_for_experiments():
    # 개선 ①(임계값 재조정)에서 같은 결과를 다른 기준으로 다시 셀 수 있어야 한다
    case = _case("success", 85)
    assert classify(case) == "FN_LIGHT"
    assert classify(case, success=80) == "TP"


def test_no_answer_is_not_counted_as_an_error_type():
    # 모델이 답을 못 낸 건은 FP·FN 어느 쪽 분모에도 넣지 않고 따로 센다
    m = metrics([_case(score=None), _case("success", 95)])
    assert m["judged"] == 1
    assert m["no_answer"] == 1
    assert m["recall"] == 1.0


def test_each_rate_uses_only_cases_where_that_error_can_happen():
    cases = [
        _case("success", 95), _case("success", 20),  # 정답 성공 2건: TP 1, 무거운 FN 1
        _case("fail", 95), _case("fail", 20), _case("fail", 20), _case("fail", 20),  # 정답 실패 4건: FP 1
        _case("fail", 95, "lookalike"), _case("fail", 20, "lookalike"),  # 닮은 장소 2건: 치명적 FP 1
    ]
    m = metrics(cases)
    assert m["fn_heavy"] == 0.5
    assert m["fp"] == 0.25
    assert m["critical_fp_lookalike"] == 0.5
    assert m["critical_fp_other"] is None  # 다른 장소 케이스가 없으면 비율을 말할 수 없다
    assert m["precision"] == pytest.approx(1 / 3, abs=1e-4)


def test_set_a_negative_missions_are_split_into_lookalike_and_other_place():
    records = [
        {"photo": "w.jpg", "place": "월정교", "target": "동궁과월지", "kind": "negative", "model": "m", "parsed": True, "match_score": 92},
        {"photo": "w.jpg", "place": "월정교", "target": "첨성대", "kind": "negative", "model": "m", "parsed": True, "match_score": 10},
        {"photo": "w.jpg", "place": "월정교", "target": "월정교", "kind": "positive", "model": "m", "parsed": True, "match_score": 95},
        {"photo": "x.jpg", "place": "월정교", "target": "월정교", "kind": "positive", "model": "m", "parsed": True, "match_score": 95},
    ]
    cases = from_a(records, {"w.jpg": "success"})  # x.jpg는 사람 정답이 없어 뺀다
    assert [(c.case_type, c.truth) for c in cases] == [("lookalike", "fail"), ("other_place", "fail"), ("target_place", "success")]


def test_same_case_measured_twice_is_counted_once():
    assert [c.match_score for c in dedupe([_case(), _case(score=10)])] == [95.0]


def test_retried_answer_replaces_earlier_error():
    # 하루 한도(429)로 실패한 줄 뒤에 다시 잰 결과가 붙으면 다시 잰 결과를 쓴다
    assert [c.match_score for c in dedupe([_case(score=None), _case(score=88)])] == [88.0]


def test_relabel_replaces_truth_and_drops_unclear():
    cases = [
        _case("fail", photo="a.jpg"),
        _case("success", photo="b.jpg"),
        _case("fail", photo="c.jpg"),  # 재표시 대상이 아니면 기존 정답 그대로
    ]
    relabel = {("B", "a.jpg"): "retry", ("B", "b.jpg"): "unclear"}
    out = apply_relabel(cases, relabel)
    assert [(c.photo, c.truth) for c in out] == [("a.jpg", "retry"), ("c.jpg", "fail")]


def test_relabel_only_applies_to_target_place_missions():
    # 재표시는 사진을 찍은 장소의 미션만 다시 본 것이다. 다른 장소·닮은 장소 미션은 정답이 항상 실패다
    cases = [
        _case("success", photo="a.jpg", eval_set="A", target="불국사"),
        _case("fail", photo="a.jpg", eval_set="A", target="석굴암", case_type="other_place"),
    ]
    out = apply_relabel(cases, {("A", "a.jpg"): "unclear"})
    assert [(c.target, c.truth) for c in out] == [("석굴암", "fail")]


def test_load_relabel_skips_items_not_yet_labeled(tmp_path):
    path = tmp_path / "relabel_labels.json"
    path.write_text(json.dumps({"items": [
        {"eval_set": "B", "photo": "a.jpg", "label": "retry", "note": ""},
        {"eval_set": "A", "photo": "b.jpg"},  # 아직 고르지 않은 사진
    ]}), encoding="utf-8")
    assert load_relabel(path) == {("B", "a.jpg"): "retry"}


def test_split_keeps_only_photos_on_that_side():
    cases = [_case(photo="a.jpg"), _case(photo="b.jpg")]
    assert [c.photo for c in apply_split(cases, {"b.jpg"})] == ["b.jpg"]


def _answer(target="enough", condition="na", place="yes"):
    return QuestionAnswer(judgeable="yes", target=target, condition=condition, place=place)


def test_question_answer_is_graded_by_rule_without_score():
    # 객관식 결과에는 점수가 없다. 점수가 없다고 "판정 없음"으로 빠지면 안 되고, 답에서 규칙으로 등급을 정한다
    assert classify(_case("success", None, answer=_answer())) == "TP"
    assert classify(_case("success", None, answer=_answer(target="short"))) == "FN_LIGHT"
    assert classify(_case("success", None, answer=_answer(target="none", condition=None))) == "FN_HEAVY"
    assert classify(_case("retry", None, answer=_answer())) == "FP_LIGHT"
    assert metrics([_case("success", None, answer=_answer())])["no_answer"] == 0


def test_question_answer_ignores_score_thresholds():
    # 기준 점수를 바꿔도 객관식 결과는 그대로다 (규칙을 바꿔야 바뀐다)
    case = _case("success", None, answer=_answer(target="short"))
    assert classify(case, success=10, retry=5) == "FN_LIGHT"


def test_from_b_reads_answer_and_regrades_with_current_rule():
    # 기록 당시 grade가 무엇이었든 답 원본에서 지금 규칙으로 다시 정한다
    record = {
        "eval_set": "B_aug", "photo": "a.jpg", "place": "황리단길", "category": "음식", "truth": "success",
        "model": "qwen", "format": "question", "parsed": True, "grade": "fail",
        "answer": {"judgeable": "yes", "target": "enough", "condition": "yes", "place": "unknown"},
    }
    [case] = from_b([record])
    assert case.match_score is None
    assert case.format == "question"
    assert classify(case) == "TP"


def test_results_of_different_models_or_formats_are_not_mixed():
    # 같은 사진이면 하나만 남기므로, 섞어 넣으면 한 모델 결과가 조용히 사라진다
    assert mixed_runs([_case(model="pro"), _case(model="pro")]) is None
    assert mixed_runs([_case(model="pro"), _case(model="qwen")]) is not None
    assert mixed_runs([_case(model="qwen"), _case(model="qwen", format="question")]) is not None


def test_answer_counts_as_judged_when_deduping():
    # 에러 줄 뒤에 다시 잰 객관식 결과가 붙으면 다시 잰 결과를 쓴다
    out = dedupe([_case(score=None, format="question"), _case(score=None, format="question", answer=_answer())])
    assert [c.answer is not None for c in out] == [True]


SPLIT = pathlib.Path("eval_split.json")


@pytest.mark.parametrize(
    ("split", "split_file", "open_final", "needs_split", "blocked"),
    [
        ("dev", SPLIT, False, True, False),
        ("final", SPLIT, False, True, True),  # final은 확인 옵션 없이 못 연다
        ("final", SPLIT, True, True, False),
        (None, None, False, True, True),  # 나누기 없는 전체 집계도 final이 섞여 막는다
        (None, None, True, True, False),
        (None, None, False, False, False),  # 평가셋 A만 볼 때는 나누기가 없다
        ("dev", None, False, True, True),  # 둘 중 하나만 주면 막는다
        (None, SPLIT, False, True, True),
    ],
)
def test_final_is_locked_unless_opened_on_purpose(split, split_file, open_final, needs_split, blocked):
    assert (split_error(split, split_file, open_final, needs_split) is not None) == blocked
