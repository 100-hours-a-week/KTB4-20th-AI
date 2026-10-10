import pytest

from scripts.eval_cases import EvalCase, classify, dedupe, from_a, metrics


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
