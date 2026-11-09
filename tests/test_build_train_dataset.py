import json

import pytest

from scripts.build_train_dataset import (
    apply_overrides,
    load_missions,
    merge_labels,
    merge_missions,
    select,
    source_of,
)


def _label(mission_id, grade="success", q2="enough", q3="na", q4="yes"):
    return {"mission_id": mission_id, "photo": mission_id.split("#")[0], "final_grade": grade,
            "q1": "yes", "q2": q2, "q3": q3, "q4": q4}


def _mission(mission_id, **kw):
    return {"mission_id": mission_id, "photo": mission_id.split("#")[0], "place": "황리단길",
            "mission": "한옥 거리 찍기", "source": "photo_written", **kw}


def test_error_line_does_not_overwrite_earlier_mission():
    # 같은 미션을 다시 만들다 에러가 나도 앞의 정상 미션을 쓴다
    lines = [json.dumps(_mission("a.jpg#pw")), json.dumps({"mission_id": "a.jpg#pw", "error": "502"}), ""]
    assert load_missions(lines)["a.jpg#pw"]["mission"] == "한옥 거리 찍기"


def test_same_mission_labeled_twice_is_refused():
    # 두 번 들어가면 그 사진을 두 배로 배운다. 어느 표시가 맞는지는 사람이 정한다
    with pytest.raises(ValueError, match="두 번"):
        merge_labels([[_label("a.jpg#pw")], [_label("a.jpg#pw", "retry", q2="short")]])
    assert len(merge_labels([[_label("a.jpg#pw")], [_label("b.jpg#pw")]])) == 2


def test_select_skips_unlabeled_unclear_and_rule_mismatch():
    missions = {m: _mission(m) for m in ("a.jpg#pw", "b.jpg#pw", "c.jpg#pw", "d.jpg#pw")}
    labels = [
        _label("a.jpg#pw"),
        {"mission_id": "b.jpg#pw", "final_grade": None},
        _label("c.jpg#pw", "unclear"),
        _label("d.jpg#pw", "retry"),  # 답은 모두 성공 조건인데 사람이 재시도로 정함 → 객관식으로 재현 안 됨
    ]
    kept, skipped = select(labels, missions)
    assert [label["mission_id"] for label, _ in kept] == ["a.jpg#pw"]
    assert skipped == {"표시 안 함": 1, "미션 애매": 1, "규칙과 다른 등급": 1}


def test_label_without_mission_gives_clear_error():
    with pytest.raises(ValueError, match="미션 파일에 없는"):
        select([_label("x.jpg#pw")], {})


def test_select_skips_hold_and_near_dup():
    # 보류는 Q1~Q4가 비어 있어 normalize에서 멈추면 안 되고, 거의 같은 사진은 두 번 배우지 않게 뺀다
    missions = {m: _mission(m) for m in ("a.jpg#own", "b.jpg#own", "c.jpg#own")}
    labels = [
        {"mission_id": "a.jpg#own", "final_grade": "hold", "q1": None, "q2": None, "q3": None, "q4": None},
        {**_label("b.jpg#own"), "near_dup": True},
        _label("c.jpg#own"),
    ]
    kept, skipped = select(labels, missions)
    assert [label["mission_id"] for label, _ in kept] == ["c.jpg#own"]
    assert skipped == {"보류": 1, "거의 같은 사진": 1}


def test_labels_outside_this_set_are_counted_when_allowed():
    # 파일럿 표시 파일에는 재사용하지 않는 미션도 있다
    kept, skipped = select([_label("a.jpg#pw"), _label("x.jpg#pw")], {"a.jpg#pw": _mission("a.jpg#pw")}, allow_extra=True)
    assert len(kept) == 1 and skipped == {"이번 세트 아님": 1}


def test_same_mission_in_two_mission_files_is_refused():
    with pytest.raises(ValueError, match="같은 미션"):
        merge_missions([{"a.jpg#pw": _mission("a.jpg#pw")}, {"a.jpg#pw": _mission("a.jpg#pw")}])


def test_override_replaces_whole_label_and_refuses_unknown():
    labels = [_label("a.jpg#gen", "fail", q2="none", q3=None, q4="na"), _label("b.jpg#gen")]
    fixed = apply_overrides(labels, [_label("a.jpg#gen", "retry", q2="short", q4="unknown")])
    assert fixed[0]["final_grade"] == "retry" and fixed[0]["q2"] == "short" and fixed[1] == labels[1]
    with pytest.raises(ValueError, match="원래 표시에 없음"):
        apply_overrides(labels, [_label("z.jpg#gen")])


def test_lookalike_mission_source_comes_from_kind():
    assert source_of({"kind": "wrong"}) == "lookalike_wrong"
    assert source_of(_mission("a.jpg#pw")) == "photo_written"
