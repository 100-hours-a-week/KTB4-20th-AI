import json

import pytest

from scripts.build_train_dataset import load_missions, merge_labels, select


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
