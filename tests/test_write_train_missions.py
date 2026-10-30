from scripts.write_train_missions import (
    PW_COND,
    PW_PLAIN,
    conflicts,
    pw_mode,
    swap_targets,
)


def _rec(photo, usable=True, with_condition=False, source="photo_written", category="FOOD"):
    return {"mission_id": f"{photo}#pw", "photo": photo, "source": source, "category": category,
            "usable": usable, "with_condition": with_condition}


def test_pw_mode_fills_condition_missions_first_then_plain():
    assert pw_mode([]) == "with_condition"
    two_cond = [_rec("a", with_condition=True), _rec("b", with_condition=True)]
    assert pw_mode(two_cond) == "plain"
    assert pw_mode([*two_cond, _rec("c"), _rec("d")]) is None  # 4장 다 채움


def test_pw_mode_ignores_unusable_photos():
    # 맞는 미션을 못 쓴 사진은 세지 않고 예비 사진으로 넘어간다
    assert pw_mode([_rec("a", usable=False, with_condition=True)]) == "with_condition"
    assert PW_COND + PW_PLAIN == 4


def test_pw_mode_can_fill_more_per_category():
    # 550개용은 카테고리마다 24개(조건 12 + 그냥 12)를 채운다. 기본값을 쓰면 파일럿처럼 4개에서 멈춘다
    twelve_cond = [_rec(f"c{i}", with_condition=True) for i in range(12)]
    assert pw_mode(twelve_cond[:4], 12, 12) == "with_condition"
    assert pw_mode(twelve_cond, 12, 12) == "plain"
    assert pw_mode([*twelve_cond, *(_rec(f"p{i}") for i in range(12))], 12, 12) is None


def test_swap_targets_are_usable_condition_missions_not_yet_swapped():
    records = [
        _rec("a", with_condition=True),
        _rec("b", with_condition=True),
        _rec("c"),  # 조건 없이 쓴 미션은 바꿀 원본으로 쓰지 않는다
        _rec("d", usable=False, with_condition=True),
        {"mission_id": "a.jpg#swap", "photo": "a", "source": "condition_swap", "parent_mission_id": "a#pw",
         "category": "FOOD", "usable": True},
    ]
    assert [r["photo"] for r in swap_targets(records)] == ["b"]


def test_swap_targets_with_labels_use_only_human_success_parents():
    # 2차: 원래 미션이 맞는 사진(사람이 성공으로 표시)에서만 바꾼다. 조건 없이 쓴 미션도 원본이 될 수 있다
    records = [_rec("a"), _rec("b"), _rec("c", with_condition=True)]
    assert [r["photo"] for r in swap_targets(records, {"a#pw", "c#pw"})] == ["a", "c"]


def test_conflicts_rejects_together_with_close_up():
    assert conflicts("다 같이 석갈비 가까이 찍기")
    assert not conflicts("다 같이 석갈비 앞에서 찍기")
