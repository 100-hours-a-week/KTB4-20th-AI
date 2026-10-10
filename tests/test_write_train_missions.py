from scripts.write_train_missions import PW_COND, PW_PLAIN, pw_mode, swap_targets


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
