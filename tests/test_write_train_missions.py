from scripts.write_train_missions import (
    PW_COND,
    PW_PLAIN,
    conflicts,
    pw_mode,
    pw_next,
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


def test_pw_next_rescues_failed_condition_photo_with_plain_once():
    # 조건 강제가 못 씀이면 같은 사진으로 plain을 한 번 더 쓴다. 이미 실패해 기록된 사진도 다시 돌리면 구제한다
    failed = _rec("a", usable=False, with_condition=True)
    assert pw_next("a", [failed], tried={"a#pw"}) == "plain_rescue"
    rescued = {**_rec("a"), "mission_id": "a#pw_plain"}
    assert pw_next("a", [failed, rescued], tried={"a#pw", "a#pw_plain"}) is None  # 성공하면 끝
    gave_up = {**rescued, "usable": False}
    assert pw_next("a", [failed, gave_up], tried={"a#pw", "a#pw_plain"}) is None  # 둘 다 못 씀이어도 다시 부르지 않음


def test_pw_next_first_call_and_no_rescue_when_plain_is_full():
    assert pw_next("new", [], tried=set()) == "with_condition"
    two_plain = [_rec("p1"), _rec("p2")]
    failed = _rec("a", usable=False, with_condition=True)
    assert pw_next("a", [failed, *two_plain], tried={"a#pw"}) is None  # 비강제 칸이 찼으면 구제 안 함
    assert pw_next("b", [_rec("c1", with_condition=True), _rec("c2", with_condition=True)], tried=set()) == "plain"


def test_pw_next_does_not_repeat_errored_call_in_same_run():
    # 에러 난 줄은 records에 없지만 tried에 있으므로 같은 실행에서 되풀이하지 않는다
    assert pw_next("a", [], tried={"a#pw"}) is None


def test_pw_next_stops_when_category_is_full():
    full = [_rec("c1", with_condition=True), _rec("c2", with_condition=True), _rec("p1"), _rec("p2")]
    assert pw_next("new", full, tried=set()) is None


def test_swap_targets_accept_human_success_plain_parent():
    # 사람이 성공으로 표시했으면 plain(구제 포함) 미션도 조건 바꾸기 원본이 된다
    records = [{**_rec("a"), "mission_id": "a#pw_plain"}, _rec("b", usable=False, with_condition=True)]
    assert [r["mission_id"] for r in swap_targets(records, {"a#pw_plain"})] == ["a#pw_plain"]


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


def _run_main(tmp_path, monkeypatch, answers, max_calls):
    # 가짜 Gemini로 main의 사진 보고 쓰기 반복문을 돌린다. answers: 사진별로 (조건 강제 결과, 비강제 결과)
    import asyncio
    import json
    import sys

    import scripts.write_train_missions as w
    from scripts.write_photo_missions import WrittenMission

    calls = []

    async def fake_write(row, image, extra):
        mode = "cond" if extra else "plain"
        calls.append((row["photo"], mode))
        ok = answers[row["photo"]][0 if extra else 1]
        return WrittenMission(usable=ok, mission="탑 찍기" if ok else "")

    class FakeZip:
        def __init__(self, *a):
            pass

        def read(self, member):
            return b""

    monkeypatch.setattr(w, "write_one", fake_write)
    monkeypatch.setattr(w.zipfile, "ZipFile", FakeZip)
    monkeypatch.setattr(w.pipeline_generate, "start", lambda: None)
    rows = [{"photo": p, "travel_id": f"t{p}", "place_name": "탑", "region": "경주", "category": "HISTORY_CULTURE",
             "role": "photo_written", "zip": "z.zip", "member": p} for p in answers]
    (tmp_path / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["x", "--pilot-dir", str(tmp_path), "--aihub", str(tmp_path),
                                      "--skip-swap", "--max-calls", str(max_calls)])
    asyncio.run(w.main())
    lines = [json.loads(x) for x in (tmp_path / "missions.jsonl").read_text(encoding="utf-8").splitlines()]
    return calls, lines


def test_main_last_allowed_call_does_not_trigger_rescue(tmp_path, monkeypatch):
    calls, lines = _run_main(tmp_path, monkeypatch, {"a": (False, True)}, max_calls=1)
    assert calls == [("a", "cond")] and [x["mission_id"] for x in lines] == ["a#pw"]


def test_main_rerun_resumes_pending_rescue_then_makes_no_more_calls(tmp_path, monkeypatch):
    answers = {"a": (False, True), "b": (False, False)}
    _run_main(tmp_path, monkeypatch, answers, max_calls=1)  # a 조건 강제만 하고 멈춤
    calls, _ = _run_main(tmp_path, monkeypatch, answers, max_calls=10)
    assert calls[0] == ("a", "plain")  # 남은 구제부터
    assert calls[1:] == [("b", "cond"), ("b", "plain")]
    calls, lines = _run_main(tmp_path, monkeypatch, answers, max_calls=10)
    assert calls == []  # 구제가 성공(a)했든 실패(b)했든 다시 부르지 않음
    assert sorted(x["mission_id"] for x in lines) == ["a#pw", "a#pw_plain", "b#pw", "b#pw_plain"]
