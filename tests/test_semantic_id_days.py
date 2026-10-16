import json

import pytest

from scripts.measure_travel_sequences import to_day_sequences
from scripts.semantic_id import build_days, loader
from scripts.semantic_id.preference import (
    has_preference,
    mission_codes,
    preference_vector,
    unknown_codes,
)
from scripts.semantic_id.sequences import (
    day_region,
    group_days,
    is_satisfied,
    visit_record,
)


def _visit(travel="t1", date="2022-10-01", order=1, type_cd="1", scores=("5", "5", "5"), addr="서울 종로구 사직로 161", area_id=None):
    return {
        "VISIT_AREA_ID": area_id or f"{travel}-{date}-{order}",
        "TRAVEL_ID": travel,
        "VISIT_ORDER": str(order),
        "VISIT_AREA_NM": " 경복궁 ",
        "VISIT_START_YMD": date,
        "ROAD_NM_ADDR": addr,
        "LOTNO_ADDR": "",
        "X_COORD": "126.97",
        "Y_COORD": "37.57",
        "POI_ID": "",
        "VISIT_AREA_TYPE_CD": type_cd,
        "DGSTFN": scores[0],
        "REVISIT_INTENTION": scores[1],
        "RCMDTN_INTENTION": scores[2],
    }


def _meta(travel="t1", mission="2;22;6;"):
    return {"TRAVEL_ID": travel, "TRAVELER_ID": f"u-{travel}", "TRAVEL_MISSION": mission}


# 정답 신호
@pytest.mark.parametrize("scores, expected", [
    (("4", "4", "4"), True),
    (("5", "4", "5"), True),
    (("3", "5", "5"), False),
    (("5", "", "5"), False),
    (("5", "5", "x"), False),
])
def test_is_satisfied_needs_all_three_scores_at_least_four(scores, expected):
    assert is_satisfied(_visit(scores=scores)) is expected


# 하루 묶기
def test_group_days_drops_home_lodging_and_unsatisfied_and_sorts_by_order():
    visits = [
        _visit(order=3),
        _visit(order=1, type_cd="21"),  # 집
        _visit(order=2, type_cd="24"),  # 숙소
        _visit(order=1),
        _visit(order=4, scores=("3", "5", "5")),
        _visit(date="2022-10-02", order=1),
    ]
    days = group_days(visits)
    assert sorted(days) == [("t1", "2022-10-01"), ("t1", "2022-10-02")]
    assert [r["VISIT_ORDER"] for r in days[("t1", "2022-10-01")]] == ["1", "3"]


def test_group_days_matches_measurement_script_on_the_same_data():
    # 측정 스크립트(PR #71)와 같은 하루 구성이어야 두 곳의 숫자가 어긋나지 않는다
    visits = [_visit(travel=f"t{i % 3}", date=f"2022-10-0{1 + i % 2}", order=i, type_cd=str(1 + i % 5),
                     scores=("5", "4" if i % 4 else "3", "5")) for i in range(40)]
    satisfied = [v for v in visits if is_satisfied(v)]
    ours = {k: [r["VISIT_AREA_ID"] for r in rows] for k, rows in group_days(visits).items()}
    theirs = {k: [r["VISIT_AREA_ID"] for r in rows] for k, rows in to_day_sequences(satisfied).items()
              if rows and all(r["VISIT_AREA_TYPE_CD"] not in {"21", "22", "23", "24"} for r in rows)}
    assert ours == theirs


def test_day_region_uses_first_service_region_address():
    rows = [_visit(addr="경기 용인시 처인구"), _visit(addr="부산 해운대구 우동", order=2)]
    assert day_region(rows) == "부산"
    assert day_region([_visit(addr="경기 용인시")]) == "기타"


def test_visit_record_keeps_selected_fields_and_handles_bad_coordinates():
    v = _visit()
    v["X_COORD"] = "4d129.06"
    rec = visit_record(v)
    assert rec["name"] == "경복궁"
    assert rec["x"] is None and rec["y"] == 37.57
    assert rec["dgstfn"] == 5.0 and rec["order"] == 1


# 취향 벡터
def test_preference_vector_maps_missions_to_four_categories():
    assert preference_vector("3;8;2;1;") == {
        "HISTORY_CULTURE": 1, "NATURE_HEALING": 1, "ACTIVITY": 1, "CONVENIENCE_SHOPPING": 1,
    }
    assert preference_vector("4;22;24;25;26;") == {
        "HISTORY_CULTURE": 0, "NATURE_HEALING": 0, "ACTIVITY": 0, "CONVENIENCE_SHOPPING": 0,
    }


def test_city_tour_is_not_shopping_but_hotel_vacation_is():
    assert preference_vector("4;")["CONVENIENCE_SHOPPING"] == 0
    assert preference_vector("23;")["CONVENIENCE_SHOPPING"] == 1


def test_has_preference_and_unknown_codes():
    assert not has_preference(preference_vector("22;24;"))
    assert has_preference(preference_vector("22;3;"))
    assert unknown_codes("22;3;99;") == {"99"}
    assert mission_codes("2;;22; 6") == {"2", "22", "6"}


# 깔때기
def _fixture():
    visits = []
    meta = []
    # 서울, 메타 있음, 취향 있음 → 남음
    visits += [_visit("a", order=1), _visit("a", order=2)]
    meta.append(_meta("a", "3;"))
    # 메타 없음 → 빠짐
    visits += [_visit("b", order=1)]
    # 메타는 있지만 미매핑 미션뿐 → 빠짐
    visits += [_visit("c", order=1)]
    meta.append(_meta("c", "22;24;"))
    # 서비스 지역 밖(경기) → 빠짐
    visits += [_visit("d", order=1, addr="경기 용인시")]
    meta.append(_meta("d", "3;"))
    # 만족하지 않은 방문만 있음 → 하루 자체가 없음
    visits += [_visit("e", order=1, scores=("2", "5", "5"))]
    meta.append(_meta("e", "3;"))
    return visits, meta


def test_funnel_counts_each_filter():
    examples, funnel = build_days.build_days(*_fixture())
    assert funnel["days_satisfied"] == 4  # a, b, c, d
    assert funnel["days_with_meta"] == 3  # a, c, d
    assert funnel["days_with_preference"] == 2  # a, d
    assert funnel["days_service_regions"] == 1  # a
    assert funnel["visits"] == 2 and funnel["trips"] == 1
    assert funnel["days_by_region"] == {"서울": 1}
    assert [e["travel_id"] for e in examples] == ["a"]
    assert examples[0]["preference"]["HISTORY_CULTURE"] == 1
    assert [v["order"] for v in examples[0]["visits"]] == [1, 2]


def test_unknown_mission_codes_are_reported():
    visits, meta = _fixture()
    meta.append(_meta("a2", "99;3;"))
    visits.append(_visit("a2", order=1))
    _, funnel = build_days.build_days(visits, meta)
    assert funnel["unknown_mission_codes"] == {"99": 1}


# 출력
def test_write_outputs_refuses_to_overwrite_without_flag(tmp_path):
    examples, funnel = build_days.build_days(*_fixture())
    build_days.write_outputs(tmp_path, examples, funnel, overwrite=False)
    lines = (tmp_path / "days.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["travel_id"] == "a"
    with pytest.raises(loader.DataRootError, match="덮어쓰지 않음"):
        build_days.write_outputs(tmp_path, examples, funnel, overwrite=False)
    build_days.write_outputs(tmp_path, examples, funnel, overwrite=True)  # 플래그가 있으면 다시 만든다
