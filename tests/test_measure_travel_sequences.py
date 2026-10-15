from scripts.measure_travel_sequences import (
    sequence_region,
    summarize,
    to_day_sequences,
)


def _visit(travel, ymd, order, type_cd="1", dgstfn="5", addr="서울 종로구 사직로 161"):
    return {
        "TRAVEL_ID": travel,
        "VISIT_START_YMD": ymd,
        "VISIT_ORDER": str(order),
        "VISIT_AREA_TYPE_CD": type_cd,
        "DGSTFN": dgstfn,
        "ROAD_NM_ADDR": addr,
        "LOTNO_ADDR": "",
    }


def test_same_travel_on_different_days_are_separate_sequences():
    visits = [_visit("t1", "2022-10-01", 1), _visit("t1", "2022-10-02", 2)]
    assert len(to_day_sequences(visits)) == 2


def test_home_and_lodging_are_excluded_and_order_is_kept():
    visits = [
        _visit("t1", "2022-10-01", 3),
        _visit("t1", "2022-10-01", 1, type_cd="21"),
        _visit("t1", "2022-10-01", 2, type_cd="24"),
        _visit("t1", "2022-10-01", 1),
    ]
    rows = next(iter(to_day_sequences(visits).values()))
    assert [r["VISIT_ORDER"] for r in rows] == ["1", "3"]


def test_unrated_visit_is_dropped_before_counting():
    visits = [_visit("t1", "2022-10-01", 1), _visit("t1", "2022-10-01", 2, dgstfn="")]
    assert summarize(visits, None)["전체"]["visits"] == 1


def test_satisfaction_threshold_filters_visits():
    visits = [_visit("t1", "2022-10-01", 1, dgstfn="4"), _visit("t1", "2022-10-01", 2, dgstfn="3")]
    assert summarize(visits, 4)["전체"]["visits"] == 1


def test_length_n_gives_n_minus_one_examples_and_len3_count():
    visits = [_visit("t1", "2022-10-01", i) for i in range(1, 4)] + [_visit("t2", "2022-10-01", 1)]
    s = summarize(visits, None)["전체"]
    assert s["days"] == 2
    assert s["examples"] == 2  # 길이 3은 2개, 길이 1은 0개
    assert s["len3_plus"] == 1


def test_region_comes_from_first_service_region_address():
    rows = [_visit("t1", "d", 1, addr="경기 용인시"), _visit("t1", "d", 2, addr="부산 해운대구")]
    assert sequence_region(rows) == "부산"
    assert sequence_region([_visit("t1", "d", 1, addr="경기 용인시")]) == "기타"


def test_all_three_scores_must_reach_threshold():
    def v(order, d, r, c):
        row = _visit("t1", "2022-10-01", order, dgstfn=d)
        row["REVISIT_INTENTION"], row["RCMDTN_INTENTION"] = r, c
        return row

    visits = [v(1, "5", "4", "4"), v(2, "5", "3", "5"), v(3, "5", "", "5")]
    fields = ("DGSTFN", "REVISIT_INTENTION", "RCMDTN_INTENTION")
    assert summarize(visits, 4, fields)["전체"]["visits"] == 1
    assert summarize(visits, 4)["전체"]["visits"] == 3  # 만족도만 보면 모두 통과
