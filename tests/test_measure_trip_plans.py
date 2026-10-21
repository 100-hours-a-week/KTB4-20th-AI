import json

import pytest

from app.trips.schemas.schemas import E_Google_Place_Type, E_Preference, E_Region
from app.trips.services import trip_orchestrator, trip_rule
from app.trips.services.model import runtime, trip_selection
from scripts import measure_trip_plans as measure
from tests.test_trip_selection import FakeDb, FakeTripRouter, make_place

SURVEYS = json.loads(measure.SURVEYS_PATH.read_text(encoding="utf-8"))


def _place(place_id, category=None, night=False):
    place = make_place(place_id)
    place.types = [E_Google_Place_Type.NIGHT_CLUB] if night else []
    place.matched_preferences = [category] if category else []
    return place


def test_difference_ignores_evening_places_and_compares_the_same_day():
    day = [_place("a"), _place("b"), _place("c"), _place("night", night=True)]
    other = [_place("a"), _place("x"), _place("night", night=True)]
    assert measure.is_evening_place(day[3]) and not measure.is_evening_place(day[0])
    assert measure.day_difference(day, other) == (2, 3)  # b, c가 다르다. 저녁 이후는 세지 않는다
    assert measure.day_difference(day, day) == (0, 3)


def test_meets_difference_needs_two_or_all_when_the_day_has_fewer_slots():
    assert measure.meets_difference(2, 5) and not measure.meets_difference(1, 5)
    assert measure.meets_difference(2, 2) and not measure.meets_difference(1, 2)  # 마지막 날(2슬롯)은 둘 다 달라야 한다
    assert measure.meets_difference(1, 1)
    assert measure.meets_difference(0, 0)


def test_extra_category_share_counts_only_non_food_daytime_places_of_rank_three_and_four():
    ranking = (E_Preference.HISTORY_CULTURE, E_Preference.NATURE_HEALING, E_Preference.ACTIVITY, E_Preference.CONVENIENCE_SHOPPING)
    trip = [[
        _place("a", E_Preference.HISTORY_CULTURE), _place("b", E_Preference.FOOD), _place("c", E_Preference.ACTIVITY),
        _place("d", E_Preference.CONVENIENCE_SHOPPING), _place("e", E_Preference.ACTIVITY, night=True),
    ]]
    assert measure.extra_category_share(trip, ranking) == (2, 3)  # FOOD와 저녁 이후는 분모에서 뺀다


def test_percentile_and_summary():
    values = [float(v) for v in range(1, 21)]
    assert measure.percentile(values, 0.95) == 20.0 and measure.percentile(values, 0.5) == 11.0
    assert measure.summarize([]) == {"n": 0}
    assert measure.summarize([1.0, 3.0])["mean"] == 2.0


def test_external_factors_for_builds_precheck_shape_and_skips_days_beyond_the_trip():
    from datetime import date

    factors = measure.external_factors_for(E_Region.GYEONGJU, date(2026, 11, 3), [1, 5], 3)
    assert [f["date"] for f in factors["forecasts"]] == ["2026-11-04"]
    assert factors["forecasts"][0]["regions"] == [
        {"region": "REGION-GYEONGJU", "has_issue": True, "reasons": [{"category": "weather_risk", "detail": "heavy_rain"}]}
    ]
    assert measure.external_factors_for(E_Region.SEOUL, date(2026, 11, 3), [5], 3) is None


def test_build_cases_is_deterministic_and_respects_the_limit():
    first = measure.build_cases(["서울", "경주"], 2, [1, 5], [2], None)
    assert len(first) == 8 and first == measure.build_cases(["서울", "경주"], 2, [1, 5], [2], None)
    assert len(measure.build_cases(["서울", "경주"], 2, [1, 5], [2], 3)) == 3
    shifted = measure.build_cases(["경주"], 2, [5], [], None, survey_start=3)
    assert sorted(case["survey"] for case in shifted) == [3, 4]  # 앞의 설문을 건너뛴다


@pytest.fixture
def fake_environment(monkeypatch):
    monkeypatch.setattr(trip_rule, "get_DB_places_by_category", FakeDb())
    monkeypatch.setattr(trip_selection, "get_DB_places_by_category", FakeDb())
    return monkeypatch


def test_run_case_records_model_calls_and_metrics_for_the_real_flow(fake_environment):
    router = FakeTripRouter()
    fake_environment.setattr(runtime, "get_router", lambda: router)
    result = measure.run_case({"region": "서울", "survey": 0, "days": 3, "signal_days": [1]}, SURVEYS)
    assert result["model_calls"]["plan1"]["used_model"] and result["model_calls"]["plan2"]["used_model"]
    assert result["model_calls"]["plan1"]["route"] == "gpu" and result["model_calls"]["plan1"]["sec"] >= 0
    assert len(result["plan_sizes"]) == 3 and all(len(sizes) == 3 for sizes in result["plan_sizes"])
    assert set(result["overlap"]) == set(measure.PAIRS) and len(result["overlap"]["plan1_vs_plan0"]["different"]) == 3
    assert set(result["extra_category_share"]) == {"plan0", "plan1", "plan2"}
    assert trip_orchestrator.select_trip_by_model is not None and trip_orchestrator.select_trip_by_model.__name__ == "select_trip_by_model"


def test_run_case_marks_a_failed_model_plan_as_not_used(fake_environment):
    class BrokenRouter:
        def generate_trip(self, *args, **kwargs):
            raise RuntimeError("GPU 실패")

    fake_environment.setattr(runtime, "get_router", lambda: BrokenRouter())
    result = measure.run_case({"region": "서울", "survey": 0, "days": 2, "signal_days": []}, SURVEYS)
    for label in ("plan1", "plan2"):
        assert not result["model_calls"][label]["used_model"] and result["model_calls"][label]["error"] == "RuntimeError"
    assert len(result["plan_sizes"]) == 3  # 규칙 변형으로 채워져 3개 안은 그대로 나온다


def _row(days, total, call1, call2, used=True, meets=True, extra=(1, 4), outdoor=0):
    return {
        "days": days, "total_sec": total,
        "model_calls": {"plan1": {"sec": call1, "used_model": used}, "plan2": {"sec": call2, "used_model": used}},
        "overlap": {name: {"different": [3] * days, "slots": [5] * days, "meets": [meets] * days} for name in measure.PAIRS},
        "extra_category_share": {"plan0": (0, 4), "plan1": (0, 4), "plan2": extra},
        "outdoor_on_signal_days": {"plan0": 0, "plan1": 0, "plan2": outdoor},
    }


def test_aggregate_applies_the_adr_criteria_and_skips_the_first_requests_for_latency():
    results = [_row(9, 40.0, 30.0, 30.0)] + [_row(9, 12.0, 8.0, 9.0), _row(9, 14.0, 9.0, 10.0)]
    summary = measure.aggregate(results, skip_first=1)
    assert summary["latency_requests"] == 2 and summary["total_max_sec"] == 14.0  # 첫 요청(콜드 스타트 가능)은 지연에서 뺀다
    assert summary["model_call_p95_sec"] == 10.0
    assert summary["criteria"] == {
        "model_call_p95_within_15s": True, "total_within_30s": True, "no_outdoor_on_signal_days": True, "plan1_vs_plan0_meets_every_day": True,
    }
    assert summary["model_used"] == {"plan1": 3, "plan2": 3}
    assert summary["extra_category_share"]["plan2"] == {"extra": 3, "total": 12, "rate": 0.25}
    slow = measure.aggregate([_row(9, 31.0, 16.0, 12.0, used=False, meets=False)])
    assert slow["criteria"] == {
        "model_call_p95_within_15s": False, "total_within_30s": False, "no_outdoor_on_signal_days": True, "plan1_vs_plan0_meets_every_day": False,
    }
    leaked = measure.aggregate([_row(9, 1.0, 1.0, 1.0, outdoor=2)])
    assert leaked["outdoor_on_signal_days"] == {"plan0": 0, "plan1": 0, "plan2": 2} and not leaked["criteria"]["no_outdoor_on_signal_days"]
    assert slow["model_used"] == {"plan1": 0, "plan2": 0} and slow["overlap"]["plan1_vs_plan0"]["meets_rate"] == 0.0


def test_should_stop_only_after_consecutive_requests_without_any_model_result():
    ok, bad = _row(5, 1.0, 1.0, 1.0), _row(5, 1.0, 0.1, 0.1, used=False)
    assert not measure.should_stop([bad, bad], 3)  # 아직 3건이 안 쌓였다
    assert measure.should_stop([ok, bad, bad, bad], 3)
    assert not measure.should_stop([bad, bad, ok], 3)  # 마지막에 하나라도 성공하면 계속한다
    assert not measure.should_stop([bad, bad, bad], 0)  # 0이면 끈다


def test_outdoor_places_on_days_counts_only_signal_days():
    outdoor, indoor = _place("o"), _place("i")
    outdoor.types = [E_Google_Place_Type.BEACH]
    trip = [[indoor, outdoor], [outdoor], [indoor]]
    assert measure.outdoor_places_on_days(trip, [0, 2]) == 1
    assert measure.outdoor_places_on_days(trip, [1, 9]) == 1  # 여행보다 긴 번호는 무시
    assert measure.outdoor_places_on_days(trip, []) == 0


class _Response:
    def __init__(self, status_code, body):
        self.status_code, self._body = status_code, body

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body


def test_preflight_reports_why_the_endpoint_does_not_accept_work():
    paused = _Response(409, {"status": 409, "title": "Conflict", "detail": "Endpoint is paused (max_workers=0).", "code": "ENDPOINT_PAUSED"})
    ok, message = measure.preflight_runpod("http://x", "k", post=lambda *a, **k: paused)
    assert not ok and "409" in message and "ENDPOINT_PAUSED" in message and "max_workers=0" in message
    assert measure.preflight_runpod("http://x", "k", post=lambda *a, **k: _Response(200, {"status": "COMPLETED"})) == (True, "접수·완료")
    assert measure.preflight_runpod("http://x", "k", post=lambda *a, **k: _Response(200, {"status": "FAILED"}))[0] is False
    assert measure.preflight_runpod("http://x", "k", post=lambda *a, **k: _Response(502, None)) == (False, "HTTP 502")

    def boom(*args, **kwargs):
        raise TimeoutError

    assert measure.preflight_runpod("http://x", "k", post=boom) == (False, "TimeoutError")
