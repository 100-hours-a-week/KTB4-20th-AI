import copy
import json
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.time import get_service_today
from app.main import app
from app.trips.services import trip_orchestrator, trip_rule
from app.trips.services.model import runtime
from tests.test_trip_selection import FakeDb, make_place

SURVEYS = Path(__file__).resolve().parent / "trips" / "data" / "representative_surveys.json"
HEADERS = {"x-internal-token": "t"}


def _body(days=0, external_factors=None, region="서울"):
    members = copy.deepcopy(json.loads(SURVEYS.read_text(encoding="utf-8"))[0]["members"])
    today = get_service_today()
    body = {"region": region, "start_date": today.isoformat(), "end_date": (today + timedelta(days=days)).isoformat(), "members": members}
    if external_factors is not None:
        body["external_factors"] = external_factors
    return body


def _factors(*entries, generated_at="2026-10-10T10:00:03Z"):
    """entries: (여행 시작일로부터 며칠 뒤, precheck 지역 표기, has_issue)"""
    today = get_service_today()
    days = {}
    for offset, region, has_issue in entries:
        day = (today + timedelta(days=offset)).isoformat()
        days.setdefault(day, []).append({"region": region, "has_issue": has_issue, "reasons": []})
    return {"generated_at": generated_at, "forecasts": [{"date": day, "regions": regions} for day, regions in days.items()]}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "internal_service_token", "t")
    monkeypatch.setattr(settings, "place_selection_mode", "rule")
    monkeypatch.setattr(trip_rule, "get_DB_places_by_category", FakeDb())
    return TestClient(app)


def _post(client, body):
    return client.post("/trips/place-selection", json=body, headers=HEADERS)


def test_response_is_three_plans_by_day_even_in_rule_mode(client):
    response = _post(client, _body(days=2))
    assert response.status_code == 200
    plans = response.json()["data"]["places"]
    assert len(plans) == 3
    assert all(len(plan) == 3 for plan in plans)  # 3일
    assert all(len(plan[2]) == 2 for plan in plans)  # 마지막 날은 점심까지
    assert all(len({p["id"] for day in plan for p in day}) == sum(len(day) for day in plan) for plan in plans)  # 한 안 안에 중복 없음
    assert plans[1] != plans[0] and plans[2] != plans[0]  # 규칙 변형은 평점순과 다른 조합이다


def test_single_day_trip_keeps_the_full_day_template(client):
    plans = _post(client, _body()).json()["data"]["places"]
    assert [len(plan) for plan in plans] == [1, 1, 1] and all(len(plan[0]) in (5, 6) for plan in plans)


def test_same_request_gives_the_same_response(client):
    first = _post(client, _body(days=3)).json()
    assert first == _post(client, _body(days=3)).json()


def test_trip_longer_than_nine_days_is_rejected_and_nine_days_is_allowed(client):
    assert _post(client, _body(days=8)).status_code == 200
    response = _post(client, _body(days=9))
    assert response.status_code == 422
    assert "end_date" in [error["loc"][-1] for error in response.json()["detail"]]


def test_signal_days_come_from_the_requests_region_and_trip_dates_only(client, monkeypatch):
    seen = []
    real = trip_orchestrator.select_trip_by_slot_plan
    monkeypatch.setattr(
        trip_orchestrator, "select_trip_by_slot_plan",
        lambda *args, **kwargs: seen.append(args[4]) or real(*args, **kwargs),
    )
    factors = _factors(
        (1, "REGION-SEOUL", True),  # 이 여행의 둘째 날, 요청 지역
        (0, "REGION-BUSAN", True),  # 다른 지역
        (2, "REGION-SEOUL", False),  # 신호 없음
        (5, "REGION-SEOUL", True),  # 여행 날짜 밖
        (0, "REGION-UNKNOWN", True),  # 모르는 표기
    )
    assert _post(client, _body(days=2, external_factors=factors)).status_code == 200
    assert seen and all(days == frozenset({1}) for days in seen)


def test_external_factors_accept_unknown_fields_and_empty_values(client):
    factors = _factors((0, "REGION-SEOUL", False))
    factors["forecasts"][0]["regions"][0]["reasons"] = [{"category": "weather_risk", "detail": "heavy_rain", "extra": 1}]
    assert _post(client, _body(external_factors=factors)).status_code == 200
    assert _post(client, _body(external_factors={"generated_at": None, "forecasts": []})).status_code == 200
    bad = _factors((0, "REGION-SEOUL", False))
    bad["forecasts"][0]["regions"][0]["has_issue"] = "maybe"
    assert _post(client, _body(external_factors=bad)).status_code == 422


def test_unrelated_external_changes_do_not_change_the_response(client):
    base = _post(client, _body(days=2, external_factors=_factors((0, "REGION-SEOUL", False), generated_at="2026-10-10T10:00:03Z"))).json()
    other_time_and_region = _factors((0, "REGION-SEOUL", False), (1, "REGION-BUSAN", True), generated_at="2026-10-11T09:00:00Z")
    assert base == _post(client, _body(days=2, external_factors=other_time_and_region)).json()
    rainy = _post(client, _body(days=2, external_factors=_factors((1, "REGION-SEOUL", True)))).json()
    assert rainy != base  # 요청 지역·여행 날짜의 신호가 바뀌면 일정도 달라질 수 있다


def _model_mode(monkeypatch, fake_model):
    monkeypatch.setattr(settings, "place_selection_mode", "model")
    monkeypatch.setattr(settings, "model_first_place_top_k", 30)
    monkeypatch.setattr(runtime, "get_router", lambda: object())
    monkeypatch.setattr(trip_orchestrator, "select_trip_by_model", fake_model)


def _fake_model_trip(trip_plan, tag):
    return [[make_place(f"{tag}{day}_{slot}") for slot, item in enumerate(day_plan.items) if item.time_slot.value != "EVENING"]
            for day, day_plan in enumerate(trip_plan.days)]


def test_model_plans_use_derived_seeds_k_from_settings_and_diversity_for_the_third(client, monkeypatch):
    seeds = {False: [], True: []}  # 안([1]=False, [2]=True)마다 요청 순서대로. 두 안은 스레드에서 동시에 불리므로 안별로 나눠 기록한다
    first_ks = []

    def fake_model(trip_plan, deal_breakers, region, members, router, outdoor_excluded_days, diversity=False, **kwargs):
        seeds[diversity].append(kwargs["seed"])
        first_ks.append(kwargs["first_k"])
        return _fake_model_trip(trip_plan, "div" if diversity else "pref"), "gpu"

    _model_mode(monkeypatch, fake_model)
    plans = _post(client, _body(days=1)).json()["data"]["places"]
    assert plans[1][0][0]["id"].startswith("pref") and plans[2][0][0]["id"].startswith("div")
    assert not plans[0][0][0]["id"].startswith(("pref", "div"))  # [0]은 항상 규칙 경로
    assert first_ks == [30, 30]
    assert seeds[False][0] != seeds[True][0]  # 안마다 다른 시드

    _post(client, _body(days=1))
    _post(client, _body(days=2))
    for plan_seeds in seeds.values():
        assert plan_seeds[0] == plan_seeds[1]  # 같은 요청은 같은 시드
        assert plan_seeds[0] != plan_seeds[2]  # 요청이 다르면 시드도 다르다


def test_model_failure_or_invalid_result_fills_only_that_plan_with_a_rule_variant(client, monkeypatch):
    def fake_model(trip_plan, deal_breakers, region, members, router, outdoor_excluded_days, diversity=False, **kwargs):
        if not diversity:
            raise RuntimeError("모델 경로 실패")
        return [[] for _ in trip_plan.days], "gpu"  # 슬롯을 거의 못 채운 결과는 유효하지 않다

    _model_mode(monkeypatch, fake_model)
    response = _post(client, _body(days=2))
    assert response.status_code == 200
    plans = response.json()["data"]["places"]
    assert [len(plan) for plan in plans] == [3, 3, 3]
    assert all(len(plan[0]) >= 5 for plan in plans)  # 두 안 모두 규칙 변형으로 채워졌다
