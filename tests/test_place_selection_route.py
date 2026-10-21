import copy
import json
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.time import get_service_today
from app.main import app
from app.trips.schemas.schemas import E_Google_Place_Type
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


class NightTypedFakeDb(FakeDb):
    """저녁 이후 풀의 장소에 야간 type을 붙인다. 운영 DB는 저녁 이후 후보에 야간 type을 요구하므로 같은 모양이다."""

    def __call__(self, category, limit, region, excluded_types=None, required_types=None):
        places = super().__call__(category, limit, region, excluded_types, required_types)
        if required_types:
            night = E_Google_Place_Type(min(required_types))
            for place in places:
                place.types = [night]
        return places


@pytest.fixture
def client(monkeypatch):
    """모델 경로를 쓸 수 없는 서버(conftest가 get_router를 막는다). 2·3안은 규칙 변형으로 채워진다."""
    monkeypatch.setattr(settings, "internal_service_token", "t")
    monkeypatch.setattr(trip_rule, "get_DB_places_by_category", NightTypedFakeDb())
    return TestClient(app)


def _post(client, body):
    return client.post("/trips/place-selection", json=body, headers=HEADERS)


def _plans(payload):
    """응답 JSON에서 [1안, 2안, 3안]을 각각 [일자][장소]로 꺼낸다."""
    data = payload["data"]
    return [data["places"], *(route["days"] for route in data["routes"])]


def test_response_has_plan_one_in_places_and_two_routes_with_taste_counts(client):
    data = _post(client, _body(days=2)).json()["data"]
    assert len(data["places"]) == 3 and all(isinstance(day, list) for day in data["places"])  # 1안은 [일자][장소]
    assert [route["strategy"] for route in data["routes"]] == ["PREFERENCE", "DIVERSITY"]
    for route in data["routes"]:
        assert set(route) == {"strategy", "preference_count", "discovery_count", "days"}
        assert len(route["days"]) == 3
        assert route["preference_count"] >= 0 and route["discovery_count"] >= 0


def test_taste_counts_cover_only_daytime_activity_places_and_the_diversity_route_discovers_more(client):
    routes = _post(client, _body(days=2)).json()["data"]["routes"]
    for route in routes:
        counted = [
            place for day in route["days"] for place in day
            if "FOOD" not in place["matched_preferences"] and not place["id"].startswith("night")  # 식당과 저녁 이후는 세지 않는다
        ]
        assert route["preference_count"] + route["discovery_count"] == len(counted)
    preference, diversity = routes
    assert preference["discovery_count"] == 0  # 후보가 충분하면 취향 상위 2개 안에서만 고른다
    assert diversity["discovery_count"] >= 1  # 3안은 오후2 슬롯을 3·4위 카테고리로 바꾼다
    assert diversity["preference_count"] < preference["preference_count"]


def test_response_is_three_plans_by_day_even_without_a_model_route(client):
    response = _post(client, _body(days=2))
    assert response.status_code == 200
    plans = _plans(response.json())
    assert len(plans) == 3
    assert all(len(plan) == 3 for plan in plans)  # 3일
    assert all(len(plan[2]) == 2 for plan in plans)  # 마지막 날은 점심까지
    assert all(len({p["id"] for day in plan for p in day}) == sum(len(day) for day in plan) for plan in plans)  # 한 안 안에 중복 없음
    assert plans[1] != plans[0] and plans[2] != plans[0]  # 세 안은 서로 다른 시드로 고른 조합이다


def test_every_rule_plan_picks_with_its_own_seed_including_plan_one(client, monkeypatch):
    seeds = []
    real = trip_orchestrator.select_trip_by_slot_plan
    monkeypatch.setattr(
        trip_orchestrator, "select_trip_by_slot_plan",
        lambda *args, **kwargs: seeds.append(args[5]) or real(*args, **kwargs),
    )
    assert _post(client, _body(days=1)).status_code == 200
    assert len(seeds) == 3 and None not in seeds  # [0]도 평점 상위 K곳 중 시드로 고른다
    assert len(set(seeds)) == 3  # 안마다 다른 시드


def test_model_fallback_is_logged_as_a_warning_not_an_error(client, caplog):
    # ERROR 로그는 Sentry 이벤트가 되므로, 모델 서버를 못 쓰는 동안 요청마다 이벤트가 쌓이지 않아야 한다
    with caplog.at_level("WARNING", logger="app.trips.services.trip_orchestrator"):
        assert _post(client, _body()).status_code == 200
    fallbacks = [record for record in caplog.records if "모델 경로 실패, 규칙 변형으로 채움" in record.getMessage()]
    assert len(fallbacks) == 2 and all(record.levelname == "WARNING" for record in fallbacks)  # [1], [2]안 각각


def test_single_day_trip_keeps_the_full_day_template(client):
    plans = _plans(_post(client, _body()).json())
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
    plans = _plans(_post(client, _body(days=1)).json())
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
    plans = _plans(response.json())
    assert [len(plan) for plan in plans] == [3, 3, 3]
    assert all(len(plan[0]) >= 5 for plan in plans)  # 두 안 모두 규칙 변형으로 채워졌다
