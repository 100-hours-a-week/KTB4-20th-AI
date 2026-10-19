from datetime import date, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.time import SERVICE_TIMEZONE
from app.main import app
from app.trips import router as trips_router
from app.trips.constants import HEX_GRID_POINTS, REGION_TO_COLLECTION_AREAS
from app.trips.schemas.api import (
    E_Weather_Detail,
    External_Factors,
    Precheck_Response_Data,
)
from app.trips.schemas.schemas import E_Region
from app.trips.services import kma_client, precheck_service
from app.trips.services.external_factors import (
    PRECHECK_REGION_TO_REGION,
    outdoor_excluded_days,
)
from app.trips.services.kma_client import KmaError, MidDay, ShortDay
from app.trips.services.precheck import convert_to_grid
from app.trips.services.precheck_service import (
    REGIONS,
    ForecastCache,
    PrecheckUnavailable,
    build_precheck,
)
from app.trips.services.weather_risk import judge_mid_day, judge_short_day, parse_amount

NOW = datetime(2026, 10, 19, 6, 0, tzinfo=SERVICE_TIMEZONE)
HEAVY, SNOW = E_Weather_Detail.HEAVY_RAIN, E_Weather_Detail.HEAVY_SNOW
RAIN_LIKELY, SNOW_LIKELY = E_Weather_Detail.RAIN_LIKELY, E_Weather_Detail.SNOW_LIKELY


# ---- 판정 (기준값은 weather_risk의 상수) ----

def test_parse_amount_handles_every_observed_and_documented_format():
    cases = {"강수없음": 0.0, "적설없음": 0.0, "0": 0.0, "1mm 미만": 0.5, "1cm 미만": 0.5, "1.0mm": 1.0, "8.0mm": 8.0,
             "30.0~50.0mm": 30.0, "50.0mm 이상": 50.0, "5.0cm 이상": 5.0, " 2.5mm ": 2.5}
    for text, expected in cases.items():
        assert parse_amount(text) == expected, text
    assert parse_amount("") is None and parse_amount("알 수 없음 값") is None and parse_amount(None) is None


def test_short_term_thresholds_are_inclusive_and_independent_for_rain_and_snow():
    assert judge_short_day([10.0, 10.0, 10.0], []) == [HEAVY]  # 하루 합계 30mm
    assert judge_short_day([10.0, 10.0, 9.5], []) == []  # 합계 29.5mm
    assert judge_short_day([15.0], []) == [HEAVY]  # 어느 한 시간 15mm
    assert judge_short_day([14.9], []) == []
    assert judge_short_day([], [5.0]) == [SNOW] and judge_short_day([], [2.5, 2.4]) == []
    assert judge_short_day([30.0], [6.0]) == [HEAVY, SNOW]
    assert judge_short_day([], []) == []


def test_mid_term_needs_sixty_percent_and_a_matching_phrase():
    assert judge_mid_day([60, 40], ["흐리고 비", "흐림"]) == [RAIN_LIKELY]
    assert judge_mid_day([59], ["흐리고 비"]) == []  # 확률이 모자람
    assert judge_mid_day([80], ["맑음"]) == []  # 문구가 비·눈이 아님
    assert judge_mid_day([70], ["구름많고 소나기"]) == [RAIN_LIKELY]
    assert judge_mid_day([60], ["흐리고 눈"]) == [SNOW_LIKELY]
    assert judge_mid_day([90], ["흐리고 비/눈"]) == [RAIN_LIKELY, SNOW_LIKELY]
    assert judge_mid_day([], []) == []


# ---- 기상청 클라이언트 ----

def test_short_base_and_mid_issue_candidates_use_the_latest_published_forecasts():
    assert kma_client.short_base_candidates(NOW) == [("20261019", "0500"), ("20261019", "0200")]  # 06:00 → 05:45 기준
    early = datetime(2026, 10, 19, 2, 10, tzinfo=SERVICE_TIMEZONE)
    assert kma_client.short_base_candidates(early) == [("20261018", "2300"), ("20261018", "2000")]  # 02시 발표분은 02:15 뒤부터
    assert [c.strftime("%Y%m%d%H") for c in kma_client.mid_issue_candidates(NOW)] == ["2026101906", "2026101818"]
    before = datetime(2026, 10, 19, 5, 59, tzinfo=SERVICE_TIMEZONE)
    assert [c.strftime("%Y%m%d%H") for c in kma_client.mid_issue_candidates(before)] == ["2026101818", "2026101806"]


def _envelope(items, total=None, code="00"):
    body = {"items": {"item": items}, "totalCount": len(items) if total is None else total} if items else {}
    return {"header": {"resultCode": code, "resultMsg": "x"}, "body": body}


def _short_item(day, time, category, value):
    return {"fcstDate": day, "fcstTime": time, "category": category, "fcstValue": value}


def test_fetch_short_reads_every_page_and_groups_rain_and_snow_by_date(monkeypatch):
    pages = {
        1: _envelope([_short_item("20261020", "0000", "PCP", "1.0mm"), _short_item("20261020", "0000", "SNO", "적설없음")], total=4),
        2: _envelope([_short_item("20261020", "0100", "PCP", "강수없음"), _short_item("20261021", "0000", "SNO", "5.0cm 이상")], total=4),
    }
    calls = []

    def fake(url, params):
        calls.append(params["pageNo"])
        return pages[params["pageNo"]]

    monkeypatch.setattr(kma_client, "_response", fake)
    days = kma_client.fetch_short(60, 127, NOW)
    assert calls == [1, 2]
    assert days[date(2026, 10, 20)].rain_mm == [1.0, 0.0] and days[date(2026, 10, 20)].hours == {"0000", "0100"}
    assert days[date(2026, 10, 21)].snow_cm == [5.0]


def test_fetch_short_falls_back_to_the_previous_base_time_when_the_latest_has_no_data(monkeypatch):
    seen = []

    def fake(url, params):
        seen.append(params["base_time"])
        return _envelope([], code="03") if params["base_time"] == "0500" else _envelope([_short_item("20261020", "0000", "PCP", "1.0mm")])

    monkeypatch.setattr(kma_client, "_response", fake)
    assert kma_client.fetch_short(60, 127, NOW)[date(2026, 10, 20)].rain_mm == [1.0]
    assert seen == ["0500", "0200"]
    monkeypatch.setattr(kma_client, "_response", lambda url, params: _envelope([], code="03"))
    with pytest.raises(KmaError):
        kma_client.fetch_short(60, 127, NOW)


def test_fetch_mid_maps_fields_to_dates_from_the_issue_date_and_merges_am_pm(monkeypatch):
    item = {"regId": "11G00000", "rnSt5Am": 60, "rnSt5Pm": 40, "rnSt8": 10, "wf5Am": "흐리고 비", "wf5Pm": "흐림", "wf8": "맑음", "rnSt10": None}
    seen = []
    monkeypatch.setattr(kma_client, "_response", lambda url, params: seen.append(params["tmFc"]) or _envelope([item]))
    early = datetime(2026, 10, 19, 20, 0, tzinfo=SERVICE_TIMEZONE)
    days = kma_client.fetch_mid("11G00000", early)
    assert seen == ["202610191800"]  # 18시 발표분은 발표일 + 5일부터
    assert days[date(2026, 10, 24)] == MidDay(probabilities=[60, 40], phrases=["흐리고 비", "흐림"])
    assert days[date(2026, 10, 27)] == MidDay(probabilities=[10], phrases=["맑음"])
    assert date(2026, 10, 29) not in days  # 값이 None인 필드는 건너뛴다


def test_kma_errors_never_carry_the_service_key_or_url(monkeypatch):
    monkeypatch.setattr(settings, "kma_api_key", "SECRET-KEY-VALUE")

    class Boom(Exception):
        pass

    def raising(url, params, timeout):
        import httpx

        raise httpx.ConnectError("failed for https://x/?serviceKey=SECRET-KEY-VALUE")

    monkeypatch.setattr(kma_client.httpx, "get", raising)
    with pytest.raises(KmaError) as info:
        kma_client._response(kma_client.SHORT_URL, {})
    assert "SECRET-KEY-VALUE" not in str(info.value) and "ConnectError" in str(info.value) and info.value.__cause__ is None

    class Fake:
        status_code = 200
        text = "<xml/>"

        def json(self):
            raise ValueError("not json")

    monkeypatch.setattr(kma_client.httpx, "get", lambda *a, **k: Fake())
    with pytest.raises(KmaError, match="형식"):
        kma_client._response(kma_client.SHORT_URL, {})
    Fake.status_code = 500
    with pytest.raises(KmaError, match="500"):
        kma_client._response(kma_client.SHORT_URL, {})


# ---- 조립 (가짜 기상청) ----

TOMORROW = NOW.date() + timedelta(days=1)


def _day(rain=(), snow=(), hours=24):
    return ShortDay(rain_mm=list(rain), snow_cm=list(snow), hours={f"{h:02d}00" for h in range(hours)})


class FakeKma:
    def __init__(self):
        self.short_calls, self.mid_calls = [], []
        self.short = {}
        self.mid = {}
        self.failing_short, self.failing_mid = set(), set()

    def fetch_short(self, nx, ny, now):
        self.short_calls.append((nx, ny))
        if (nx, ny) in self.failing_short:
            raise KmaError("단기 실패")
        return self.short.get((nx, ny), {TOMORROW + timedelta(days=i): _day() for i in range(3)})

    def fetch_mid(self, reg_id, now):
        self.mid_calls.append(reg_id)
        if reg_id in self.failing_mid:
            raise KmaError("중기 실패")
        return self.mid.get(reg_id, {TOMORROW + timedelta(days=i): MidDay([10], ["맑음"]) for i in range(3, 9)})


@pytest.fixture
def kma(monkeypatch):
    fake = FakeKma()
    monkeypatch.setattr(settings, "kma_api_key", "k")
    monkeypatch.setattr(kma_client, "fetch_short", fake.fetch_short)
    monkeypatch.setattr(kma_client, "fetch_mid", fake.fetch_mid)
    return fake


def _by_day(data):
    return {f.date: {r.region: r for r in f.regions} for f in data.forecasts}


def test_precheck_covers_nine_days_and_five_regions_from_tomorrow(kma):
    data = build_precheck(NOW, ForecastCache())
    assert [f.date for f in data.forecasts] == [TOMORROW + timedelta(days=i) for i in range(9)]
    assert all([r.region for r in f.regions] == [r.name for r in REGIONS] for f in data.forecasts)
    assert all(not r.has_issue and r.reasons == [] for f in data.forecasts for r in f.regions)
    assert data.generated_at.tzinfo is not None and data.generated_at.microsecond == 0


def test_short_term_is_used_for_full_days_and_mid_term_for_the_rest(kma):
    seoul = next(r for r in REGIONS if r.name == "REGION-SEOUL")
    kma.short[seoul.short_grids[0]] = {TOMORROW: _day(rain=[35.0] + [0.0] * 23), TOMORROW + timedelta(days=3): _day(rain=[40.0], hours=8)}
    kma.mid[seoul.mid_reg_id] = {TOMORROW + timedelta(days=3): MidDay([90], ["비"]), TOMORROW + timedelta(days=4): MidDay([80], ["흐리고 눈"])}
    days = _by_day(build_precheck(NOW, ForecastCache()))
    assert [x.detail for x in days[TOMORROW]["REGION-SEOUL"].reasons] == ["heavy_rain"]
    # 예보가 8시간뿐인 날짜는 단기가 아니라 중기로 판정한다
    assert [x.detail for x in days[TOMORROW + timedelta(days=3)]["REGION-SEOUL"].reasons] == ["rain_likely"]
    assert [x.detail for x in days[TOMORROW + timedelta(days=4)]["REGION-SEOUL"].reasons] == ["snow_likely"]
    assert days[TOMORROW]["REGION-SEOUL"].reasons[0].category == "weather_risk"


def test_jeju_is_an_issue_when_either_grid_has_one(kma):
    jeju = next(r for r in REGIONS if r.name == "REGION-JEJU")
    assert len(jeju.short_grids) == 2
    kma.short[jeju.short_grids[1]] = {TOMORROW: _day(snow=[6.0])}
    kma.short[jeju.short_grids[0]] = {TOMORROW: _day()}
    jeju_day = _by_day(build_precheck(NOW, ForecastCache()))[TOMORROW]["REGION-JEJU"]
    assert jeju_day.has_issue and [x.detail for x in jeju_day.reasons] == ["heavy_snow"]
    kma.failing_short = {jeju.short_grids[1]}  # 한 격자가 실패하면 남은 격자로 판정한다
    assert _by_day(build_precheck(NOW, ForecastCache()))[TOMORROW]["REGION-JEJU"].has_issue is False


def test_partial_failures_drop_only_what_could_not_be_judged(kma):
    busan = next(r for r in REGIONS if r.name == "REGION-BUSAN")
    kma.failing_mid = {busan.mid_reg_id}
    days = _by_day(build_precheck(NOW, ForecastCache()))
    assert "REGION-BUSAN" in days[TOMORROW] and "REGION-BUSAN" not in days[TOMORROW + timedelta(days=4)]  # 중기 구간만 빠진다
    assert "REGION-SEOUL" in days[TOMORROW + timedelta(days=4)]
    kma.failing_mid = set()
    kma.failing_short = {grid for r in REGIONS for grid in r.short_grids}
    days = _by_day(build_precheck(NOW, ForecastCache()))
    assert TOMORROW not in days and TOMORROW + timedelta(days=3) in days  # 단기 구간 날짜는 통째로 빠진다
    assert all(not r.has_issue for f in days.values() for r in f.values())  # 실패를 has_issue false로 채우지 않는다


def test_everything_failing_or_a_missing_key_is_unavailable(kma, monkeypatch):
    kma.failing_short = {grid for r in REGIONS for grid in r.short_grids}
    kma.failing_mid = {r.mid_reg_id for r in REGIONS}
    with pytest.raises(PrecheckUnavailable):
        build_precheck(NOW, ForecastCache())
    monkeypatch.setattr(settings, "kma_api_key", "  ")
    with pytest.raises(PrecheckUnavailable, match="KMA_API_KEY"):
        build_precheck(NOW, ForecastCache())


def test_the_cache_serves_repeated_triggers_expires_and_does_not_store_failures(kma):
    clock = [0.0]
    cache = ForecastCache(clock=lambda: clock[0])
    build_precheck(NOW, cache)
    first = (len(kma.short_calls), len(kma.mid_calls))
    assert first == (6, 5)  # 단기 격자 6개(제주 2개), 중기 구역 5개
    build_precheck(NOW, cache)
    assert (len(kma.short_calls), len(kma.mid_calls)) == first  # 두 번째 트리거는 기상청을 부르지 않는다
    clock[0] = settings.precheck_cache_ttl_seconds + 1
    build_precheck(NOW, cache)
    assert (len(kma.short_calls), len(kma.mid_calls)) == (12, 10)
    cache.clear()
    kma.failing_mid = {REGIONS[0].mid_reg_id}
    build_precheck(NOW, cache)
    before = len(kma.mid_calls)
    kma.failing_mid = set()
    build_precheck(NOW, cache)
    assert len(kma.mid_calls) == before + 1  # 실패한 구역만 다시 받고 성공한 것은 캐시를 쓴다


def test_cache_fetches_a_key_once_under_concurrent_triggers():
    import threading

    cache, calls = ForecastCache(), []
    gate = threading.Event()

    def fetch():
        calls.append(1)
        gate.wait(1)
        return "v"

    results = []
    threads = [threading.Thread(target=lambda: results.append(cache.get(("k",), fetch, 60))) for _ in range(5)]
    [t.start() for t in threads]
    gate.set()
    [t.join() for t in threads]
    assert len(calls) == 1 and len(results) == 5


# ---- 계약 ----

def test_regions_match_the_place_selection_mapping_and_the_collection_centers():
    assert {r.name for r in REGIONS} == set(PRECHECK_REGION_TO_REGION)
    assert {PRECHECK_REGION_TO_REGION[r.name] for r in REGIONS} == set(E_Region)
    for region in REGIONS:
        areas = REGION_TO_COLLECTION_AREAS[PRECHECK_REGION_TO_REGION[region.name]]
        expected = []
        for area in areas:
            points = HEX_GRID_POINTS[area]
            expected.append(convert_to_grid(sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points)))
        assert list(region.short_grids) == expected, region.name  # 격자는 수집 지점의 중심에서 나온다


def test_the_precheck_response_can_be_forwarded_as_external_factors(kma):
    seoul = next(r for r in REGIONS if r.name == "REGION-SEOUL")
    kma.short[seoul.short_grids[0]] = {TOMORROW + timedelta(days=1): _day(rain=[40.0])}
    data = build_precheck(NOW, ForecastCache())
    forwarded = External_Factors.model_validate(Precheck_Response_Data.model_validate(data.model_dump(mode="json")).model_dump(mode="json"))
    assert outdoor_excluded_days(forwarded, E_Region.SEOUL, TOMORROW, 9) == frozenset({1})
    assert outdoor_excluded_days(forwarded, E_Region.BUSAN, TOMORROW, 9) == frozenset()


def test_precheck_endpoint_returns_the_envelope_503_and_requires_the_token(kma, monkeypatch):
    monkeypatch.setattr(settings, "internal_service_token", "t")
    monkeypatch.setattr(precheck_service, "_cache", ForecastCache())
    client = TestClient(app)
    headers = {"x-internal-token": "t"}
    assert client.post("/trips/precheck").status_code == 401
    response = client.post("/trips/precheck", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["status_code"] == 200 and set(body["data"]) == {"generated_at", "forecasts"}
    assert body["data"]["generated_at"].endswith("Z") and len(body["data"]["forecasts"]) == 9

    def unavailable():
        raise PrecheckUnavailable("기상청 예보를 하나도 받지 못함")

    monkeypatch.setattr(trips_router, "build_precheck", unavailable)
    failed = client.post("/trips/precheck", headers=headers)
    assert failed.status_code == 503 and "기상청" in failed.json()["detail"]
