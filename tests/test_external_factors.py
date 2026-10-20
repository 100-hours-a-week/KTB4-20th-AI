from datetime import date

from app.trips.schemas.api import External_Factors
from app.trips.schemas.schemas import E_Region
from app.trips.services.external_factors import (
    PRECHECK_REGION_TO_REGION,
    outdoor_excluded_days,
)
from app.trips.services.slot_plan import build_trip_slot_plan
from app.trips.services.trip_orchestrator import derive_seed, is_valid_model_trip
from tests.test_trip_selection import MEMBERS, make_place

START = date(2026, 11, 3)


def _factors(*entries):
    forecasts: dict[str, list] = {}
    for day, region, has_issue in entries:
        forecasts.setdefault(day, []).append({"region": region, "has_issue": has_issue})
    return External_Factors.model_validate(
        {"generated_at": "2026-10-10T10:00:03Z", "forecasts": [{"date": d, "regions": r} for d, r in forecasts.items()]}
    )


def test_every_precheck_region_maps_to_a_service_region():
    assert set(PRECHECK_REGION_TO_REGION.values()) == set(E_Region)


def test_signal_days_are_trip_day_numbers_for_the_requested_region_only():
    factors = _factors(
        ("2026-11-03", "REGION-SEOUL", True),
        ("2026-11-04", "REGION-SEOUL", False),
        ("2026-11-05", "REGION-SEOUL", True),
        ("2026-11-05", "REGION-BUSAN", False),
        ("2026-11-04", "REGION-BUSAN", True),
        ("2026-11-20", "REGION-SEOUL", True),
        ("2026-11-03", "REGION-???", True),
    )
    assert outdoor_excluded_days(factors, E_Region.SEOUL, START, 3) == frozenset({0, 2})
    assert outdoor_excluded_days(factors, E_Region.BUSAN, START, 3) == frozenset({1})
    assert outdoor_excluded_days(factors, E_Region.JEJU, START, 3) == frozenset()
    assert outdoor_excluded_days(factors, E_Region.SEOUL, START, 1) == frozenset({0})  # 여행 날짜 밖은 무시
    assert outdoor_excluded_days(None, E_Region.SEOUL, START, 3) == frozenset()


def test_a_model_plan_is_valid_only_when_every_non_evening_slot_is_filled():
    plan = build_trip_slot_plan(MEMBERS, 3)  # 6슬롯(저녁 이후 포함), 6슬롯, 2슬롯

    def trip(*counts):
        return [[make_place(f"p{d}_{i}") for i in range(count)] for d, count in enumerate(counts)]

    assert is_valid_model_trip(plan, trip(5, 5, 2))  # 저녁 이후는 비어도 된다
    assert is_valid_model_trip(plan, trip(6, 6, 2))
    assert not is_valid_model_trip(plan, trip(5, 4, 2))
    assert not is_valid_model_trip(plan, trip(5, 5, 1))  # 마지막 날은 2곳이 필요하다
    assert not is_valid_model_trip(plan, trip(5, 5))  # 일자 수가 다르다


def test_derived_seeds_are_stable_and_differ_by_label():
    assert derive_seed(1, "plan1") == derive_seed(1, "plan1")
    assert len({derive_seed(1, "plan1"), derive_seed(1, "plan2"), derive_seed(1, "ties"), derive_seed(2, "plan1")}) == 4
