import random
from types import SimpleNamespace

import pytest

from app.trips.schemas.schemas import (
    E_Breaker,
    E_Preference,
    E_Region,
    Member_Survey,
    User,
)
from app.trips.services import trip_rule
from app.trips.services.model import trip_selection
from app.trips.services.slot_plan import E_Time_Slot, build_trip_slot_plan
from app.trips.services.trip_rule import OUTDOOR_TYPE_VALUES

# 야간 점수(10번 문항)가 3.0을 넘어 저녁 이후 슬롯이 생기는 대표 설문
MEMBERS = [Member_Survey(user=User(user_id="u"), survey_result=[5, 5, 5, 4, 4, 4, 3, 3, 3, 5, 2, 2, 1, 1, 1])]
NIGHT_OFF_MEMBERS = [Member_Survey(user=User(user_id="u"), survey_result=[5, 5, 5, 4, 4, 4, 3, 3, 3, 1, 2, 2, 1, 1, 1])]
PER_CATEGORY = 40


def _plan(num_days, members=MEMBERS, diversity=False):
    return build_trip_slot_plan(members, num_days, random.Random(0), diversity)


class FakeDb:
    """카테고리마다 PER_CATEGORY곳을 돌려주는 가짜 DB. 야외 type 제외가 걸리면 NATURE_HEALING은 비고(야외 목록에 전부 들어 있음),
    저녁 이후 풀은 evening_size곳이다."""

    def __init__(self, evening_size=40):
        self.calls, self.evening_size = [], evening_size

    def __call__(self, category, limit, region, excluded_types=None, required_types=None):
        excluded = set(excluded_types or ())
        self.calls.append((category, limit, excluded, required_types))
        if required_types:
            count = self.evening_size
            name = "night"
        elif category == E_Preference.NATURE_HEALING and excluded & OUTDOOR_TYPE_VALUES:
            return []
        else:
            count, name = PER_CATEGORY, category.value
        return [
            SimpleNamespace(id=f"{name}{i}", matched_preferences=[], selected_for=[]) for i in range(min(count, limit))
        ]


def _ids(trip):
    return [[place.id for place in day] for day in trip]


def test_trip_slot_plan_shortens_only_the_last_day_and_keeps_single_day_full():
    full = [(i.time_slot, i.category) for i in _plan(1).days[0].items]
    assert [slot for slot, _ in full] == [
        E_Time_Slot.MORNING, E_Time_Slot.LUNCH, E_Time_Slot.AFTERNOON_FIRST, E_Time_Slot.AFTERNOON_SECOND, E_Time_Slot.DINNER,
        E_Time_Slot.EVENING,
    ]
    plan = _plan(4)
    assert [len(day.items) for day in plan.days] == [6, 6, 6, 2]
    assert [i.time_slot for i in plan.days[3].items] == [E_Time_Slot.MORNING, E_Time_Slot.LUNCH]
    assert {day.daytime_category_ranking for day in plan.days} == {plan.days[0].daytime_category_ranking}
    assert [len(day.items) for day in _plan(3, NIGHT_OFF_MEMBERS).days] == [5, 5, 2]  # 야간 점수가 낮으면 저녁 이후 슬롯 없음
    with pytest.raises(ValueError):
        build_trip_slot_plan(MEMBERS, 0)


def test_trip_slot_plan_diversity_alternates_third_and_fourth_ranked_categories():
    plain, diverse = _plan(4), _plan(4, diversity=True)
    ranking = plain.days[0].daytime_category_ranking

    def second_afternoon(day):
        return next((i.category for i in day.items if i.time_slot == E_Time_Slot.AFTERNOON_SECOND), None)

    assert [second_afternoon(day) for day in plain.days] == [ranking[1], ranking[1], ranking[1], None]
    assert [second_afternoon(day) for day in diverse.days] == [ranking[2], ranking[3], ranking[2], None]
    assert [len(day.items) for day in diverse.days] == [6, 6, 6, 2]


def test_rule_trip_has_no_duplicates_and_is_deterministic_without_a_variant_seed(monkeypatch):
    monkeypatch.setattr(trip_rule, "get_DB_places_by_category", FakeDb())
    plan = _plan(9)
    trip = trip_rule.select_trip_by_slot_plan(plan, [], E_Region.SEOUL, MEMBERS)
    assert [len(day) for day in trip] == [6] * 8 + [2]
    flat = [place_id for day in _ids(trip) for place_id in day]
    assert len(flat) == len(set(flat))
    assert _ids(trip) == _ids(trip_rule.select_trip_by_slot_plan(plan, [], E_Region.SEOUL, MEMBERS))
    assert all(p.matched_preferences for day in trip for p in day)


def test_rule_trip_variant_depends_on_the_seed_and_stays_unique(monkeypatch):
    monkeypatch.setattr(trip_rule, "get_DB_places_by_category", FakeDb())
    plan = _plan(5)
    base = _ids(trip_rule.select_trip_by_slot_plan(plan, [], E_Region.SEOUL, MEMBERS))
    one = _ids(trip_rule.select_trip_by_slot_plan(plan, [], E_Region.SEOUL, MEMBERS, variant_seed=1))
    assert one == _ids(trip_rule.select_trip_by_slot_plan(plan, [], E_Region.SEOUL, MEMBERS, variant_seed=1))
    assert one != base
    assert one != _ids(trip_rule.select_trip_by_slot_plan(plan, [], E_Region.SEOUL, MEMBERS, variant_seed=2))
    flat = [place_id for day in one for place_id in day]
    assert len(flat) == len(set(flat))
    assert [len(day) for day in one] == [len(day) for day in base]


def test_rule_trip_excludes_outdoor_types_only_on_signal_days(monkeypatch):
    db = FakeDb()
    monkeypatch.setattr(trip_rule, "get_DB_places_by_category", db)
    plan = _plan(3, NIGHT_OFF_MEMBERS)
    trip = trip_rule.select_trip_by_slot_plan(plan, [], E_Region.SEOUL, NIGHT_OFF_MEMBERS, outdoor_excluded_days={1})
    assert all(E_Preference.NATURE_HEALING not in p.matched_preferences for p in trip[1])  # 신호가 있는 날
    assert [len(day) for day in trip] == [5, 5, 2]  # 후보가 모자라지 않으면 슬롯 수는 그대로(다음 순위 카테고리로 보충)
    flagged = [excluded for _, _, excluded, _ in db.calls if excluded & OUTDOOR_TYPE_VALUES]
    unflagged = [excluded for _, _, excluded, _ in db.calls if not excluded & OUTDOOR_TYPE_VALUES]
    assert flagged and unflagged  # 신호가 있는 날과 없는 날의 조회가 따로 있다
    deal_breaker_trip = FakeDb()
    monkeypatch.setattr(trip_rule, "get_DB_places_by_category", deal_breaker_trip)
    trip_rule.select_trip_by_slot_plan(plan, [E_Breaker.SEAFOOD], E_Region.SEOUL, NIGHT_OFF_MEMBERS)
    assert all("seafood_restaurant" in excluded for _, _, excluded, _ in deal_breaker_trip.calls)  # 회피조건은 모든 조회에 적용


def test_rule_trip_fills_evening_slots_only_while_night_places_remain(monkeypatch):
    monkeypatch.setattr(trip_rule, "get_DB_places_by_category", FakeDb(evening_size=2))
    trip = trip_rule.select_trip_by_slot_plan(_plan(5), [], E_Region.SEOUL, MEMBERS)
    evening_counts = [sum(1 for p in day if p.id.startswith("night")) for day in trip]
    assert evening_counts == [1, 1, 0, 0, 0]  # 후보 2곳이 앞 일자부터 채워지고 마지막 날에는 저녁 이후가 없다
    assert [len(day) for day in trip] == [6, 6, 5, 5, 2]


class FakeTripRouter:
    def __init__(self, none_for=()):
        self.none_for, self.sent = set(none_for), None

    def generate_trip(self, header, pools, days, width=10, first_k=1, seed=None):
        self.sent = {"header": header, "pools": pools, "days": days, "first_k": first_k, "seed": seed}
        used, chosen_days = set(), []
        for day_index, day in enumerate(days):
            chosen = []
            for slot_index, name in enumerate(day):
                gid = None if (day_index, slot_index) in self.none_for else next((g for g in pools[name] if g not in used), None)
                if gid:
                    used.add(gid)
                chosen.append(gid)
            chosen_days.append(chosen)
        return chosen_days, "gpu"


def test_model_trip_sends_named_pools_once_and_maps_the_answer_back(monkeypatch):
    monkeypatch.setattr(trip_selection, "get_DB_places_by_category", FakeDb())
    plan = _plan(3)
    router = FakeTripRouter(none_for={(0, 3)})
    trip, route = trip_selection.select_trip_by_model(plan, [], E_Region.SEOUL, MEMBERS, router, first_k=30, seed=7)
    assert route == "gpu" and router.sent["first_k"] == 30 and router.sent["seed"] == 7
    assert set(router.sent["pools"]) == {"daytime", "food", "evening"}
    assert [names[1] for names in router.sent["days"]] == ["food"] * 3
    assert [len(day) for day in trip] == [5, 6, 2]  # 모델이 못 고른 슬롯(None)은 건너뛴다
    flat = [place_id for day in _ids(trip) for place_id in day]
    assert len(flat) == len(set(flat))
    top_two = {c.value for c in plan.days[0].daytime_category_ranking[:2]}
    daytime_ids = set(router.sent["pools"]["daytime"])
    assert {i.rstrip("0123456789") for i in daytime_ids} == top_two  # 후보는 취향 상위 두 카테고리의 합집합
    assert all(
        p.matched_preferences == [E_Preference.FOOD] for day in trip for p in day if p.id.startswith(E_Preference.FOOD.value)
    )


def test_model_trip_uses_indoor_pools_on_signal_days_and_four_categories_for_the_diversity_plan(monkeypatch):
    db = FakeDb()
    monkeypatch.setattr(trip_selection, "get_DB_places_by_category", db)
    plan = _plan(3, NIGHT_OFF_MEMBERS)
    router = FakeTripRouter()
    trip_selection.select_trip_by_model(plan, [], E_Region.SEOUL, NIGHT_OFF_MEMBERS, router, outdoor_excluded_days={1})
    assert router.sent["days"][0][0] == "daytime" and router.sent["days"][1][0] == "daytime_indoor"
    assert router.sent["days"][1][1] == "food_indoor" and router.sent["days"][2][1] == "food"
    assert set(router.sent["pools"]) == {"daytime", "daytime_indoor", "food", "food_indoor"}  # 쓰는 풀만 보낸다

    plain_calls, diverse_calls = FakeDb(), FakeDb()
    monkeypatch.setattr(trip_selection, "get_DB_places_by_category", plain_calls)
    trip_selection.select_trip_by_model(plan, [], E_Region.SEOUL, NIGHT_OFF_MEMBERS, FakeTripRouter())
    monkeypatch.setattr(trip_selection, "get_DB_places_by_category", diverse_calls)
    trip_selection.select_trip_by_model(plan, [], E_Region.SEOUL, NIGHT_OFF_MEMBERS, FakeTripRouter(), diversity=True)

    def daytime_categories(db_calls):
        return {category for category, _, _, required in db_calls.calls if category != E_Preference.FOOD and not required}

    assert len(daytime_categories(plain_calls)) == 2
    assert len(daytime_categories(diverse_calls)) == 4
