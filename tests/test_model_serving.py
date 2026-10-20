import math

import pytest

from app.trips.services.model import generation
from app.trips.services.model.inference import (
    CircuitBreaker,
    InferenceRouter,
    InferenceUnavailable,
    LocalEngine,
    RunPodEngine,
)
from scripts.semantic_id.tokens import VOCAB_SIZE, place_tokens

CODES = {"A": (1, 1, 1, 0), "B": (2, 2, 2, 0), "C": (3, 3, 3, 0)}


def _prefer(gid):
    # 지정한 장소의 토큰들을 높게 주는 가짜 모델
    wanted = set(place_tokens(CODES[gid]))

    def fn(prefixes):
        return [[math.log(0.9) if t in wanted else math.log(1e-6) for t in range(VOCAB_SIZE)] for _ in prefixes]

    return fn


def test_generate_itinerary_uses_candidates_only_and_never_repeats():
    header = generation.preference_header(["HISTORY_CULTURE", "ACTIVITY"], "서울")
    out = generation.generate_itinerary(header, [["A", "B"], ["A", "B"], ["Z"], []], CODES, _prefer("A"))
    assert out[0] == "A"
    assert out[1] == "B"  # A는 이미 썼으므로 남은 후보
    assert out[2] is None and out[3] is None  # 코드 사전에 없거나 후보가 없는 슬롯
    assert out == generation.generate_itinerary(header, [["A", "B"], ["A", "B"], ["Z"], []], CODES, _prefer("A"))


class _Gpu:
    def __init__(self, behavior):
        self.behavior, self.calls = behavior, 0

    def generate(self, header, slots, width=10, first_k=1, seed=None):
        self.calls += 1
        if self.behavior == "fail":
            raise InferenceUnavailable("x")
        return ["A"] * len(slots)


class _Cpu:
    def generate(self, header, slots, width=10, first_k=1, seed=None):
        return ["B"] * len(slots)


def test_router_prefers_gpu_then_falls_back_to_cpu():
    ok = InferenceRouter(_Gpu("ok"), _Cpu())
    assert ok.generate([1], [["A"]]) == (["A"], "gpu")
    bad = InferenceRouter(_Gpu("fail"), _Cpu())
    assert bad.generate([1], [["A"]]) == (["B"], "cpu")
    with pytest.raises(InferenceUnavailable):
        InferenceRouter(_Gpu("fail"), None).generate([1], [["A"]])


def test_circuit_breaker_blocks_gpu_after_consecutive_failures_and_recovers():
    now = [0.0]
    gpu = _Gpu("fail")
    router = InferenceRouter(gpu, _Cpu(), CircuitBreaker(3, 60.0, clock=lambda: now[0]))
    for _ in range(3):
        router.generate([1], [["A"]])
    assert gpu.calls == 3
    router.generate([1], [["A"]])
    assert gpu.calls == 3  # 막힌 동안은 GPU를 부르지 않는다
    now[0] = 61.0
    gpu.behavior = "ok"
    assert router.generate([1], [["A"]]) == (["A"], "gpu")  # 쿨다운 뒤 시험 호출이 성공하면 복구
    assert gpu.calls == 4 and router.breaker.failures == 0


def test_runpod_engine_validates_response_and_wraps_errors():
    good = RunPodEngine("http://x", "", 1.0, post=lambda url, payload, timeout: {"status": "COMPLETED", "output": {"places": ["A", None]}})
    assert good.generate([1], [["A"], []]) == ["A", None]
    wrong_len = RunPodEngine("http://x", "", 1.0, post=lambda *a: {"status": "COMPLETED", "output": {"places": ["A"]}})
    with pytest.raises(InferenceUnavailable):
        wrong_len.generate([1], [["A"], []])
    failed = RunPodEngine("http://x", "", 1.0, post=lambda *a: {"status": "FAILED"})
    with pytest.raises(InferenceUnavailable):
        failed.generate([1], [["A"]])

    def boom(*a):
        raise TimeoutError

    with pytest.raises(InferenceUnavailable):
        RunPodEngine("http://x", "", 1.0, post=boom).generate([1], [["A"]])


def test_local_engine_rejects_when_concurrency_is_exhausted():
    engine = LocalEngine.__new__(LocalEngine)  # onnxruntime 없이 동시 실행 상한만 확인한다
    import threading

    engine._slots = threading.BoundedSemaphore(1)
    engine.codes = CODES
    engine._slots.acquire()
    with pytest.raises(InferenceUnavailable):
        engine.generate([1], [["A"]])


def test_selection_limits_non_food_slots_to_top_two_preference_categories(monkeypatch):
    from types import SimpleNamespace

    from app.trips.schemas.schemas import (
        E_Preference,
        E_Region,
        Member_Survey,
        User,
    )
    from app.trips.services.model import selection
    from app.trips.services.slot_plan import build_slot_plan

    def place(gid):
        return SimpleNamespace(id=gid, matched_preferences=[], selected_for=[])

    pools = {
        "HISTORY_CULTURE": [place("h1"), place("both")],
        "NATURE_HEALING": [place("n1"), place("both")],
        "ACTIVITY": [place("a1")],
        "CONVENIENCE_SHOPPING": [place("c1")],
        "FOOD": [place("f1"), place("f2")],
    }
    monkeypatch.setattr(selection, "get_DB_places_by_category", lambda category, limit, region, **kw: list(pools[category.value]))
    seen = {}

    class FakeRouter:
        def generate(self, header, slots, width=10, first_k=1, seed=None):
            seen["slots"] = slots
            seen["first_k"], seen["seed"] = first_k, seed
            return [s[0] if s else None for s in slots], "cpu"

    members = [Member_Survey(user=User(user_id="u"), survey_result=[5, 5, 5, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1])]
    plan = build_slot_plan(members, selection.model_slot_plan_seed())
    places, route = selection.select_places_by_model(plan, [], E_Region.SEOUL, members, FakeRouter())
    non_food = [s for i, s in enumerate(seen["slots"]) if i not in (1, 4)]
    top_two = [c.value for c in plan.daytime_category_ranking[:2]]
    expected = {g for c in top_two for g in [p.id for p in pools[c]]}
    assert all(set(s) == expected and len(s) == len(set(s)) for s in non_food)  # 취향 상위 둘의 합집합, 중복 없이
    assert all(p.id in expected or p.id in ("f1", "f2") for p in places)
    assert seen["slots"][1] == seen["slots"][4] == ["f1", "f2"]
    assert route == "cpu" and len(places) == len(plan.items)
    assert all(E_Preference.FOOD in p.matched_preferences for i, p in enumerate(places) if i in (1, 4))


def test_selection_adds_next_ranked_category_when_top_two_have_too_few_places(monkeypatch):
    from types import SimpleNamespace

    from app.trips.schemas.schemas import E_Region, Member_Survey, User
    from app.trips.services.model import selection
    from app.trips.services.slot_plan import build_slot_plan

    def place(gid):
        return SimpleNamespace(id=gid, matched_preferences=[], selected_for=[])

    members = [Member_Survey(user=User(user_id="u"), survey_result=[5, 5, 5, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1])]
    plan = build_slot_plan(members, selection.model_slot_plan_seed())
    first, second, third = (c.value for c in plan.daytime_category_ranking[:3])
    pools = {c.value: [] for c in plan.daytime_category_ranking}
    pools["FOOD"] = [place("f1"), place("f2")]
    pools[first] = [place("x1")]  # 상위 둘을 합쳐도 3곳이 안 된다
    pools[third] = [place("y1"), place("y2")]
    monkeypatch.setattr(selection, "get_DB_places_by_category", lambda category, limit, region, **kw: list(pools[category.value]))
    seen, trace = {}, {}

    class FakeRouter:
        def generate(self, header, slots, width=10, first_k=1, seed=None):
            seen["slots"] = slots
            seen["first_k"], seen["seed"] = first_k, seed
            return [s[0] if s else None for s in slots], "cpu"

    selection.select_places_by_model(plan, [], E_Region.SEOUL, members, FakeRouter(), trace=trace)
    assert trace["daytime_categories"] == [first, second, third]
    assert set(seen["slots"][0]) == {"x1", "y1", "y2"}



def _ranked_codes(n):
    # c1이 1..n인 장소 n곳. 가짜 모델은 c1이 작을수록 높은 점수를 준다(1번 장소가 1위)
    return {chr(ord("A") + i): (i + 1, 1, 1, 0) for i in range(n)}


def _prefer_small_c1(n):
    first_tokens = [place_tokens((i + 1, 1, 1, 0))[0] for i in range(n)]

    def fn(prefixes):
        out = []
        for _ in prefixes:
            lp = [math.log(1e-6)] * VOCAB_SIZE
            for i, t in enumerate(first_tokens):
                lp[t] = math.log(0.5) - 0.3 * i  # 앞 번호일수록 높다
            out.append(lp)
        return out

    return fn


def test_first_slot_picks_among_top_k_with_a_fixed_seed():
    codes = _ranked_codes(8)
    slots = [list(codes), list(codes)]
    fn = _prefer_small_c1(8)
    header = generation.preference_header(["HISTORY_CULTURE", "ACTIVITY"], "서울")
    top1 = generation.generate_itinerary(header, slots, codes, fn)
    assert top1[0] == "A"  # first_k=1은 모델 1위
    picks = {generation.generate_itinerary(header, slots, codes, fn, first_k=3, seed=s)[0] for s in range(40)}
    assert picks == {"A", "B", "C"}  # 상위 3곳 안에서만, 시드에 따라 달라진다
    one = generation.generate_itinerary(header, slots, codes, fn, first_k=3, seed=7)
    assert one == generation.generate_itinerary(header, slots, codes, fn, first_k=3, seed=7)  # 같은 시드는 같은 결과
    assert one[1] != one[0]  # 둘째 슬롯은 첫 장소를 제외한 후보에서 고른다


def test_first_k_one_ignores_the_seed_and_later_slots_stay_deterministic():
    codes = _ranked_codes(6)
    slots = [list(codes), list(codes), list(codes)]
    fn = _prefer_small_c1(6)
    header = generation.preference_header(["HISTORY_CULTURE", "ACTIVITY"], "서울")
    assert generation.generate_itinerary(header, slots, codes, fn, first_k=1, seed=1) == generation.generate_itinerary(header, slots, codes, fn, first_k=1, seed=99)
    # first_k가 커도 둘째 슬롯부터는 모델 1위(남은 후보 중)로 고정된다
    result = generation.generate_itinerary(header, slots, codes, fn, first_k=4, seed=3)
    remaining = [g for g in codes if g != result[0]]
    assert result[1] == remaining[0] and result[2] == remaining[1]


def test_first_k_is_clamped_and_empty_first_slot_stays_none():
    codes = _ranked_codes(3)
    fn = _prefer_small_c1(3)
    header = generation.preference_header(["HISTORY_CULTURE", "ACTIVITY"], "서울")
    out = generation.generate_itinerary(header, [list(codes)], codes, fn, first_k=10_000, seed=1)
    assert out[0] in codes  # 상한을 넘어도 후보 수만큼만
    assert generation.generate_itinerary(header, [[], list(codes)], codes, fn, first_k=5, seed=1)[0] is None


def test_selection_passes_first_k_and_seed_to_the_router(monkeypatch):
    from types import SimpleNamespace

    from app.trips.schemas.schemas import E_Region, Member_Survey, User
    from app.trips.services.model import selection
    from app.trips.services.slot_plan import build_slot_plan

    pools = {c: [SimpleNamespace(id=f"{c}{i}", matched_preferences=[], selected_for=[]) for i in range(3)] for c in
             ("HISTORY_CULTURE", "NATURE_HEALING", "ACTIVITY", "CONVENIENCE_SHOPPING", "FOOD")}
    monkeypatch.setattr(selection, "get_DB_places_by_category", lambda category, limit, region, **kw: list(pools[category.value]))
    seen = {}

    class FakeRouter:
        def generate(self, header, slots, width=10, first_k=1, seed=None):
            seen["first_k"], seen["seed"] = first_k, seed
            return [s[0] if s else None for s in slots], "cpu"

    members = [Member_Survey(user=User(user_id="u"), survey_result=[5, 5, 5, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1])]
    plan = build_slot_plan(members, selection.model_slot_plan_seed())
    selection.select_places_by_model(plan, [], E_Region.SEOUL, members, FakeRouter(), first_k=30, seed=12345)
    assert seen == {"first_k": 30, "seed": 12345}


def test_generate_trip_shares_used_places_across_days_and_restarts_the_context_each_day():
    header = generation.preference_header(["HISTORY_CULTURE", "ACTIVITY"], "서울")
    seen = []
    prefer_a = _prefer("A")

    def fn(prefixes):
        seen.extend(len(p) for p in prefixes)
        return prefer_a(prefixes)

    out = generation.generate_trip(header, [[["A", "B"]], [["A", "B"]]], CODES, fn)
    assert out == [["A"], ["B"]]  # 첫날 A를 썼으면 둘째 날은 A가 후보에서 빠진다
    assert seen.count(len(header)) == 2  # 일자마다 앞 일자의 장소 없이 헤더만으로 시작한다


def test_generate_trip_derives_the_first_place_seed_per_day():
    codes = _ranked_codes(8)
    fn = _prefer_small_c1(8)
    header = generation.preference_header(["HISTORY_CULTURE", "ACTIVITY"], "서울")
    days = [[list(codes)], [list(codes)], [list(codes)]]
    trip = generation.generate_trip(header, days, codes, fn, first_k=3, seed=10)
    used: set[str] = set()
    expected = [
        generation.generate_itinerary(header, day, codes, fn, first_k=3, seed=10 + index, used=used)
        for index, day in enumerate(days)
    ]
    assert trip == expected
    assert len({place for day in trip for place in day}) == 3  # 여행 전체에서 중복 없음
    assert trip == generation.generate_trip(header, days, codes, fn, first_k=3, seed=10)  # 같은 시드는 같은 결과


def test_resolve_days_turns_pool_names_into_candidates_and_rejects_bad_input():
    pools = {"daytime": ["A", "B"], "food": ["C"]}
    assert generation.resolve_days(pools, [["daytime", "food"], ["food"]]) == [[["A", "B"], ["C"]], [["C"]]]
    bad_inputs = [
        (None, [["daytime"]]),  # pools가 없음
        ({"daytime": "A"}, [["daytime"]]),  # 풀이 목록이 아님
        (pools, []),  # 일수 0
        (pools, [["daytime"]] * (generation.MAX_TRIP_DAYS + 1)),  # 일수 초과
        (pools, [[]]),  # 슬롯 0
        (pools, [["daytime"] * (generation.MAX_DAY_SLOTS + 1)]),  # 하루 슬롯 초과
        (pools, [["unknown"]]),  # 없는 풀 이름
        (pools, [[1]]),  # 풀 이름이 문자열이 아님
    ]
    for bad_pools, bad_days in bad_inputs:
        with pytest.raises(ValueError):
            generation.resolve_days(bad_pools, bad_days)
    assert len(generation.resolve_days(pools, [["daytime"]] * generation.MAX_TRIP_DAYS)) == generation.MAX_TRIP_DAYS


def test_run_trip_request_returns_days_or_an_error_without_raising():
    header = generation.preference_header(["HISTORY_CULTURE", "ACTIVITY"], "서울")
    payload = {"header": header, "pools": {"all": ["A", "B", "C"]}, "days": [["all", "all"], ["all"]], "width": 10}
    assert generation.run_trip_request(payload, CODES, _prefer("A")) == {"days": [["A", "B"], ["C"]]}
    for broken in (
        {**payload, "header": None},
        {**payload, "days": [["missing"]]},
        {**payload, "seed": "1"},
    ):
        assert "error" in generation.run_trip_request(broken, CODES, _prefer("A"))


def test_runpod_engine_generate_trip_sends_pools_once_and_validates_the_shape():
    sent = []

    def post(url, payload, timeout):
        sent.append(payload)
        return {"status": "COMPLETED", "output": {"days": [["A", None], ["B"]]}}

    engine = RunPodEngine("http://x", "", 1.0, post=post)
    pools, days = {"all": ["A", "B"]}, [["all", "all"], ["all"]]
    assert engine.generate_trip([1], pools, days, first_k=30, seed=5) == [["A", None], ["B"]]
    assert sent[0]["input"] == {"header": [1], "pools": pools, "days": days, "width": 10, "first_k": 30, "seed": 5}
    for bad_output in ({"days": [["A", None]]}, {"days": [["A"], ["B"]]}, {"error": "x"}, {}):
        broken = RunPodEngine("http://x", "", 1.0, post=lambda *a, out=bad_output: {"status": "COMPLETED", "output": out})
        with pytest.raises(InferenceUnavailable):
            broken.generate_trip([1], pools, days)  # 일수·일자별 슬롯 수가 다르거나 이전 이미지가 보낸 오류 응답


def test_local_engine_generate_trip_uses_the_shared_generation_and_the_concurrency_limit():
    import threading

    engine = LocalEngine.__new__(LocalEngine)
    engine._slots = threading.BoundedSemaphore(1)
    engine.codes = CODES
    engine.next_logprobs = lambda: _prefer("A")
    assert engine.generate_trip([1], {"all": ["A", "B"]}, [["all"], ["all"]]) == [["A"], ["B"]]
    engine._slots.acquire()
    with pytest.raises(InferenceUnavailable):
        engine.generate_trip([1], {"all": ["A"]}, [["all"]])


def test_router_generate_trip_falls_back_from_gpu_to_cpu_and_counts_failures():
    class TripGpu:
        calls = 0

        def generate_trip(self, header, pools, days, width=10, first_k=1, seed=None):
            TripGpu.calls += 1
            raise InferenceUnavailable("x")

    class TripCpu:
        def generate_trip(self, header, pools, days, width=10, first_k=1, seed=None):
            return [["B"]]

    router = InferenceRouter(TripGpu(), TripCpu(), CircuitBreaker(2, 60.0, clock=lambda: 0.0))
    assert router.generate_trip([1], {"all": ["B"]}, [["all"]]) == ([["B"]], "cpu")
    router.generate_trip([1], {"all": ["B"]}, [["all"]])
    router.generate_trip([1], {"all": ["B"]}, [["all"]])
    assert TripGpu.calls == 2  # 연속 2번 실패하면 막혀서 더 부르지 않는다
    with pytest.raises(InferenceUnavailable):
        InferenceRouter(TripGpu(), None).generate_trip([1], {"all": ["B"]}, [["all"]])
