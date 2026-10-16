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

    def generate(self, header, slots, width=10):
        self.calls += 1
        if self.behavior == "fail":
            raise InferenceUnavailable("x")
        return ["A"] * len(slots)


class _Cpu:
    def generate(self, header, slots, width=10):
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
