"""모델 추론 경로: RunPod GPU, 컨테이너 CPU, 그리고 둘을 잇는 폴백 (ADR-0004).

  - CPU 경로: ONNX Runtime으로 AI 서버 컨테이너 안에서 추론한다.
  - GPU 경로: RunPod Serverless 엔드포인트(모델 전용)에 슬롯별 후보 목록을 보내 일정 전체를 받는다.
  - 라우터: GPU를 먼저 시도하고 타임아웃이나 오류가 나면 CPU로 넘긴다. 연속 실패가 쌓이면 일정 시간 GPU 호출을 막는다(서킷 브레이커).
    CPU 동시 실행 수에 상한을 둬서 폴백이 몰려도 서버를 못 쓰게 되지 않게 한다.
"""

import json
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

from app.trips.services.model.generation import (
    BEAM_WIDTH,
    NextLogprobs,
    generate_itinerary,
)

logger = logging.getLogger(__name__)


class InferenceUnavailable(Exception):
    """이 경로로는 일정을 만들 수 없음(오류, 타임아웃, 막힌 상태, 용량 초과)."""


class LocalEngine:
    """ONNX Runtime CPU 추론. 모델과 코드 사전은 model_dir의 model.onnx, codes.json에서 읽는다."""

    def __init__(self, model_dir: Path, max_concurrency: int = 2):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1  # 서버 2 vCPU를 다른 컨테이너와 나눠 쓴다
        self.session = ort.InferenceSession(str(model_dir / "model.onnx"), options, providers=["CPUExecutionProvider"])
        bundle = json.loads((model_dir / "codes.json").read_text(encoding="utf-8"))
        self.codes: dict[str, list[int]] = bundle["codes"]
        self.version: dict = bundle.get("version", {})
        self._slots = threading.BoundedSemaphore(max_concurrency)

    def next_logprobs(self) -> NextLogprobs:
        import numpy as np

        def fn(prefixes: list[list[int]]):
            return self.session.run(None, {"tokens": np.asarray(prefixes, dtype=np.int64)})[0]

        return fn

    def generate(self, header: list[int], slot_candidates: list[list[str]], width: int = BEAM_WIDTH) -> list[str | None]:
        if not self._slots.acquire(blocking=False):
            raise InferenceUnavailable("CPU 동시 실행 상한 초과")
        try:
            return generate_itinerary(header, slot_candidates, self.codes, self.next_logprobs(), width)
        finally:
            self._slots.release()


class RunPodEngine:
    """RunPod Serverless 동기 호출(/runsync). 후보 목록과 헤더를 보내고 슬롯별 장소를 받는다."""

    def __init__(self, url: str, api_key: str, timeout_seconds: float, post: Callable | None = None):
        self.url, self.api_key, self.timeout = url.rstrip("/"), api_key, timeout_seconds
        self._post = post

    def generate(self, header: list[int], slot_candidates: list[list[str]], width: int = BEAM_WIDTH) -> list[str | None]:
        payload = {"input": {"header": header, "slots": slot_candidates, "width": width}}
        try:
            if self._post is not None:
                body = self._post(f"{self.url}/runsync", payload, self.timeout)
            else:
                import httpx

                response = httpx.post(
                    f"{self.url}/runsync", json=payload, timeout=self.timeout,
                    headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
                )
                response.raise_for_status()
                body = response.json()
        except Exception as error:  # 타임아웃, 연결 오류, 상태 코드 오류를 모두 같은 실패로 본다
            raise InferenceUnavailable(f"RunPod 호출 실패: {type(error).__name__}") from error
        if body.get("status") != "COMPLETED":
            raise InferenceUnavailable(f"RunPod 상태: {body.get('status')}")
        places = (body.get("output") or {}).get("places")
        if not isinstance(places, list) or len(places) != len(slot_candidates):
            raise InferenceUnavailable("RunPod 응답 형식 오류")
        return places


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, cooldown_seconds: float = 60.0, clock: Callable[[], float] = time.monotonic):
        self.failure_threshold, self.cooldown, self.clock = failure_threshold, cooldown_seconds, clock
        self.failures, self.opened_at = 0, None
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            if self.opened_at is None:
                return True
            return self.clock() - self.opened_at >= self.cooldown  # 쿨다운이 지나면 한 번 시험 호출을 허용한다

    def record_success(self) -> None:
        with self._lock:
            self.failures, self.opened_at = 0, None

    def record_failure(self) -> None:
        with self._lock:
            self.failures += 1
            if self.failures >= self.failure_threshold:
                self.opened_at = self.clock()


class InferenceRouter:
    """GPU 우선, 실패하면 CPU. 어느 경로가 썼는지 route를 함께 돌려준다."""

    def __init__(self, gpu: RunPodEngine | None, cpu: LocalEngine | None, breaker: CircuitBreaker | None = None):
        self.gpu, self.cpu, self.breaker = gpu, cpu, breaker or CircuitBreaker()

    def generate(self, header: list[int], slot_candidates: list[list[str]], width: int = BEAM_WIDTH) -> tuple[list[str | None], str]:
        if self.gpu is not None and self.breaker.allow():
            try:
                places = self.gpu.generate(header, slot_candidates, width)
                self.breaker.record_success()
                return places, "gpu"
            except InferenceUnavailable as error:
                self.breaker.record_failure()
                logger.warning("GPU 경로 실패, CPU로 폴백: %s", error)
        if self.cpu is not None:
            return self.cpu.generate(header, slot_candidates, width), "cpu"
        raise InferenceUnavailable("사용 가능한 추론 경로가 없음")
