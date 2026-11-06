"""모델 추론 경로: RunPod GPU와 오프라인 도구용 CPU 엔진 (ADR-0004).

  - GPU 경로: RunPod Serverless 엔드포인트(모델 전용)에 슬롯별 후보 목록을 보내 일정 전체를 받는다.
  - CPU 엔진: ONNX Runtime으로 로컬에서 추론한다. 서비스는 쓰지 않고 오프라인 검증·측정 스크립트(scripts/)만 쓴다.
  - 라우터: GPU가 있으면 GPU만 쓰고, 실패해도 CPU로 넘기지 않는다(CPU 폴백 없음).
"""

import json
import logging
import threading
from collections.abc import Callable
from pathlib import Path

from app.trips.services.model.generation import (
    BEAM_WIDTH,
    NextLogprobs,
    generate_itinerary,
    generate_trip,
    resolve_days,
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

    def generate(
        self, header: list[int], slot_candidates: list[list[str]], width: int = BEAM_WIDTH, first_k: int = 1, seed: int | None = None,
    ) -> list[str | None]:
        if not self._slots.acquire(blocking=False):
            raise InferenceUnavailable("CPU 동시 실행 상한 초과")
        try:
            return generate_itinerary(header, slot_candidates, self.codes, self.next_logprobs(), width, first_k, seed)
        finally:
            self._slots.release()

    def generate_trip(
        self, header: list[int], pools: dict[str, list[str]], days: list[list[str]], width: int = BEAM_WIDTH,
        first_k: int = 1, seed: int | None = None,
    ) -> list[list[str | None]]:
        if not self._slots.acquire(blocking=False):
            raise InferenceUnavailable("CPU 동시 실행 상한 초과")
        try:
            return generate_trip(header, resolve_days(pools, days), self.codes, self.next_logprobs(), width, first_k, seed)
        finally:
            self._slots.release()


class RunPodEngine:
    """RunPod Serverless 동기 호출(/runsync). 후보 목록과 헤더를 보내고 슬롯별 장소를 받는다."""

    def __init__(self, url: str, api_key: str, timeout_seconds: float, post: Callable | None = None):
        self.url, self.api_key, self.timeout = url.rstrip("/"), api_key, timeout_seconds
        self._post = post

    def generate(
        self, header: list[int], slot_candidates: list[list[str]], width: int = BEAM_WIDTH, first_k: int = 1, seed: int | None = None,
    ) -> list[str | None]:
        payload = {"input": {"header": header, "slots": slot_candidates, "width": width, "first_k": first_k, "seed": seed}}
        places = self._run(payload).get("places")
        if not isinstance(places, list) or len(places) != len(slot_candidates):
            raise InferenceUnavailable("RunPod 응답 형식 오류")
        return places

    def generate_trip(
        self, header: list[int], pools: dict[str, list[str]], days: list[list[str]], width: int = BEAM_WIDTH,
        first_k: int = 1, seed: int | None = None,
    ) -> list[list[str | None]]:
        # 후보 풀은 한 번만 보내고 슬롯은 풀 이름으로 가리킨다(요청 크기가 일수에 비례하지 않게, ADR-0008)
        payload = {"input": {"header": header, "pools": pools, "days": days, "width": width, "first_k": first_k, "seed": seed}}
        trip = self._run(payload).get("days")
        if not isinstance(trip, list) or len(trip) != len(days) or any(
            not isinstance(day, list) or len(day) != len(slots) for day, slots in zip(trip, days, strict=True)
        ):
            raise InferenceUnavailable("RunPod 응답 형식 오류")
        return trip

    def _run(self, payload: dict) -> dict:
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
        return body.get("output") or {}


class InferenceRouter:
    """GPU가 있으면 GPU만 쓴다(실패해도 CPU로 폴백하지 않음). GPU 없이 CPU 엔진만 넘기면 CPU로 추론한다(오프라인 스크립트용).
    어느 경로가 썼는지 route를 함께 돌려준다. 부하 테스트 baseline: 서킷 브레이커 없이 매번 GPU를 부른다."""

    def __init__(self, gpu: RunPodEngine | None, cpu: LocalEngine | None = None):
        self.gpu, self.cpu = gpu, cpu

    def generate(
        self, header: list[int], slot_candidates: list[list[str]], width: int = BEAM_WIDTH, first_k: int = 1, seed: int | None = None,
    ) -> tuple[list[str | None], str]:
        return self._route(lambda engine: engine.generate(header, slot_candidates, width, first_k, seed))

    def generate_trip(
        self, header: list[int], pools: dict[str, list[str]], days: list[list[str]], width: int = BEAM_WIDTH,
        first_k: int = 1, seed: int | None = None,
    ) -> tuple[list[list[str | None]], str]:
        return self._route(lambda engine: engine.generate_trip(header, pools, days, width, first_k, seed))

    def _route(self, run: Callable):
        if self.gpu is not None:
            try:
                return run(self.gpu), "gpu"
            except InferenceUnavailable as error:
                logger.warning("GPU 경로 실패: %s", error)
                raise InferenceUnavailable("사용 가능한 추론 경로가 없음") from error
        if self.cpu is not None:  # GPU 없이 CPU 엔진만 넘긴 경우(오프라인 스크립트)
            return run(self.cpu), "cpu"
        raise InferenceUnavailable("사용 가능한 추론 경로가 없음")
