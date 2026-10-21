"""설정에서 추론 경로를 만든다. 처음 쓸 때 한 번 만들고 재사용한다."""

import logging
from functools import lru_cache
from pathlib import Path

from app.core.config import settings
from app.trips.services.model.inference import (
    CircuitBreaker,
    InferenceRouter,
    LocalEngine,
    RunPodEngine,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_router() -> InferenceRouter:
    gpu = (
        RunPodEngine(settings.runpod_model_url, settings.runpod_model_api_key, settings.runpod_model_timeout_seconds)
        if settings.runpod_model_url else None
    )
    cpu = None
    if settings.model_dir:
        cpu = LocalEngine(Path(settings.model_dir), settings.model_cpu_max_concurrency)
    if gpu is None and cpu is None:
        raise RuntimeError("model_dir도 runpod_model_url도 없어 모델 추론 경로를 만들 수 없음")
    breaker = CircuitBreaker(settings.model_breaker_failures, settings.model_breaker_cooldown_seconds)
    logger.info("모델 추론 경로: gpu=%s cpu=%s", gpu is not None, cpu is not None)
    return InferenceRouter(gpu, cpu, breaker)
