"""설정에서 추론 경로를 만든다. 처음 쓸 때 한 번 만들고 재사용한다."""

import logging
from functools import lru_cache

from app.core.config import settings
from app.trips.services.model.inference import (
    InferenceRouter,
    RunPodEngine,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_router() -> InferenceRouter:
    if not settings.runpod_model_url:
        raise RuntimeError("runpod_model_url이 없어 모델 추론 경로를 만들 수 없음")
    gpu = RunPodEngine(settings.runpod_model_url, settings.runpod_model_api_key, settings.runpod_model_timeout_seconds)
    logger.info("모델 추론 경로: gpu")
    return InferenceRouter(gpu)
