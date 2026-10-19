import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import settings
from app.core.sentry import init_sentry
from app.photomissions import pipeline_generate, vlm_client
from app.photomissions.router import router as photomissions_router
from app.trips.router import router as trips_router

init_sentry()  # FastAPI 앱을 만들기 전에 초기화해야 요청 에러가 수집된다


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    try:
        pipeline_generate.start()  # 설정이 틀리면 여기서 멈춘다
        # 자체 모델로 판정할 때만 설정을 확인한다. 틀리면 요청을 받기 전에 시작이 멈춘다
        if settings.verify_backend == "self":
            vlm_client.check_settings(settings)
            vlm_client.start(settings)  # 검사를 통과한 뒤에만 클라이언트를 만든다
        # uvicorn 로그에 같이 찍혀서 켜자마자 지금 어느 모델로 판정하는지 보인다
        logger = logging.getLogger("uvicorn.error")
        logger.info(vlm_client.backend_summary(settings))
        if warning := vlm_client.deadline_warning(settings):
            logger.warning(warning)
        yield
    finally:
        # 시작 중 실패하거나 종료 정리가 실패해도 generate 상태는 비운다
        try:
            await vlm_client.close()  # 시작하지 않았으면 아무것도 하지 않는다
        finally:
            pipeline_generate.close()


app = FastAPI(title="KTB4-20th-AI", lifespan=lifespan)
app.include_router(photomissions_router)
app.include_router(trips_router)

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
