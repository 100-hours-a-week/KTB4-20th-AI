import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import settings
from app.core.sentry import init_sentry
from app.photomissions import vlm_client
from app.photomissions.router import router as photomissions_router
from app.trips.router import router as trips_router

init_sentry()  # FastAPI 앱을 만들기 전에 초기화해야 요청 에러가 수집된다


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # 자체 모델로 판정할 때만 설정을 확인한다. 틀리면 요청을 받기 전에 시작이 멈춘다
    if settings.verify_backend == "self":
        vlm_client.check_settings(settings)
    # uvicorn 로그에 같이 찍혀서 켜자마자 지금 어느 모델로 판정하는지 보인다
    logging.getLogger("uvicorn.error").info(vlm_client.backend_summary(settings))
    yield


app = FastAPI(title="KTB4-20th-AI", lifespan=lifespan)
app.include_router(photomissions_router)
app.include_router(trips_router)

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
