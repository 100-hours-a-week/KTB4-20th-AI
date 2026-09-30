from fastapi import FastAPI

from app.core.sentry import init_sentry
from app.photomissions.router import router as photomissions_router
from app.trips.router import router as trips_router

init_sentry()  # FastAPI 앱을 만들기 전에 초기화해야 요청 에러가 수집된다

app = FastAPI(title="KTB4-20th-AI")
app.include_router(photomissions_router)
app.include_router(trips_router)

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
