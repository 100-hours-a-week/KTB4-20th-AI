from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.core.sentry import init_sentry
from app.photomissions.router import router as photomissions_router
from app.trips.router import router as trips_router

init_sentry()

app = FastAPI(title="KTB4-20th-AI")
app.include_router(photomissions_router)
app.include_router(trips_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


Instrumentator(
    should_group_status_codes=False,
    excluded_handlers=[r"^/health$", r"^/metrics$"],
).instrument(app).expose(
    app,
    endpoint="/metrics",
    include_in_schema=False,
)