from fastapi import FastAPI

from app.photomissions.router import router as photomissions_router
from app.trips.router import router as trips_router

app = FastAPI(title="KTB4-20th-AI")
app.include_router(photomissions_router)
app.include_router(trips_router)

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
