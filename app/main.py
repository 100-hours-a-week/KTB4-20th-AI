from fastapi import FastAPI

from app.photomissions.router import router as photomissions_router

app = FastAPI(title="KTB4-20th-AI")
app.include_router(photomissions_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
