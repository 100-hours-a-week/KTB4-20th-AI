from fastapi import FastAPI

app = FastAPI(title="KTB4-20th-AI")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
