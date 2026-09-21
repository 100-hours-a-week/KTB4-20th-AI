from fastapi import APIRouter, Depends

from app.core.auth import verify_internal_token
from app.photomissions.pipeline_generate import generate_missions
from app.photomissions.schemas import (
    PhotoMissionGenerateRequest,
    PhotoMissionGenerateResponse,
)

router = APIRouter(
    prefix="/photomissions",
    tags=["photomissions"],
    dependencies=[Depends(verify_internal_token)],
)


@router.post("/generate", response_model=PhotoMissionGenerateResponse)
async def generate(request: PhotoMissionGenerateRequest) -> PhotoMissionGenerateResponse:
    return await generate_missions(request)
