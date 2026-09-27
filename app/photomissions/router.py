from fastapi import APIRouter, Depends

from app.core.auth import verify_internal_token
from app.photomissions.pipeline_generate import generate_missions
from app.photomissions.pipeline_verify import verify_photo
from app.photomissions.schemas import (
    PhotoMissionGenerateRequest,
    PhotoMissionGenerateResponse,
    VerifyRequest,
    VerifyResponse,
)

router = APIRouter(
    prefix="/photomissions",
    tags=["photomissions"],
    dependencies=[Depends(verify_internal_token)],
)


@router.post("/generate", response_model=PhotoMissionGenerateResponse)
async def generate(request: PhotoMissionGenerateRequest) -> PhotoMissionGenerateResponse:
    return await generate_missions(request)


@router.post("/{mission_id}/verify", response_model=VerifyResponse)
async def verify(mission_id: str, request: VerifyRequest) -> VerifyResponse:
    # mission_id는 BE가 어떤 미션인지 식별하려고 URL에 넣는 값 - 판정 로직엔 안 쓰임
    return await verify_photo(request)
