from fastapi import Header, HTTPException, status

from app.core.config import settings


async def verify_internal_token(x_internal_token: str | None = Header(default=None)) -> None:
    # BE-AI 내부 통신 검증 -> 값이 설정 안 됐거나(빈 문자열) 안 맞으면 거부
    if not settings.internal_service_token or x_internal_token != settings.internal_service_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid internal token",
        )
