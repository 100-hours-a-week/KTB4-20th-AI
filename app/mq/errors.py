"""AI 처리 실패를 계약 문서 4.6절 errorKind로 나눈다."""

import asyncio

import httpx
from fastapi import HTTPException, status
from pydantic import ValidationError

from app.mq.topology import JobType
from app.trips.fifo_gate import QueueFull, QueueWaitTimeout


class AiJobError(Exception):
    """kind는 계약 문서 4.6절 errorKind, retryable이면 Retry Queue로 다시 보낸다."""

    def __init__(self, kind: str, retryable: bool, message: str):
        super().__init__(f"{kind}: {message}")
        self.kind = kind
        self.retryable = retryable
        self.message = message


AI_TIMEOUT = "AI_TIMEOUT"
AI_UNAVAILABLE = "AI_UNAVAILABLE"
INVALID_MESSAGE = "INVALID_MESSAGE"
UNSUPPORTED_SCHEMA_VERSION = "UNSUPPORTED_SCHEMA_VERSION"
INVALID_REQUEST = "INVALID_REQUEST"
IMAGE_NOT_FOUND = "IMAGE_NOT_FOUND"
UNSUPPORTED_IMAGE = "UNSUPPORTED_IMAGE"

# 사진 주소가 없어졌거나(404) presigned URL이 만료된(403) 경우. 다시 받아도 결과가 같다
_IMAGE_GONE_STATUSES = {status.HTTP_403_FORBIDDEN, status.HTTP_404_NOT_FOUND}


def classify(error: BaseException, job_type: JobType) -> AiJobError:
    """다시 시도해서 풀릴 수 있는 오류(시간 초과, 외부 서비스 장애)만 재시도 가능으로 본다."""
    if isinstance(error, AiJobError):
        return error
    if isinstance(error, (asyncio.TimeoutError, TimeoutError)):
        return AiJobError(AI_TIMEOUT, True, "AI 처리 시간 초과")
    if isinstance(error, ValidationError):
        return AiJobError(INVALID_REQUEST, False, _first_error(error))
    if isinstance(error, (QueueFull, QueueWaitTimeout)):
        # HTTP 요청과 함께 쓰는 place-selection 줄이 가득 찼거나 오래 기다렸다. 잠시 뒤 다시 하면 풀린다
        return AiJobError(AI_UNAVAILABLE, True, "place-selection 줄이 비지 않음")
    if isinstance(error, HTTPException):
        return _classify_http(error, job_type)
    return AiJobError(AI_UNAVAILABLE, True, f"{type(error).__name__}: {error}")


def _classify_http(error: HTTPException, job_type: JobType) -> AiJobError:
    detail = str(error.detail)
    if job_type is JobType.PHOTO_JUDGE:
        cause = error.__cause__
        if isinstance(cause, httpx.HTTPStatusError) and cause.response.status_code in _IMAGE_GONE_STATUSES:
            return AiJobError(IMAGE_NOT_FOUND, False, detail)
        if error.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT:
            # 허용하지 않은 이미지 호스트는 요청 값 문제, 그 밖의 422는 사진 크기·형식 문제다
            kind = INVALID_REQUEST if "호스트" in detail else UNSUPPORTED_IMAGE
            return AiJobError(kind, False, detail)
    if error.status_code == status.HTTP_504_GATEWAY_TIMEOUT:
        return AiJobError(AI_TIMEOUT, True, detail)
    if error.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT:
        return AiJobError(INVALID_REQUEST, False, detail)
    # 429(호출 한도), 502·503(외부 모델 실패), 500
    return AiJobError(AI_UNAVAILABLE, True, detail)


def _first_error(error: ValidationError) -> str:
    first = error.errors()[0]
    location = ".".join(str(part) for part in first.get("loc", ()))
    return f"{location}: {first.get('msg', '')}"
