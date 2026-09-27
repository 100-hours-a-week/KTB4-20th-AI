import io
import math
from typing import Literal
from urllib.parse import urlparse

import httpx
from fastapi import HTTPException, status
from google.genai import errors, types
from PIL import Image, ImageOps

from app.core.config import settings
from app.photomissions.gemini_client import generate_structured
from app.photomissions.prompts import VERIFY_SYSTEM_PROMPT
from app.photomissions.schemas import (
    Coordinates,
    DetectedLabel,
    VerifyRequest,
    VerifyResponse,
    VlmResult,
)

_EARTH_RADIUS_KM = 6371.0
# 평가셋에서 맞는 사진은 목표에서 최대 1.6km라 5km에서 잘못 반려된 사진이 없었다. 줄이면 넓은 장소에서 오반려 위험이 있다
CLEAR_MISMATCH_KM = 5.0

_MAX_DIMENSION = 1024  # TODO: baseline 측정(2단계) 후 토큰·정확도 트레이드오프로 조정
_JPEG_QUALITY = 85

# 2단계 평가셋(경주 4곳 400건, 사람이 매긴 정답 200장)으로 gemini-3.1-pro-preview 기준 확정.
# 점수 분포는 모델마다 달라서 모델을 바꾸면 다시 측정해야 한다
SUCCESS_THRESHOLD = 90.0  # 정답 성공 사진은 66장 중 65장이 90점 이상, 70~89점은 정답 재시도·실패뿐이었다
RETRY_THRESHOLD = 40.0
LANDMARK_THRESHOLD = 60.0  # 정답 성공 사진의 인식 신뢰도는 전부 95 이상, 정답 실패 사진은 중앙값 10


def assert_allowed_source(image_url: str) -> None:
    # 1. 입력 검증 - 허용된 호스트가 아니면 거부한다 (SSRF 방지)
    host = urlparse(image_url).hostname
    if host not in settings.allowed_image_hosts_list:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="허용되지 않은 이미지 호스트입니다",
        )


async def fetch_image(image_url: str) -> bytes:
    # 3. 이미지 획득 - presigned URL로 원본 사진을 가져온다
    # TODO: 타임아웃 값(httpx 기본 5초)은 2단계 baseline 측정 후 확정
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(image_url)
            response.raise_for_status()
        except httpx.TimeoutException as e:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="이미지 다운로드 시간 초과",
            ) from e
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="이미지 다운로드 실패",
            ) from e
    return response.content


def _haversine_km(a: Coordinates, b: Coordinates) -> float:
    # 두 좌표 사이의 직선거리(km). 지구를 구로 근사하는 Haversine 공식.
    lat1, lon1 = math.radians(a.latitude), math.radians(a.longitude)
    lat2, lon2 = math.radians(b.latitude), math.radians(b.longitude)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(h))


def is_clear_mismatch(photo_coordinates: Coordinates, place_coordinates: Coordinates) -> bool:
    # 2. 위치 사전 판정 - 사진 좌표가 목표 장소에서 명백히 먼지만 본다
    return _haversine_km(photo_coordinates, place_coordinates) > CLEAR_MISMATCH_KM


def normalize_image(image_bytes: bytes) -> bytes:
    # 4. 이미지 전처리 - 리사이즈·포맷 정규화로 입력 크기 축소 (prefill 병목 완화)
    image = Image.open(io.BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image)  # EXIF 방향을 실제 픽셀에 반영 (세로사진이 눕는 것 방지)
    image = image.convert("RGB")  # PNG 투명배경 등도 JPEG로 통일

    image.thumbnail((_MAX_DIMENSION, _MAX_DIMENSION), Image.Resampling.LANCZOS)  # 비율 유지, 더 작으면 확대 안 함

    output = io.BytesIO()
    image.save(output, format="JPEG", quality=_JPEG_QUALITY)
    return output.getvalue()


async def score_photo(
    image_bytes: bytes, place_name: str, mission_description: str
) -> VlmResult:
    # 5. 사진 판정 - 이미지+장소명+미션 내용을 VLM한테 주고 관찰→인식→판단→조언을 한 번에 받음
    # TODO: Retry-After 값(30초)은 baseline 측정 후 확정
    user_prompt = f"목표 장소: {place_name}\n미션 내용: {mission_description}"
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
    try:
        return await generate_structured(
            system=VERIFY_SYSTEM_PROMPT,
            user=[user_prompt, image_part],
            response_schema=VlmResult,
        )
    except errors.APIError as e:
        if e.code == status.HTTP_429_TOO_MANY_REQUESTS:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Gemini 호출 한도 초과",
                headers={"Retry-After": "30"},
            ) from e
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="사진 판정 실패",
        ) from e
    except httpx.TimeoutException as e:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="사진 판정 시간 초과",
        ) from e
    except ValueError as e:  # 재시도 후에도 응답이 스키마와 맞지 않음
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="사진 판정 실패",
        ) from e


def to_grade(match_score: float) -> Literal["success", "retry", "fail"]:
    # 6. 등급 변환 - match_score 구간으로 success/retry/fail 판정
    if match_score >= SUCCESS_THRESHOLD:
        return "success"
    if match_score >= RETRY_THRESHOLD:
        return "retry"
    return "fail"


def build_response(
    match_score: float,
    detected_labels: list[DetectedLabel] | None = None,
    raw_landmark_confidence: float | None = None,
    raw_retry_hint: str | None = None,
    reason: Literal["location_mismatch"] | None = None,
) -> VerifyResponse:
    # 7. 응답 조립 - landmark_confidence·retry_hint의 노출 여부를 임계값·등급으로 정한다. 문구는 안 붙임
    # 등급은 점수로만 정해지므로 받지 않고 여기서 6번을 호출한다. 점수와 어긋난 등급이 응답에 들어갈 수 없다
    result = to_grade(match_score)
    landmark_confidence = (
        raw_landmark_confidence
        if raw_landmark_confidence is not None and raw_landmark_confidence >= LANDMARK_THRESHOLD
        else None
    )
    retry_hint = raw_retry_hint if result == "retry" else None
    return VerifyResponse(
        result=result,
        reason=reason,
        match_score=match_score,
        detected_labels=detected_labels or [],
        landmark_confidence=landmark_confidence,
        retry_hint=retry_hint,
    )


async def verify_photo(request: VerifyRequest) -> VerifyResponse:
    # 오케스트레이터 - 판단하지 않고 1~7단계를 순서대로 호출만 한다
    assert_allowed_source(request.image_url)  # 1
    # 좌표가 없으면 위치로는 판정할 수 없으므로 사진 판정(VLM)으로 넘긴다
    if request.photo_coordinates is not None and is_clear_mismatch(  # 2
        request.photo_coordinates, request.place_coordinates
    ):
        return build_response(match_score=0, reason="location_mismatch")  # 6번에서 fail이 된다

    image_bytes = await fetch_image(request.image_url)  # 3
    processed = normalize_image(image_bytes)  # 4
    vlm = await score_photo(processed, request.place_name, request.mission_description)  # 5

    # VLM 원값을 그대로 7번에 넘긴다. 등급(6번)과 노출 여부는 build_response가 정한다.
    return build_response(  # 6·7
        vlm.match_score,
        vlm.detected_labels,
        raw_landmark_confidence=vlm.landmark_confidence,
        raw_retry_hint=vlm.retry_hint,
    )
