import io
import logging
import math
from typing import Literal, NamedTuple, cast
from urllib.parse import urlparse

import httpx
import sentry_sdk
from fastapi import HTTPException, status
from google.genai import errors, types
from PIL import Image, ImageOps

from app.core.config import settings
from app.photomissions import vlm_client
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

# 백엔드 업로드 상한(5MB, 단위 미확인)을 참고해 여유를 둔 AI 쪽 다운로드 상한. 업로드 검사와 따로 비정상적으로 큰 파일을 막는다
MAX_IMAGE_BYTES = 10 * 1024 * 1024
# Pillow가 DecompressionBombError를 내는 기준(약 1.79억 픽셀)과 같다. 지금 동작은 그대로 두고 500만 422로 바꾼다.
# 그 아래(약 9천만~1.79억)는 Pillow가 경고만 내고 원본 크기로 연다. 메모리 상한은 서버 메모리를 확인한 뒤 따로 정한다
MAX_IMAGE_PIXELS = 2 * Image.MAX_IMAGE_PIXELS
# 사진이 깨졌거나 형식이 맞지 않을 때 Pillow가 내는 오류. ValueError는 지나치게 큰 PNG 텍스트 청크에서 확인했고,
# SyntaxError는 일부 형식 처리기가 깨진 파일에 내는 오류다(재현 사례는 없음)
_IMAGE_ERRORS = (OSError, SyntaxError, ValueError)

logger = logging.getLogger(__name__)

# 2단계 평가셋(경주 4곳 400건, 사람이 매긴 정답 200장)으로 gemini-3.1-pro-preview 기준 확정.
# 점수 분포는 모델마다 달라서 모델을 바꾸면 다시 측정해야 한다. 자체 모델은 SELF_* 설정값을 쓴다
SUCCESS_THRESHOLD = 90.0  # 정답 성공 사진은 66장 중 65장이 90점 이상, 70~89점은 정답 재시도·실패뿐이었다
RETRY_THRESHOLD = 40.0
LANDMARK_THRESHOLD = 60.0  # 정답 성공 사진의 인식 신뢰도는 전부 95 이상, 정답 실패 사진은 중앙값 10


class Thresholds(NamedTuple):
    success: float
    retry: float
    landmark: float


GEMINI_THRESHOLDS = Thresholds(SUCCESS_THRESHOLD, RETRY_THRESHOLD, LANDMARK_THRESHOLD)


def current_thresholds() -> Thresholds:
    # 판정 백엔드에 맞는 기준. self 값이 있고 범위가 맞는지는 앱 시작 때 vlm_client.check_settings가 확인한다
    if settings.verify_backend == "self":
        return Thresholds(
            cast(float, settings.self_success_threshold),
            cast(float, settings.self_retry_threshold),
            cast(float, settings.self_landmark_threshold),
        )
    return GEMINI_THRESHOLDS


def assert_allowed_source(image_url: str) -> None:
    # 1. 입력 검증 - 허용된 호스트가 아니면 거부한다 (SSRF 방지)
    host = urlparse(image_url).hostname
    if host not in settings.allowed_image_hosts_list:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="허용되지 않은 이미지 호스트입니다",
        )


def _unprocessable(detail: str, **sizes: int) -> HTTPException:
    # 사진 자체의 문제라 422다. 사진 주소·본문·원본 오류 문장은 남기지 않는다(presigned URL)
    logger.info("verify 사진 거부: %s %s", detail, sizes)
    return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=detail)


def _too_large(**sizes: int) -> HTTPException:
    return _unprocessable("이미지 파일 크기가 제한을 초과했습니다", **sizes)


def _declared_length(response: httpx.Response) -> int | None:
    # 숫자가 아니거나 지나치게 긴 Content-Length는 무시하고, 실제로 받은 양으로 막는다
    value = response.headers.get("Content-Length", "")
    if not (value.isascii() and value.isdigit()):
        return None
    try:
        return int(value)
    except ValueError:  # 파이썬 정수 변환 길이 제한(4300자리)을 넘는 값
        return None


async def fetch_image(image_url: str) -> bytes:
    # 3. 이미지 획득 - presigned URL로 원본 사진을 가져온다. MAX_IMAGE_BYTES를 넘으면 끝까지 받지 않고 422
    # 압축 전송은 받지 않는다. 압축을 푸는 도중의 메모리 증가는 받은 양을 세는 것으로 막기 어렵다(S3 사진에는 압축이 필요 없다)
    # TODO: 타임아웃 값(httpx 기본 5초)은 2단계 baseline 측정 후 확정
    async with httpx.AsyncClient() as client:
        try:
            async with client.stream("GET", image_url, headers={"Accept-Encoding": "identity"}) as response:
                response.raise_for_status()  # 큰 404·500 본문도 읽지 않고 502로
                if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                    raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="이미지 다운로드 실패")
                declared = _declared_length(response)
                if declared is not None and declared > MAX_IMAGE_BYTES:
                    raise _too_large(declared_bytes=declared)
                # 헤더가 없거나 틀려도 실제로 받은 양으로 다시 막는다. 압축은 위에서 거절해서 받은 바이트 그대로다
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body += chunk
                    if len(body) > MAX_IMAGE_BYTES:
                        raise _too_large(received_bytes=len(body))
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
    return bytes(body)


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


def prepare_image(image_bytes: bytes) -> bytes:
    # 4. 이미지 전처리(서비스) - 열 수 없거나 너무 큰 사진을 500 대신 422로 돌려준다.
    # normalize_image는 측정·학습 스크립트도 같이 쓰므로 웹 오류를 넣지 않고, 결과도 그대로 둔다
    try:
        with Image.open(io.BytesIO(image_bytes)) as probe:  # 헤더만 읽는다. 픽셀은 아직 풀지 않는다
            width, height = probe.size
    except Image.DecompressionBombError as e:
        raise _unprocessable("이미지 해상도가 제한을 초과했습니다", file_bytes=len(image_bytes)) from e
    except _IMAGE_ERRORS as e:
        raise _unprocessable("이미지를 열 수 없습니다", file_bytes=len(image_bytes)) from e
    if width * height > MAX_IMAGE_PIXELS:
        raise _unprocessable("이미지 해상도가 제한을 초과했습니다", width=width, height=height)
    try:
        return normalize_image(image_bytes)
    except _IMAGE_ERRORS as e:  # 헤더는 멀쩡하고 본문이 깨진 사진(중간에 잘린 JPEG 등)
        raise _unprocessable("이미지를 열 수 없습니다", file_bytes=len(image_bytes)) from e


async def score_photo(
    image_bytes: bytes, place_name: str, mission_description: str
) -> VlmResult:
    # 5. 사진 판정 - 이미지+장소명+미션 내용을 VLM한테 주고 관찰→인식→판단→조언을 한 번에 받음
    # VERIFY_BACKEND가 판정 모델을 정한다. 자체 모델의 시간 제한·에러 처리는 vlm_client가 맡는다
    if settings.verify_backend == "self":
        return await vlm_client.score(image_bytes, place_name, mission_description)

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
    thresholds = current_thresholds()
    if match_score >= thresholds.success:
        return "success"
    if match_score >= thresholds.retry:
        return "retry"
    return "fail"


def location_mismatch_response() -> VerifyResponse:
    # 2번에서 끝난 사전 판정이라 모델 점수·기준을 거치지 않고 항상 실패다
    # 0점을 등급 기준에 넣으면 self 기준이 0일 때 재시도나 성공이 된다(SELF_*_THRESHOLD는 0을 허용한다)
    return VerifyResponse(
        result="fail",
        reason="location_mismatch",
        match_score=0,
        detected_labels=[],
        landmark_confidence=None,
        retry_hint=None,
    )


def build_response(
    match_score: float,
    detected_labels: list[DetectedLabel] | None = None,
    raw_landmark_confidence: float | None = None,
    raw_retry_hint: str | None = None,
) -> VerifyResponse:
    # 7. 응답 조립 - landmark_confidence·retry_hint의 노출 여부를 임계값·등급으로 정한다. 문구는 안 붙임
    # 등급은 점수로만 정해지므로 받지 않고 여기서 6번을 호출한다. 점수와 어긋난 등급이 응답에 들어갈 수 없다
    result = to_grade(match_score)
    landmark_confidence = (
        raw_landmark_confidence
        if raw_landmark_confidence is not None and raw_landmark_confidence >= current_thresholds().landmark
        else None
    )
    retry_hint = raw_retry_hint if result == "retry" else None
    return VerifyResponse(
        result=result,
        reason=None,
        match_score=match_score,
        detected_labels=detected_labels or [],
        landmark_confidence=landmark_confidence,
        retry_hint=retry_hint,
    )


async def verify_photo(request: VerifyRequest) -> VerifyResponse:
    # 오케스트레이터 - 판단하지 않고 1~7단계를 순서대로 호출만 한다
    # 모델을 부르기 전에 끝나는 요청(위치 반려·사진 다운로드 실패)도 Sentry에서 백엔드별로 볼 수 있게 맨 앞에서 붙인다
    sentry_sdk.set_tag("verify_backend", settings.verify_backend)
    assert_allowed_source(request.image_url)  # 1
    # 좌표가 없으면 위치로는 판정할 수 없으므로 사진 판정(VLM)으로 넘긴다
    if request.photo_coordinates is not None and is_clear_mismatch(  # 2
        request.photo_coordinates, request.place_coordinates
    ):
        return location_mismatch_response()

    image_bytes = await fetch_image(request.image_url)  # 3
    processed = prepare_image(image_bytes)  # 4
    vlm = await score_photo(processed, request.place_name, request.mission_description)  # 5

    # VLM 원값을 그대로 7번에 넘긴다. 등급(6번)과 노출 여부는 build_response가 정한다.
    return build_response(  # 6·7
        vlm.match_score,
        vlm.detected_labels,
        raw_landmark_confidence=vlm.landmark_confidence,
        raw_retry_hint=vlm.retry_hint,
    )
