import asyncio
import io
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, NamedTuple, cast
from urllib.parse import urlparse

import httpx
import sentry_sdk
from fastapi import HTTPException, status
from google.genai import errors, types
from PIL import Image, ImageOps

from app.core.config import settings
from app.core.time import SERVICE_TIMEZONE
from app.photomissions import vlm_client
from app.photomissions.gemini_client import generate_structured
from app.photomissions.prompts import VERIFY_SYSTEM_PROMPT
from app.photomissions.question_format import Grade, QuestionAnswer, rule_grade
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
# 요청별 판정 기록 한 줄. uvicorn 기본 설정은 uvicorn.* 로거만 출력해서 이 모듈 로거의 INFO는 서버 로그에 안 나온다
# 시작 로그(main.py)처럼 uvicorn.error 로거로 남긴다
verify_log = logging.getLogger("uvicorn.error")

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

type AnsweredBy = Literal["self", "gemini"]


class ScoreResult(NamedTuple):
    # 판정 결과와 그 판정을 실제로 낸 모델. 점수 분포가 모델마다 달라 등급 기준을 고를 때 쓴다
    # vlm은 점수 형식(VlmResult)이거나 자체 모델 객관식(QuestionAnswer)이다. 등급을 정하는 길은 이 타입으로 고른다
    vlm: VlmResult | QuestionAnswer
    answered_by: AnsweredBy


# 객관식은 점수가 없어서 응답의 match_score(0~100 필수)에 등급 대표값을 넣는다. 모델 확신도가 아니다
# BE 확인(10/18): 완료 처리는 result로만 하고, match_score는 0~100 검증 후 저장한다.
# 화면 일치도 구간(90 이상 성공 / 40~89 재시도 / 40 미만 실패)과 이 대표값이 맞는다.
# 저장된 값으로 통계를 낼 때는 Gemini 실제 점수와 섞이지 않게 판정 모델별로 구분해야 한다
QUESTION_GRADE_SCORE: dict[Grade, float] = {"success": 100.0, "retry": 50.0, "fail": 0.0}

# 객관식 재시도 안내. 어떤 조건을 어겼는지는 답에 없어서 구체적인 이유를 추측하지 않는다
# 여러 원인이 겹치면 위에서부터 하나만 쓴다(장소 → 대상 → 조건). 정확도로 정한 순서가 아니라 안내 정책이다
QUESTION_RETRY_HINTS: list[tuple[str, str, str]] = [
    ("place", "unknown", "목표 장소의 특징이 보이게 다시 찍어주세요"),
    ("target", "short", "미션 대상이 더 잘 보이게 다시 찍어주세요"),
    ("condition", "no", "미션 조건이 사진에 보이도록 다시 찍어주세요"),
]


# 자체 모델 호출이 이 상태로 끝나면 Gemini로 다시 판정한다(폴백 설정 시). 정상 판정(retry·fail 포함)은 넘기지 않는다
# 429도 넘기므로 자체 모델 호출 제한이 Gemini 트래픽·비용으로 바뀐다
FALLBACK_STATUSES = frozenset({429, 502, 503, 504})

type Route = Literal["self", "off_hours", "gemini_backend", "not_started"]


@dataclass
class VerifyTrace:
    # 요청 한 건의 모델 판정 경로 기록. 요청마다 새로 만든다. 사진·키·주소·에러 원문은 넣지 않는다
    route: Route = "not_started"  # self: 자체 모델 먼저 / off_hours: 운영 시간 밖이라 바로 Gemini / gemini_backend: 설정이 gemini
    answered_by: AnsweredBy | None = None  # 모델이 정상적으로 답했을 때만(성공 판정이라는 뜻이 아니다)
    fallback_enabled: bool = False
    fallback_attempted: bool = False
    fallback_succeeded: bool = False
    fallback_skip_reason: str | None = None
    self_error_status: int | None = None  # 이 서버가 매핑한 상태(504 등). 실제 상대 서버 상태는 self_call["upstream_status"]
    self_call: dict = field(default_factory=dict)  # vlm_client.score가 채우는 기록(시도 횟수·상대 상태·원인·걸린 시간)
    budget_sec: float | None = None
    self_sec: float | None = None
    gemini_sec: float | None = None
    model_stage_sec: float | None = None


def _now_in_service_timezone() -> datetime:
    # 운영 시간 판단용 시계. 테스트에서 바꿔 끼운다(마감은 이벤트 루프의 단조 시계로 따로 잰다)
    return datetime.now(SERVICE_TIMEZONE)


def thresholds_for(answered_by: AnsweredBy) -> Thresholds:
    # 설정(VERIFY_BACKEND)이 아니라 답한 모델의 기준. self 값이 있고 범위가 맞는지는 앱 시작 때 vlm_client.check_settings가 확인한다
    if answered_by == "self":
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
    image_bytes: bytes, place_name: str, mission_description: str, trace: VerifyTrace | None = None,
) -> ScoreResult:
    # 5. 사진 판정 - 이미지+장소명+미션 내용을 VLM한테 주고 관찰→인식→판단→조언을 한 번에 받음
    # VERIFY_BACKEND가 판정 모델을 정한다. 자체 모델의 시간 제한·에러 처리는 vlm_client가 맡는다
    # self + VERIFY_MODEL_BUDGET_SECONDS면 자체 모델이 실패하거나 운영 시간 밖일 때 Gemini로 판정한다(전체 예산 안에서)
    t = trace if trace is not None else VerifyTrace()
    loop = asyncio.get_running_loop()
    started = loop.time()
    args = (image_bytes, place_name, mission_description)
    try:
        if settings.verify_backend != "self":
            t.route = "gemini_backend"
            return await _gemini(t, *args)
        budget = settings.verify_model_budget_seconds
        if budget is None:  # 폴백 없음: 자체 모델 결과·에러를 그대로 돌려준다
            t.route = "self"
            return await _self(t, *args)
        t.fallback_enabled, t.budget_sec = True, budget
        return await _with_fallback(t, started + budget, *args)
    finally:
        t.model_stage_sec = round(loop.time() - started, 3)


async def _with_fallback(t: VerifyTrace, model_deadline: float, *args) -> ScoreResult:
    # 마감은 진입할 때 한 번 정하고, 운영 시간 판단도 진입할 때 한 번만 한다(도중에 종료 시각을 넘어도 그대로 끝낸다)
    # 자체 모델·Gemini 모두 이 마감 안에서만 돈다. 실제로 마감을 넘기면 504
    loop = asyncio.get_running_loop()
    try:
        async with asyncio.timeout_at(model_deadline + vlm_client.DEADLINE_GUARD):
            hours = vlm_client.parse_serving_hours(settings.vlm_serving_hours)
            if not vlm_client.in_serving_hours(_now_in_service_timezone(), hours):
                t.route = "off_hours"  # 폴백이 아니다. 폴백률에 섞지 않는다
                return await _gemini(t, *args)
            t.route = "self"
            # Gemini 몫과 여유를 남기고 자체 모델을 기다린다. 설정 마감(VLM_DEADLINE_SECONDS)이 더 이르면 그쪽이다
            self_deadline = model_deadline - vlm_client.MIN_GEMINI_START_BUDGET_SECONDS - vlm_client.BUDGET_MARGIN_SECONDS
            try:
                return await _self(t, *args, deadline=self_deadline)
            except HTTPException as e:
                if e.status_code not in FALLBACK_STATUSES:
                    raise
                t.self_error_status = e.status_code
                if model_deadline - loop.time() < vlm_client.MIN_GEMINI_START_BUDGET_SECONDS:
                    # 시간이 남았어도 Gemini를 끝낼 만큼은 아니다. 원인을 504로 바꾸지 않고 자체 모델 에러를 그대로 둔다
                    t.fallback_skip_reason = "insufficient_budget"
                    raise
                t.fallback_attempted = True
                result = await _gemini(t, *args)
                t.fallback_succeeded = True
                return result
    except TimeoutError as e:  # 모델 단계 전체 마감을 넘김. timeout 컨텍스트 바깥에서 바꾼다
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="사진 판정 시간 초과") from e


async def _self(t: VerifyTrace, *args, deadline: float | None = None) -> ScoreResult:
    loop = asyncio.get_running_loop()
    started = loop.time()
    try:
        result = ScoreResult(await vlm_client.score(*args, deadline=deadline, trace=t.self_call), "self")
    finally:
        t.self_sec = round(loop.time() - started, 3)
    t.answered_by = "self"
    return result


async def _gemini(t: VerifyTrace, *args) -> ScoreResult:
    loop = asyncio.get_running_loop()
    started = loop.time()
    try:
        result = ScoreResult(await _score_with_gemini(*args), "gemini")
    finally:
        t.gemini_sec = round(loop.time() - started, 3)
    t.answered_by = "gemini"
    return result


async def _score_with_gemini(image_bytes: bytes, place_name: str, mission_description: str) -> VlmResult:
    # Gemini 판정과 에러 → HTTP 상태 변환
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


def to_grade(match_score: float, thresholds: Thresholds) -> Literal["success", "retry", "fail"]:
    # 6. 등급 변환 - match_score 구간으로 success/retry/fail 판정. 기준은 답한 모델의 것을 받는다
    if match_score >= thresholds.success:
        return "success"
    if match_score >= thresholds.retry:
        return "retry"
    return "fail"


def question_response(answer: QuestionAnswer) -> VerifyResponse:
    # 6·7. 자체 모델 객관식 - 등급은 답에서 규칙으로 한 번만 정한다. 점수 기준(to_grade)을 다시 거치지 않는다
    result = rule_grade(answer)
    retry_hint = (
        next((hint for field, value, hint in QUESTION_RETRY_HINTS if getattr(answer, field) == value), None)
        if result == "retry"
        else None
    )
    return VerifyResponse(
        result=result,
        reason=None,  # 사진으로 본 "다른 장소"는 GPS 위치 반려(location_mismatch)와 다르다
        match_score=QUESTION_GRADE_SCORE[result],
        detected_labels=[],
        landmark_confidence=None,
        retry_hint=retry_hint,
    )


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
    thresholds: Thresholds,
    detected_labels: list[DetectedLabel] | None = None,
    raw_landmark_confidence: float | None = None,
    raw_retry_hint: str | None = None,
) -> VerifyResponse:
    # 7. 응답 조립 - landmark_confidence·retry_hint의 노출 여부를 임계값·등급으로 정한다. 문구는 안 붙임
    # 등급은 점수로만 정해지므로 받지 않고 여기서 6번을 호출한다. 점수와 어긋난 등급이 응답에 들어갈 수 없다
    result = to_grade(match_score, thresholds)
    landmark_confidence = (
        raw_landmark_confidence
        if raw_landmark_confidence is not None and raw_landmark_confidence >= thresholds.landmark
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
    trace = VerifyTrace()
    response: VerifyResponse | None = None
    final_status, outcome = 500, "error"
    try:
        response = await _verify(request, trace)
        final_status, outcome = 200, "ok"
        return response
    except HTTPException as e:
        final_status = e.status_code
        raise
    except asyncio.CancelledError:
        outcome = "cancelled"  # 응답을 보내지 않았으니 상태 코드를 지어내지 않는다
        raise
    finally:
        if trace.answered_by is not None:
            sentry_sdk.set_tag("verify_answered_by", trace.answered_by)
        verify_log.info("verify %s", json.dumps(_log_fields(trace, response, final_status, outcome), ensure_ascii=False))


async def _verify(request: VerifyRequest, trace: VerifyTrace) -> VerifyResponse:
    assert_allowed_source(request.image_url)  # 1
    # 좌표가 없으면 위치로는 판정할 수 없으므로 사진 판정(VLM)으로 넘긴다
    if request.photo_coordinates is not None and is_clear_mismatch(  # 2
        request.photo_coordinates, request.place_coordinates
    ):
        return location_mismatch_response()

    image_bytes = await fetch_image(request.image_url)  # 3
    processed = prepare_image(image_bytes)  # 4
    scored = await score_photo(processed, request.place_name, request.mission_description, trace)  # 5
    vlm = scored.vlm
    if isinstance(vlm, QuestionAnswer):  # 설정이 아니라 실제로 받은 답의 형식으로 고른다
        return question_response(vlm)  # 6·7

    # VLM 원값을 그대로 7번에 넘긴다. 등급(6번)과 노출 여부는 build_response가 답한 모델의 기준으로 정한다.
    return build_response(  # 6·7
        vlm.match_score,
        thresholds_for(scored.answered_by),
        vlm.detected_labels,
        raw_landmark_confidence=vlm.landmark_confidence,
        raw_retry_hint=vlm.retry_hint,
    )


def _log_fields(trace: VerifyTrace, response: VerifyResponse | None, final_status: int, outcome: str) -> dict:
    # 폴백률·원인을 나중에 셀 수 있게 요청마다 같은 칸으로 남긴다. 사진·주소·키·에러 원문은 넣지 않는다
    # route: not_started(위치 반려·사진 처리에서 끝남) / self / off_hours(운영 시간 밖, 폴백 아님) / gemini_backend
    return {
        "configured_backend": settings.verify_backend,
        "vlm_format": settings.vlm_format,
        "route": trace.route,
        "answered_by": trace.answered_by,
        "result": response.result if response else None,
        "reason": response.reason if response else None,
        "fallback_enabled": trace.fallback_enabled,
        "fallback_attempted": trace.fallback_attempted,
        "fallback_succeeded": trace.fallback_succeeded,
        "fallback_skip_reason": trace.fallback_skip_reason,
        "self_error_status": trace.self_error_status,
        "self_upstream_status": trace.self_call.get("upstream_status"),
        "self_last_error": trace.self_call.get("last_error"),
        "self_attempts": trace.self_call.get("attempts"),
        "budget_sec": trace.budget_sec,
        "self_sec": trace.self_sec,
        "gemini_sec": trace.gemini_sec,
        "model_stage_sec": trace.model_stage_sec,
        "final_status": final_status if outcome != "cancelled" else None,
        "outcome": outcome,
    }
