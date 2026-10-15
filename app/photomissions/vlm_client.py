# 자체 모델(OpenAI 호환 서버, RunPod vLLM 등) 호출을 한 곳에 모은다. VERIFY_BACKEND=self일 때만 쓴다

import asyncio
import base64
import math
from typing import NoReturn
from urllib.parse import urlparse

import httpx
import sentry_sdk
from fastapi import HTTPException, status

from app.core.config import Settings
from app.photomissions.prompts import VERIFY_SYSTEM_PROMPT
from app.photomissions.schemas import VlmResult

MAX_ATTEMPTS = 2  # 최초 1회 + 재시도 1회 (Gemini와 같은 정책)
MAX_TOKENS = 512  # 점수 형식 답은 10/13 측정 287건에서 중앙값 111, 최대 141토큰이었다
# 연결·전송·연결 대기는 정상이면 금방 끝나서 짧게 둔다. 답을 기다리는 시간(읽기)은 전체 마감까지 쓴다
CONNECT_TIMEOUT = 5.0
WRITE_TIMEOUT = 10.0
POOL_TIMEOUT = 2.0
# TODO: 동시 연결 수는 32번(서버리스 동시성·콜드스타트 측정) 후 확정
MAX_CONNECTIONS = 20
DEFAULT_RETRY_AFTER = "30"  # 상대 서버가 Retry-After를 안 주면 Gemini 429와 같은 값을 쓴다
# 바깥 안전망은 마감보다 조금 늦게 둔다. 마감 경계에서는 httpx 시간 초과가 먼저 나서 원인별 코드(503·502·504)가 남는다
# 이벤트 루프가 오래 멈추면 안전망이 먼저 날 수 있고, 그때는 504다
DEADLINE_GUARD = 0.1

_client: httpx.AsyncClient | None = None
_settings: Settings | None = None


def settings_errors(s: Settings) -> list[str]:
    # self 판정에 필요한 설정이 빠지거나 틀린 것을 모두 모은다. 하나라도 있으면 앱이 시작되지 않는다
    # 서버는 켜졌는데 판정마다 502가 나는 것보다, 배포할 때 바로 멈추는 편이 원인을 찾기 쉽다
    errors = []
    url = urlparse(s.vlm_base_url.strip())
    if url.scheme not in ("http", "https") or not url.hostname:
        errors.append("VLM_BASE_URL은 http(s)://로 시작하는 서버 주소여야 함")
    if not s.vlm_model.strip():
        errors.append("VLM_MODEL이 비어 있음")
    deadline = s.vlm_deadline_seconds
    if deadline is None or not math.isfinite(deadline) or deadline <= 0:  # inf면 마감이 사라지고 nan은 비교가 안 된다
        errors.append("VLM_DEADLINE_SECONDS는 0보다 큰 유한한 수여야 함")

    success, retry, landmark = s.self_success_threshold, s.self_retry_threshold, s.self_landmark_threshold
    if success is None or retry is None or landmark is None:
        # Gemini 기준(90/40/60)은 Pro로 정한 값이라 자체 모델에 그대로 쓰면 판정 의미가 달라진다
        errors.append("SELF_SUCCESS_THRESHOLD·SELF_RETRY_THRESHOLD·SELF_LANDMARK_THRESHOLD가 모두 필요함")
    else:
        if not 0 <= retry <= success <= 100:
            errors.append("판정 기준은 0 ≤ SELF_RETRY_THRESHOLD ≤ SELF_SUCCESS_THRESHOLD ≤ 100이어야 함")
        if not 0 <= landmark <= 100:
            errors.append("SELF_LANDMARK_THRESHOLD는 0~100이어야 함")
    return errors


def check_settings(s: Settings) -> None:
    if errors := settings_errors(s):
        raise RuntimeError("VERIFY_BACKEND=self 설정 오류: " + " / ".join(errors))


def backend_summary(s: Settings) -> str:
    # 서버를 켤 때 어느 모델로 판정하는지 한 줄로 보여 준다. 키와 서버 주소(Pod ID가 들어 있음)는 넣지 않는다
    if s.verify_backend != "self":
        return "verify 판정 백엔드: gemini"
    return (
        f"verify 판정 백엔드: self (모델 {s.vlm_model}, 마감 {s.vlm_deadline_seconds:g}초, "
        f"기준 성공 {s.self_success_threshold:g} / 재시도 {s.self_retry_threshold:g} / "
        f"랜드마크 {s.self_landmark_threshold:g})"
    )


# 클라이언트 수명 — 앱이 켜질 때 하나 만들어 꺼질 때까지 재사용한다
# score_photo는 요청 객체를 받지 않아 app.state로 꺼낼 수 없어서 이 모듈이 보관한다

def start(s: Settings, transport: httpx.AsyncBaseTransport | None = None) -> None:
    # 앱 시작 때 check_settings가 통과한 뒤에만 부른다. transport는 테스트에서 가짜 서버를 끼울 때만 넘긴다
    global _client, _settings
    headers = {"Authorization": f"Bearer {s.vlm_api_key}"} if s.vlm_api_key else {}
    _client = httpx.AsyncClient(
        base_url=s.vlm_base_url.strip().rstrip("/") + "/",
        headers=headers,
        limits=httpx.Limits(max_connections=MAX_CONNECTIONS),
        transport=transport,
    )
    _settings = s


async def close() -> None:
    global _client, _settings
    try:
        if _client is not None:
            await _client.aclose()
    finally:  # 닫다가 실패하거나 취소돼도 닫힌 클라이언트를 다시 쓰지 않게 비운다
        _client, _settings = None, None


def get_client() -> tuple[httpx.AsyncClient, Settings]:
    # 시작 단계를 거치지 않았으면 몰래 새로 만들지 않고 에러를 낸다 (연결 설정 누락을 숨기지 않기 위해)
    if _client is None or _settings is None:
        raise RuntimeError("자체 모델 클라이언트가 시작되지 않음 (앱 시작 단계에서 vlm_client.start 필요)")
    return _client, _settings


# 판정 호출

def build_payload(model: str, image_bytes: bytes, place_name: str, mission_description: str) -> dict:
    # 측정 스크립트(judge_local)와 같은 요청이어야 dev 측정 결과가 서비스에도 그대로 맞는다
    # 온도는 측정 때처럼 넣지 않는다 (vLLM은 모델 설정 파일의 기본값을 쓴다)
    image_url = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode()
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": f"목표 장소: {place_name}\n미션 내용: {mission_description}"},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": VlmResult.__name__, "schema": VlmResult.model_json_schema()},
        },
        "max_tokens": MAX_TOKENS,
    }


class _Retryable(Exception):
    # 한 번 더 시도하면 나을 수 있는 실패. 재시도를 못 하면 담긴 응답(502)을 그대로 올린다
    def __init__(self, error: HTTPException):
        self.error = error


def _bad_gateway() -> HTTPException:
    return HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="사진 판정 실패")


def _retry_after(response: httpx.Response) -> str:
    # 상대 서버가 숫자(초)로 준 경우만 그대로 넘긴다. 날짜 형식 등은 기본값을 쓴다
    value = response.headers.get("Retry-After", "").strip()
    return value if value.isdigit() else DEFAULT_RETRY_AFTER


def _parse(response: httpx.Response) -> VlmResult:
    # 답 형식이 하나라도 어긋나면 재시도 대상이다 (JSON 깨짐, choices 비어 있음, content 없음, 스키마 불일치)
    try:
        content = response.json()["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise TypeError("content가 문자열이 아님")
        return VlmResult.model_validate_json(content)
    # ValueError: JSON 깨짐과 UTF-8이 아닌 응답(UnicodeDecodeError)을 함께 잡는다. ValidationError도 여기 속한다
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise _Retryable(_bad_gateway()) from e


async def _attempt(
    client: httpx.AsyncClient, payload: dict, remaining: float, trace: dict
) -> VlmResult:
    # 한 번 호출한다. 시간 초과 값은 모두 남은 시간 이하로 둔다
    timeout = httpx.Timeout(
        connect=min(CONNECT_TIMEOUT, remaining),
        read=remaining,
        write=min(WRITE_TIMEOUT, remaining),
        pool=min(POOL_TIMEOUT, remaining),
    )
    trace["upstream_status"], trace["last_error"] = None, None  # 마지막 시도 결과만 남긴다
    try:
        response = await client.post("chat/completions", json=payload, timeout=timeout)
    except httpx.HTTPError as e:
        trace["last_error"] = type(e).__name__  # 예외 문자열에는 서버 주소가 들어 있을 수 있어 종류만 남긴다
        _raise_transport_error(e)

    trace["upstream_status"] = response.status_code
    if response.status_code == status.HTTP_429_TOO_MANY_REQUESTS:  # 바로 다시 불러도 또 막히므로 재시도 안 함
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="자체 모델 호출 한도 초과",
            headers={"Retry-After": _retry_after(response)},
        )
    if response.status_code >= 500:
        raise _Retryable(_bad_gateway())
    if response.status_code >= 400:  # 401·403·404 등은 설정 문제라 다시 불러도 같다
        raise _bad_gateway()
    try:
        return _parse(response)
    except _Retryable as e:
        trace["last_error"] = type(e.__cause__).__name__  # 예: JSONDecodeError, ValidationError
        raise


def _raise_transport_error(e: httpx.HTTPError) -> NoReturn:
    # 응답을 못 받은 경우. httpx 시간 초과는 상속 관계가 있어 구체적인 것부터 본다
    if isinstance(e, httpx.PoolTimeout):  # 우리 쪽 동시 연결이 꽉 참
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="사진 판정 요청이 몰림",
        ) from e
    if isinstance(e, httpx.ConnectTimeout):  # 연결도 못 함 → 상대 서버가 아직 안 떴을 수 있어 재시도
        raise _Retryable(_bad_gateway()) from e
    if isinstance(e, (httpx.ReadTimeout, httpx.WriteTimeout)):  # 답을 기다리다 남은 시간을 다 씀
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="사진 판정 시간 초과",
        ) from e
    raise _Retryable(_bad_gateway()) from e  # 연결 실패·끊김


def _can_retry(remaining: float) -> bool:
    # 남은 시간이 없으면 새 요청을 시작하지 않는다. 연결 시간 초과값보다 짧으면 연결만 하다 끝날 수 있어 하지 않는다
    return remaining > 0 and remaining > CONNECT_TIMEOUT


async def score(image_bytes: bytes, place_name: str, mission_description: str) -> VlmResult:
    # 재시도까지 합쳐 VLM_DEADLINE_SECONDS 안에 끝낸다. 마감은 여기 들어온 순간부터 잰다 (사진 받기·줄이기 제외)
    client, s = get_client()
    payload = build_payload(s.vlm_model, image_bytes, place_name, mission_description)
    loop = asyncio.get_running_loop()
    started = loop.time()  # 시스템 시계가 바뀌어도 흔들리지 않는 시계
    deadline = started + float(s.vlm_deadline_seconds or 0)
    trace: dict = {"attempts": 0, "upstream_status": None, "last_error": None}
    try:
        # 마감(deadline)은 한 번만 정하고 재시도 때 다시 계산하지 않는다. 바깥 안전망만 DEADLINE_GUARD만큼 늦다
        async with asyncio.timeout_at(deadline + DEADLINE_GUARD):
            while True:
                trace["attempts"] += 1
                try:
                    return await _attempt(client, payload, deadline - loop.time(), trace)
                except _Retryable as e:
                    if trace["attempts"] >= MAX_ATTEMPTS or not _can_retry(deadline - loop.time()):
                        raise e.error from e.__cause__  # 재시도를 못 하면 원래 원인(502) 그대로
    except TimeoutError as e:  # 바깥 안전망이 요청을 끊음
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="사진 판정 시간 초과",
        ) from e
    finally:
        # 키·사진·서버 주소는 남기지 않는다
        sentry_sdk.set_context("vlm", {**trace, "elapsed_sec": round(loop.time() - started, 3)})
