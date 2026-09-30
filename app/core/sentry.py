# Sentry 에러 모니터링 초기화. SENTRY_DSN이 비어 있으면 켜지 않는다 (로컬·테스트)

import tomllib
from pathlib import Path
from typing import Any, cast

import sentry_sdk

from app.core.config import settings

# 우리가 만든 헤더라 SDK 기본 필터(Authorization, Cookie 등)에 없어서 직접 지운다
_SENSITIVE_HEADERS = {"x-internal-token"}
_FILTERED = "[Filtered]"
_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _app_version() -> str:
    # 배포 버전은 pyproject.toml의 version 한 곳에서만 관리한다
    try:
        return tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["project"]["version"]
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return "unknown"


def _strip_query(url: str) -> str:
    # presigned URL의 서명(X-Amz-Signature 등)이 쿼리에 있어서 쿼리를 통째로 뗀다
    return url.split("?", 1)[0]


def scrub_event(event: dict[str, Any], hint: dict[str, Any] | None = None) -> dict[str, Any]:
    # 에러·트랜잭션 이벤트를 보내기 직전에 민감한 값을 지운다
    request = event.get("request") or {}
    headers = request.get("headers") or {}
    for key in list(headers):
        if key.lower() in _SENSITIVE_HEADERS:
            headers[key] = _FILTERED
    # 요청 본문에는 사진 주소 서명(verify)과 사용자 정보(trips)가 있어 보내지 않는다
    request.pop("data", None)
    request.pop("query_string", None)

    for span in event.get("spans") or []:
        data = span.get("data") or {}
        data.pop("http.query", None)
        if isinstance(data.get("url"), str):
            data["url"] = _strip_query(data["url"])
    return event


def scrub_breadcrumb(crumb: dict[str, Any], hint: dict[str, Any] | None = None) -> dict[str, Any]:
    # 외부 호출(S3 사진 다운로드 등) 기록에서 URL 쿼리를 지운다
    data = crumb.get("data") or {}
    data.pop("http.query", None)
    if isinstance(data.get("url"), str):
        data["url"] = _strip_query(data["url"])
    return crumb


def init_sentry() -> bool:
    # 켰으면 True. DSN이 없으면 아무것도 하지 않는다
    if not settings.sentry_dsn:
        return False
    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.sentry_environment,
        release=f"ktb4-ai-server@{_app_version()}",
        traces_sample_rate=settings.sentry_traces_sample_rate,
        send_default_pii=False,  # IP·쿠키 같은 개인정보를 보내지 않는다
        include_local_variables=False,  # 스택의 지역 변수에 토큰·사진 주소가 들어 있을 수 있다
        max_request_body_size="never",  # 요청 본문은 수집하지 않는다
        # SDK의 이벤트 타입(TypedDict)은 키 조작이 번거로워 일반 dict로 다루고 여기서만 맞춘다
        before_send=cast(Any, scrub_event),
        before_send_transaction=cast(Any, scrub_event),
        before_breadcrumb=cast(Any, scrub_breadcrumb),
    )
    return True
