# Sentry 에러 모니터링 초기화. SENTRY_DSN이 비어 있으면 켜지 않는다 (로컬·테스트)

import tomllib
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

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


def _release() -> str:
    # 버전 번호는 사람이 올려야 해서 깜빡할 수 있다. 커밋 SHA를 붙여 배포마다 release가 달라지게 한다
    # "+"는 semver의 빌드 정보 표기다. SHA가 없으면(로컬) 버전만 쓴다
    sha = settings.git_sha.strip()[:7]
    version = f"ktb4-ai-server@{_app_version()}"
    return f"{version}+{sha}" if sha else version


def _strip_query(url: str) -> str:
    # presigned URL의 서명(X-Amz-Signature 등)이 쿼리에 있어서 쿼리를 통째로 뗀다
    return url.split("?", 1)[0]


def _is_vlm_url(url: str) -> bool:
    # 자체 모델 서버 주소에는 Pod·엔드포인트 ID가 호스트나 경로에 들어 있다. 호스트 이름으로 비교한다
    vlm_host = urlparse(settings.vlm_base_url.strip()).hostname
    return vlm_host is not None and urlparse(url).hostname == vlm_host


def _clean_url(url: str) -> str:
    return _FILTERED if _is_vlm_url(url) else _strip_query(url)


def _clean_http_data(data: dict[str, Any]) -> None:
    # httpx 호출 기록(breadcrumb·span)의 URL을 정리한다
    data.pop("http.query", None)
    if isinstance(data.get("url"), str):
        data["url"] = _clean_url(data["url"])


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
        _clean_http_data(span.get("data") or {})
        # span 이름도 "POST https://..." 형태라 주소가 들어 있다
        method, _, url = (span.get("description") or "").partition(" ")
        if url and _is_vlm_url(url):
            span["description"] = f"{method} {_FILTERED}"
    return event


def scrub_breadcrumb(crumb: dict[str, Any], hint: dict[str, Any] | None = None) -> dict[str, Any]:
    # 외부 호출 기록에서 URL 쿼리를 지우고(S3 사진 다운로드 등), 자체 모델 서버 주소는 통째로 가린다
    _clean_http_data(crumb.get("data") or {})
    return crumb


def init_sentry() -> bool:
    # 켰으면 True. DSN이 없으면 아무것도 하지 않는다
    if not settings.sentry_dsn:
        return False
    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.sentry_environment,
        release=_release(),
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
