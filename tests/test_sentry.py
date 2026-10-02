import sentry_sdk

from app.core import sentry
from app.core.config import settings

SIGNED_URL = "https://my-bucket.s3.ap-northeast-2.amazonaws.com/photo.jpg?X-Amz-Signature=secret"


def test_internal_token_header_is_filtered():
    event = {"request": {"headers": {"X-Internal-Token": "real-token", "Content-Type": "application/json"}}}
    headers = sentry.scrub_event(event)["request"]["headers"]
    assert headers["X-Internal-Token"] == "[Filtered]"
    assert headers["Content-Type"] == "application/json"


def test_request_body_and_query_are_removed():
    # verify 본문의 사진 주소 서명, trips 본문의 사용자 정보가 나가지 않아야 한다
    event = {"request": {"data": {"image_url": SIGNED_URL}, "query_string": "a=1"}}
    request = sentry.scrub_event(event)["request"]
    assert "data" not in request
    assert "query_string" not in request


def test_span_url_query_is_removed():
    event = {"spans": [{"data": {"url": SIGNED_URL, "http.query": "X-Amz-Signature=secret"}}]}
    data = sentry.scrub_event(event)["spans"][0]["data"]
    assert data["url"] == SIGNED_URL.split("?")[0]
    assert "http.query" not in data


def test_breadcrumb_url_query_is_removed():
    # S3 사진 다운로드 같은 외부 호출 기록에 서명이 남지 않아야 한다
    crumb = {"category": "httplib", "data": {"url": SIGNED_URL, "http.query": "X-Amz-Signature=secret"}}
    data = sentry.scrub_breadcrumb(crumb)["data"]
    assert "secret" not in data["url"]
    assert "http.query" not in data


def test_sentry_not_initialized_without_dsn(monkeypatch):
    calls = []
    monkeypatch.setattr(settings, "sentry_dsn", "")
    monkeypatch.setattr(sentry_sdk, "init", lambda **kwargs: calls.append(kwargs))
    assert sentry.init_sentry() is False
    assert calls == []


def test_release_has_commit_sha_when_given(monkeypatch):
    monkeypatch.setattr(settings, "git_sha", "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678")
    assert sentry._release() == f"ktb4-ai-server@{sentry._app_version()}+a1b2c3d"


def test_release_is_version_only_without_sha(monkeypatch):
    monkeypatch.setattr(settings, "git_sha", "")
    assert sentry._release() == f"ktb4-ai-server@{sentry._app_version()}"


def test_sentry_initialized_with_safe_options(monkeypatch):
    calls = []
    monkeypatch.setattr(settings, "git_sha", "")
    monkeypatch.setattr(settings, "sentry_dsn", "https://key@o0.ingest.sentry.io/0")
    monkeypatch.setattr(sentry_sdk, "init", lambda **kwargs: calls.append(kwargs))
    assert sentry.init_sentry() is True
    options = calls[0]
    assert options["send_default_pii"] is False
    assert options["include_local_variables"] is False
    assert options["max_request_body_size"] == "never"
    assert options["before_send"] is sentry.scrub_event
    assert options["before_breadcrumb"] is sentry.scrub_breadcrumb
    # 버전은 pyproject.toml 한 곳에서 관리하므로 숫자를 박지 않고 그 값과 비교한다
    assert options["release"] == f"ktb4-ai-server@{sentry._app_version()}"
    assert sentry._app_version() != "unknown"
