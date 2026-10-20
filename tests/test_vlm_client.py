import inspect
import io
import json
import logging
import time

import anyio
import httpx
import pytest
import sentry_sdk
from fastapi import HTTPException
from PIL import Image
from sentry_sdk.integrations.httpx import HttpxIntegration
from sentry_sdk.integrations.logging import LoggingIntegration

from app.core import sentry
from app.core.config import Settings, settings
from app.main import app, lifespan
from app.photomissions import pipeline_generate, vlm_client
from app.photomissions.pipeline_verify import normalize_image
from app.photomissions.question_format import QUESTION_SYSTEM_PROMPT, QuestionAnswer
from app.photomissions.schemas import VlmResult
from scripts.baseline_verify import judge_local

pytestmark = pytest.mark.anyio

SELF_OK = {
    "verify_backend": "self",
    "vlm_base_url": "https://vlm.example.com/v1",
    "vlm_model": "qwen",
    "vlm_deadline_seconds": 60,
    "self_success_threshold": 80,
    "self_retry_threshold": 40,
    "self_landmark_threshold": 50,
}


def _settings(**overrides) -> Settings:
    # .env를 읽지 않아야 로컬 설정에 따라 결과가 달라지지 않는다
    return Settings(_env_file=None, **{**SELF_OK, **overrides})


# 1. 시작 검사 — self에 필요한 설정이 빠지거나 틀리면 모두 잡는다

def test_complete_self_settings_pass():
    assert vlm_client.settings_errors(_settings()) == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"vlm_base_url": ""},
        {"vlm_base_url": "   "},
        {"vlm_base_url": "vlm.example.com/v1"},  # http(s)://가 빠짐
        {"vlm_base_url": "ftp://vlm.example.com/v1"},
        {"vlm_model": " "},
        {"vlm_deadline_seconds": None},
        {"vlm_deadline_seconds": 0},
        {"vlm_deadline_seconds": float("inf")},  # 마감이 사라짐
        {"vlm_deadline_seconds": float("nan")},  # 어떤 비교도 거짓이라 검사를 빠져나감
        {"self_success_threshold": None},
        {"self_retry_threshold": None},
        {"self_landmark_threshold": None},
        {"self_retry_threshold": 90},  # 재시도 기준이 성공 기준보다 높음
        {"self_success_threshold": 120},
        {"self_retry_threshold": -1},
        {"self_landmark_threshold": 101},
    ],
)
def test_wrong_self_settings_are_reported(overrides):
    assert vlm_client.settings_errors(_settings(**overrides))


def test_check_settings_reports_every_problem_at_once():
    # 배포할 때 한 번에 다 고칠 수 있게 하나씩 멈추지 않는다
    with pytest.raises(RuntimeError) as exc:
        vlm_client.check_settings(_settings(vlm_model="", vlm_deadline_seconds=None))
    assert "VLM_MODEL" in str(exc.value) and "VLM_DEADLINE_SECONDS" in str(exc.value)


# 2. 앱 시작 — 공용 test_client(테스트 전체에서 한 번만 시작)와 섞이지 않게 lifespan을 직접 실행한다

async def test_self_backend_with_missing_settings_does_not_start(monkeypatch):
    monkeypatch.setattr(settings, "verify_backend", "self")
    monkeypatch.setattr(settings, "vlm_base_url", "")
    with pytest.raises(RuntimeError):
        async with lifespan(app):
            pass


async def test_startup_failure_still_clears_generate(monkeypatch):
    # generate 상한을 만든 뒤 self 설정 검사가 실패해도, 만든 것은 비우고 멈춘다
    monkeypatch.setattr(settings, "verify_backend", "self")
    monkeypatch.setattr(settings, "vlm_base_url", "")
    with pytest.raises(RuntimeError):
        async with lifespan(app):
            pass
    assert pipeline_generate._call_slots is None


async def test_generate_is_cleared_even_if_vlm_close_fails(monkeypatch):
    monkeypatch.setattr(settings, "verify_backend", "gemini")

    async def broken_close():
        raise RuntimeError("종료 정리 실패")

    monkeypatch.setattr(vlm_client, "close", broken_close)
    with pytest.raises(RuntimeError, match="종료 정리 실패"):
        async with lifespan(app):
            assert pipeline_generate._call_slots is not None
    assert pipeline_generate._call_slots is None


async def test_gemini_backend_starts_without_vlm_settings(monkeypatch, caplog):
    monkeypatch.setattr(settings, "verify_backend", "gemini")
    monkeypatch.setattr(settings, "vlm_base_url", "")
    monkeypatch.setattr(settings, "self_success_threshold", None)
    with caplog.at_level("INFO", logger="uvicorn.error"):
        async with lifespan(app):
            pass
    assert "verify 판정 백엔드: gemini" in caplog.text


# 3. 시작 로그 — 개발 중에 지금 어느 모델로 판정하는지 바로 보이게 한다

def test_self_backend_summary_shows_model_and_thresholds_without_secrets():
    summary = vlm_client.backend_summary(_settings(vlm_api_key="secret-key"))
    assert summary == "verify 판정 백엔드: self (모델 qwen, 마감 60초, 형식 점수, 기준 성공 80 / 재시도 40 / 랜드마크 50) · 폴백 없음"
    assert "secret-key" not in summary and "vlm.example.com" not in summary


# 1-2. 폴백·운영 시간 설정

FALLBACK = {"verify_model_budget_seconds": 20, "gemini_api_key": "gemini-key"}


def test_fallback_settings_pass():
    assert vlm_client.settings_errors(_settings(**FALLBACK)) == []
    assert vlm_client.settings_errors(_settings(**FALLBACK, vlm_serving_hours="09:00-17:00")) == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"vlm_serving_hours": "09:00-17:00"},  # 운영 시간만 있고 예산 없음
        {**FALLBACK, "gemini_api_key": ""},  # 폴백할 Gemini 키 없음
        {**FALLBACK, "gemini_api_key": "  "},
        {**FALLBACK, "verify_model_budget_seconds": 0},
        {**FALLBACK, "verify_model_budget_seconds": float("inf")},
        {**FALLBACK, "verify_model_budget_seconds": float("nan")},
        {**FALLBACK, "verify_model_budget_seconds": 15},  # Gemini 몫 10초 + 여유를 빼면 자체 모델이 연결도 못 함
        {**FALLBACK, "vlm_serving_hours": "9:00-17:00"},
        {**FALLBACK, "vlm_serving_hours": "09:00~17:00"},
        {**FALLBACK, "vlm_serving_hours": "09:00"},
        {**FALLBACK, "vlm_serving_hours": "25:00-26:00"},
        {**FALLBACK, "vlm_serving_hours": "09:60-17:00"},
        {**FALLBACK, "vlm_serving_hours": "17:00-17:00"},
        {**FALLBACK, "vlm_serving_hours": "22:00-06:00"},  # 자정 넘는 구간은 지원하지 않음
    ],
)
def test_wrong_fallback_settings_are_reported(overrides):
    assert vlm_client.settings_errors(_settings(**overrides))


def test_fallback_checks_also_run_in_question_format():
    # 객관식은 점수 기준 검사를 건너뛰지만 폴백 검사는 건너뛰면 안 된다
    no_thresholds = {"self_success_threshold": None, "self_retry_threshold": None, "self_landmark_threshold": None}
    assert vlm_client.settings_errors(_settings(vlm_format="question", vlm_serving_hours="09:00-17:00", **no_thresholds))
    assert vlm_client.settings_errors(_settings(vlm_format="question", **no_thresholds, **FALLBACK)) == []


def test_budget_at_the_minimum_passes():
    least = vlm_client.MIN_GEMINI_START_BUDGET_SECONDS + vlm_client.BUDGET_MARGIN_SECONDS + vlm_client.CONNECT_TIMEOUT
    assert vlm_client.settings_errors(_settings(**{**FALLBACK, "verify_model_budget_seconds": least})) == []


@pytest.mark.parametrize(
    ("clock", "expected"),
    [("08:59:59", False), ("09:00:00", True), ("16:59:59", True), ("17:00:00", False), ("00:00:00", False)],
)
def test_serving_hours_include_start_exclude_end(clock, expected):
    from datetime import datetime

    from app.core.time import SERVICE_TIMEZONE

    now = datetime.fromisoformat(f"2026-10-20T{clock}").replace(tzinfo=SERVICE_TIMEZONE)
    assert vlm_client.in_serving_hours(now, vlm_client.parse_serving_hours("09:00-17:00")) is expected


def test_empty_serving_hours_means_always():
    from datetime import datetime

    from app.core.time import SERVICE_TIMEZONE

    assert vlm_client.parse_serving_hours("  ") is None
    assert vlm_client.in_serving_hours(datetime(2026, 10, 20, 3, 0, tzinfo=SERVICE_TIMEZONE), None) is True


def test_summary_shows_fallback_without_secrets():
    summary = vlm_client.backend_summary(_settings(**FALLBACK, vlm_serving_hours="09:00-17:00"))
    assert summary.endswith(" · 폴백 Gemini (모델 단계 예산 20초, 자체 모델 운영 시간 09:00-17:00)")
    assert "gemini-key" not in summary


def test_budget_warning_replaces_self_only_warning():
    # 폴백을 켜면 모델 단계 상한은 예산이라, 경고도 예산 기준이다
    assert vlm_client.deadline_warning(_settings(**FALLBACK, vlm_deadline_seconds=25)) is None
    assert "VERIFY_MODEL_BUDGET_SECONDS=25.5초" in vlm_client.deadline_warning(_settings(**{**FALLBACK, "verify_model_budget_seconds": 25.5}))
    assert vlm_client.deadline_warning(_settings(**{**FALLBACK, "verify_model_budget_seconds": 20})) is None


def test_question_format_needs_no_score_thresholds():
    # 객관식은 등급 규칙을 쓰므로 점수 기준 없이 시작한다. 점수 형식은 지금처럼 세 값이 모두 필요하다
    no_thresholds = {"self_success_threshold": None, "self_retry_threshold": None, "self_landmark_threshold": None}
    assert vlm_client.settings_errors(_settings(vlm_format="question", **no_thresholds)) == []
    assert vlm_client.settings_errors(_settings(vlm_format="score", **no_thresholds))
    assert vlm_client.settings_errors(_settings(vlm_format="question", vlm_model=" "))  # 다른 검사는 그대로


def test_question_summary_names_the_rule_not_thresholds():
    summary = vlm_client.backend_summary(_settings(
        vlm_format="question", vlm_api_key="secret-key",
        self_success_threshold=None, self_retry_threshold=None, self_landmark_threshold=None,
    ))
    assert summary.startswith("verify 판정 백엔드: self (모델 qwen, 마감 60초, 형식 객관식, 등급 규칙 rule_grade")
    assert "None" not in summary and "secret-key" not in summary and "vlm.example.com" not in summary


async def test_self_backend_starts_and_closes_client(monkeypatch):
    # 시작 단계에서 클라이언트를 만들고, 앱이 꺼질 때(중간에 에러가 나도) 닫는지
    for key, value in SELF_OK.items():
        monkeypatch.setattr(settings, key, value)
    calls = []
    monkeypatch.setattr(vlm_client, "start", lambda s: calls.append("start"))

    async def fake_close():
        calls.append("close")

    monkeypatch.setattr(vlm_client, "close", fake_close)
    with pytest.raises(ValueError):
        async with lifespan(app):
            assert calls == ["start"]
            raise ValueError("앱 실행 중 에러")
    assert calls == ["start", "close"]


# 4. 판정 호출 — 가짜 서버(MockTransport)를 vlm_client에만 끼운다. 실제 네트워크 호출은 없다

VLM_RESULT = VlmResult(detected_labels=[], landmark_confidence=90, match_score=85, retry_hint="")
PLACE, MISSION = "첨성대", "첨성대 정면이 보이게 찍기"


def _chat(content) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def ok(request):
    return _chat(VLM_RESULT.model_dump_json())


def status_(code, headers=None):
    return lambda request: httpx.Response(code, headers=headers)


def raise_(error_cls):
    def outcome(request):
        raise error_cls("fake", request=request)
    return outcome


def slow(seconds, then):
    async def outcome(request):
        await anyio.sleep(seconds)
        return then(request)
    return outcome


class FakeVlm:
    # outcomes: 호출마다 순서대로 쓸 응답. 몇 번 불렸는지와 받은 요청을 남긴다
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests: list[httpx.Request] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        result = self.outcomes[len(self.requests) - 1](request)
        return await result if inspect.isawaitable(result) else result


@pytest.fixture
async def fake_vlm():
    def install(*outcomes, **overrides) -> FakeVlm:
        server = FakeVlm(outcomes)
        vlm_client.start(_settings(vlm_api_key="test-key", **overrides), transport=httpx.MockTransport(server))
        return server

    yield install
    await vlm_client.close()


async def _score():
    return await vlm_client.score(b"jpeg", PLACE, MISSION)


async def test_success_on_first_try(fake_vlm):
    server = fake_vlm(ok)
    assert await _score() == VLM_RESULT
    assert len(server.requests) == 1


ANSWER = QuestionAnswer(judgeable="yes", target="short", condition="na", place="unknown")


async def test_question_format_returns_the_answer(fake_vlm):
    server = fake_vlm(lambda r: _chat(ANSWER.model_dump_json()), vlm_format="question")
    assert await _score() == ANSWER
    sent = json.loads(server.requests[0].content)
    assert sent["messages"][0]["content"] == QUESTION_SYSTEM_PROMPT
    assert sent["response_format"]["json_schema"]["name"] == "QuestionAnswer"


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(json.dumps({"judgeable": "yes", "target": None, "condition": None, "place": "yes"}), id="empty-target"),
        pytest.param(json.dumps({"judgeable": "yes", "target": "enough", "condition": None, "place": "yes"}), id="empty-condition"),
        pytest.param(VLM_RESULT.model_dump_json(), id="score-answer-in-question-mode"),
    ],
)
async def test_invalid_question_answer_is_retried_then_502(fake_vlm, content):
    # 서버가 스키마를 강제해도 Pydantic 검사(빈 칸이면 성공이 되는 답)는 여기서 다시 막는다
    server = fake_vlm(lambda r: _chat(content), lambda r: _chat(content), vlm_format="question")
    with pytest.raises(HTTPException) as exc:
        await _score()
    assert exc.value.status_code == 502
    assert len(server.requests) == 2


def question_ok(request):
    return _chat(ANSWER.model_dump_json())


async def test_question_bad_answer_then_valid_answer_succeeds(fake_vlm, monkeypatch):
    recorded = {}
    monkeypatch.setattr(sentry_sdk, "set_context", lambda name, value: recorded.update({name: value}))
    bad = json.dumps({"judgeable": "yes", "target": None, "condition": None, "place": "yes"})
    server = fake_vlm(lambda r: _chat(bad), question_ok, vlm_format="question")
    assert await _score() == ANSWER
    assert len(server.requests) == 2
    assert recorded["vlm"]["attempts"] == 2 and recorded["vlm"]["last_error"] is None


async def test_question_failure_is_recorded_like_score_mode(fake_vlm, monkeypatch):
    # 객관식도 마지막 시도의 원인이 Sentry에 남는다 (형식 검증 실패는 ValidationError)
    recorded = {}
    monkeypatch.setattr(sentry_sdk, "set_context", lambda name, value: recorded.update({name: value}))
    bad = json.dumps({"judgeable": "yes", "target": "enough", "condition": None, "place": "yes"})
    fake_vlm(status_(500), lambda r: _chat(bad), vlm_format="question")
    with pytest.raises(HTTPException):
        await _score()
    assert recorded["vlm"]["attempts"] == 2
    assert recorded["vlm"]["upstream_status"] == 200
    assert recorded["vlm"]["last_error"] == "ValidationError"


async def test_question_rate_limit_keeps_retry_after_without_retry(fake_vlm):
    server = fake_vlm(status_(429, {"Retry-After": "12"}), vlm_format="question")
    with pytest.raises(HTTPException) as exc:
        await _score()
    assert (exc.value.status_code, exc.value.headers) == (429, {"Retry-After": "12"})
    assert len(server.requests) == 1


async def test_question_read_timeout_maps_to_504(fake_vlm):
    server = fake_vlm(raise_(httpx.ReadTimeout), vlm_format="question")
    with pytest.raises(HTTPException) as exc:
        await _score()
    assert exc.value.status_code == 504
    assert len(server.requests) == 1


async def test_question_answer_in_score_mode_is_not_accepted(fake_vlm):
    server = fake_vlm(lambda r: _chat(ANSWER.model_dump_json()), ok)
    assert await _score() == VLM_RESULT  # 첫 답은 점수 스키마와 안 맞아 재시도
    assert len(server.requests) == 2


RETRYABLE = [
    pytest.param(status_(500), id="500"),
    pytest.param(status_(503), id="503"),
    pytest.param(raise_(httpx.ConnectError), id="connect-error"),
    pytest.param(raise_(httpx.ConnectTimeout), id="connect-timeout"),
    pytest.param(raise_(httpx.RemoteProtocolError), id="connection-dropped"),
    pytest.param(lambda r: httpx.Response(200, content=b"not json"), id="broken-json"),
    pytest.param(lambda r: httpx.Response(200, content=b"\xff"), id="not-utf8"),
    pytest.param(lambda r: httpx.Response(200, json={"choices": []}), id="empty-choices"),
    pytest.param(lambda r: _chat(None), id="content-missing"),
    pytest.param(lambda r: _chat(123), id="content-not-string"),
    pytest.param(lambda r: _chat('{"match_score": 200}'), id="schema-mismatch"),
]


@pytest.mark.parametrize("failure", RETRYABLE)
async def test_retryable_failure_is_retried_once_then_succeeds(fake_vlm, failure):
    server = fake_vlm(failure, ok)
    assert await _score() == VLM_RESULT
    assert len(server.requests) == 2


@pytest.mark.parametrize("failure", RETRYABLE)
async def test_retryable_failure_twice_maps_to_502(fake_vlm, failure):
    server = fake_vlm(failure, failure)
    with pytest.raises(HTTPException) as exc:
        await _score()
    assert exc.value.status_code == 502
    assert len(server.requests) == 2


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        pytest.param(status_(401), 502, id="401"),
        pytest.param(status_(403), 502, id="403"),
        pytest.param(status_(404), 502, id="404"),
        pytest.param(raise_(httpx.ReadTimeout), 504, id="read-timeout"),
        pytest.param(raise_(httpx.WriteTimeout), 504, id="write-timeout"),
        pytest.param(raise_(httpx.PoolTimeout), 503, id="pool-full"),
    ],
)
async def test_non_retryable_failure_is_not_retried(fake_vlm, failure, expected):
    server = fake_vlm(failure, ok)
    with pytest.raises(HTTPException) as exc:
        await _score()
    assert exc.value.status_code == expected
    assert len(server.requests) == 1


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        pytest.param({"Retry-After": "12"}, "12", id="seconds"),
        pytest.param({}, "30", id="missing"),
        pytest.param({"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}, "30", id="date"),
        # 그대로 넘기면 응답 헤더 인코딩에서 500. 실제 서버처럼 바이트로 보낸다
        pytest.param([(b"Retry-After", "٣".encode())], "30", id="non-ascii-digit"),
        pytest.param({"Retry-After": "9" * 400}, "3600", id="too-large"),
    ],
)
async def test_rate_limit_maps_to_429_without_retry(fake_vlm, headers, expected):
    server = fake_vlm(status_(429, headers), ok)
    with pytest.raises(HTTPException) as exc:
        await _score()
    assert exc.value.status_code == 429
    assert exc.value.headers == {"Retry-After": expected}
    assert len(server.requests) == 1


# 5. 전체 마감 시간 — 재시도까지 합쳐 VLM_DEADLINE_SECONDS를 넘지 않는다

async def test_no_retry_when_too_little_time_left_keeps_502(fake_vlm, monkeypatch):
    # 0.15초 뒤 503 → 남은 시간(약 0.15초)이 연결 시간 초과값(0.2초)보다 짧아 재시도하지 않는다
    monkeypatch.setattr(vlm_client, "CONNECT_TIMEOUT", 0.2)
    server = fake_vlm(slow(0.15, status_(503)), ok, vlm_deadline_seconds=0.3)
    with pytest.raises(HTTPException) as exc:
        await _score()
    assert exc.value.status_code == 502  # 시간 초과(504)가 아니라 원래 원인
    assert len(server.requests) == 1


async def test_deadline_during_second_attempt_maps_to_504(fake_vlm, monkeypatch):
    monkeypatch.setattr(vlm_client, "CONNECT_TIMEOUT", 0.05)
    server = fake_vlm(status_(503), slow(5, ok), vlm_deadline_seconds=0.3)
    started = time.monotonic()
    with pytest.raises(HTTPException) as exc:
        await _score()
    assert exc.value.status_code == 504
    assert len(server.requests) == 2
    assert time.monotonic() - started < 1  # 두 번째 답(5초)을 기다리지 않고 마감에 끊긴다


async def test_one_deadline_for_all_attempts(fake_vlm):
    # 마감은 한 번만 정한다. 재시도의 시간 초과 값은 남은 시간이라 첫 시도보다 작아야 한다
    server = fake_vlm(status_(500), ok)
    await _score()
    first, second = (r.extensions["timeout"] for r in server.requests)
    assert first == {"connect": 5.0, "write": 10.0, "pool": 2.0, "read": pytest.approx(60, abs=0.5)}
    assert second["read"] < first["read"]


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        pytest.param(httpx.PoolTimeout, 503, id="pool-full"),
        pytest.param(httpx.ConnectTimeout, 502, id="connect-timeout"),
    ],
)
async def test_inner_timeout_at_deadline_keeps_its_cause(fake_vlm, monkeypatch, failure, expected):
    # 마감 경계에서 httpx 시간 초과가 나면 바깥 안전망(504)보다 먼저라 원인별 코드가 남는다
    # 시간 흔들림에 테스트가 흔들리지 않게 안전망 여유만 넓힌다
    monkeypatch.setattr(vlm_client, "DEADLINE_GUARD", 0.3)
    server = fake_vlm(slow(0.25, raise_(failure)), ok, vlm_deadline_seconds=0.2)
    with pytest.raises(HTTPException) as exc:
        await _score()
    assert exc.value.status_code == expected
    assert len(server.requests) == 1  # 남은 시간이 없어 새 요청을 시작하지 않음


@pytest.mark.parametrize(
    ("remaining", "connect_timeout", "expected"),
    [
        (10, 5, True),
        (5, 5, False),  # 연결 시간 초과값과 같으면 하지 않음
        (0, 0, False),  # 남은 시간이 없으면 연결 시간 초과값과 상관없이 하지 않음
        (-1, 0, False),
    ],
)
def test_can_retry(monkeypatch, remaining, connect_timeout, expected):
    monkeypatch.setattr(vlm_client, "CONNECT_TIMEOUT", connect_timeout)
    assert vlm_client._can_retry(remaining) is expected


async def test_close_clears_client_even_if_closing_fails(monkeypatch):
    vlm_client.start(_settings())
    client, _ = vlm_client.get_client()

    async def broken_close():
        raise RuntimeError("닫기 실패")

    monkeypatch.setattr(client, "aclose", broken_close)
    with pytest.raises(RuntimeError, match="닫기 실패"):
        await vlm_client.close()
    with pytest.raises(RuntimeError, match="시작되지 않음"):
        vlm_client.get_client()


def test_deadline_warning_only_when_close_to_backend_timeout():
    assert vlm_client.deadline_warning(_settings(vlm_deadline_seconds=20)) is None
    assert "30초" in vlm_client.deadline_warning(_settings(vlm_deadline_seconds=25))
    assert vlm_client.deadline_warning(_settings(verify_backend="gemini", vlm_deadline_seconds=60)) is None


async def test_lifespan_logs_deadline_warning(monkeypatch, caplog):
    for key, value in {**SELF_OK, "vlm_deadline_seconds": 25}.items():
        monkeypatch.setattr(settings, key, value)
    monkeypatch.setattr(vlm_client, "start", lambda s: None)
    with caplog.at_level("WARNING", logger="uvicorn.error"):
        async with lifespan(app):
            pass
    assert "VLM_DEADLINE_SECONDS=25초" in caplog.text


# 5-2. 부르는 쪽 마감(deadline)·요청별 기록(trace) — 폴백이 Gemini 몫을 남기려고 쓴다

def _now() -> float:
    import asyncio

    return asyncio.get_running_loop().time()


async def test_earlier_caller_deadline_shortens_the_wait(fake_vlm):
    server = fake_vlm(slow(5, ok))  # 설정 마감은 60초
    started = _now()
    with pytest.raises(HTTPException) as exc:
        await vlm_client.score(b"jpeg", PLACE, MISSION, deadline=started + 0.3)
    assert exc.value.status_code == 504
    assert _now() - started < 2
    assert len(server.requests) == 1


async def test_later_caller_deadline_does_not_extend_the_setting(fake_vlm):
    fake_vlm(slow(5, ok), vlm_deadline_seconds=0.3)
    started = _now()
    with pytest.raises(HTTPException) as exc:
        await vlm_client.score(b"jpeg", PLACE, MISSION, deadline=started + 60)
    assert exc.value.status_code == 504
    assert _now() - started < 2


async def test_passed_caller_deadline_sends_nothing(fake_vlm):
    server = fake_vlm(ok)
    trace: dict = {}
    with pytest.raises(HTTPException) as exc:
        await vlm_client.score(b"jpeg", PLACE, MISSION, deadline=_now() - 1, trace=trace)
    assert exc.value.status_code == 504
    assert server.requests == []
    assert trace["attempts"] == 0


async def test_outer_deadline_is_recorded_apart_from_read_timeout(fake_vlm, monkeypatch):
    # 안전망이 끊은 경우는 원인이 비지 않고 DeadlineExceeded로 남는다(httpx ReadTimeout과 구분)
    async def stuck(request):
        await anyio.sleep(5)  # 읽기 시간 초과보다 먼저 안전망이 끊게 httpx 시간 초과를 우회한다

    monkeypatch.setattr(vlm_client, "_attempt", lambda *a, **k: stuck(None))
    fake_vlm(ok)
    trace: dict = {}
    with pytest.raises(HTTPException) as exc:
        await vlm_client.score(b"jpeg", PLACE, MISSION, deadline=_now() + 0.2, trace=trace)
    assert exc.value.status_code == 504
    assert trace["last_error"] == "DeadlineExceeded"


async def test_trace_is_filled_per_call_and_does_not_leak(fake_vlm):
    fake_vlm(status_(500), raise_(httpx.ConnectError), ok)
    first: dict = {}
    with pytest.raises(HTTPException):
        await vlm_client.score(b"jpeg", PLACE, MISSION, trace=first)
    assert (first["attempts"], first["upstream_status"], first["last_error"]) == (2, None, "ConnectError")
    assert first["elapsed_sec"] >= 0
    second: dict = {"attempts": 9, "last_error": "old"}  # 넘긴 dict의 예전 값은 지운다
    assert await vlm_client.score(b"jpeg", PLACE, MISSION, trace=second) == VLM_RESULT
    assert (second["attempts"], second["upstream_status"], second["last_error"]) == (1, 200, None)
    assert "test-key" not in str(first) + str(second) and "vlm.example.com" not in str(first) + str(second)


async def test_trace_records_time_even_when_cancelled(fake_vlm):
    import asyncio

    fake_vlm(slow(5, ok))
    trace: dict = {}
    task = asyncio.create_task(vlm_client.score(b"jpeg", PLACE, MISSION, trace=trace))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):  # 취소는 504 등으로 바꾸지 않고 그대로 올라간다
        await task
    assert trace["attempts"] == 1 and trace["elapsed_sec"] >= 0


async def test_score_without_start_raises():
    # 시작 단계를 거치지 않았으면 몰래 클라이언트를 만들지 않는다
    with pytest.raises(RuntimeError):
        await _score()


# 6. 요청 형식 — 측정 스크립트(judge_local)와 같아야 dev 측정 결과가 서비스에 그대로 맞는다

async def test_request_matches_measurement_script(fake_vlm, monkeypatch):
    buf = io.BytesIO()
    Image.new("RGB", (1600, 1200), "white").save(buf, format="JPEG")
    raw = buf.getvalue()

    # judge_local이 보내는 요청을 가짜 서버로 받아 둔다
    measured = []
    real_client = httpx.AsyncClient

    def capture(request):
        measured.append(json.loads(request.content))
        return ok(request)

    monkeypatch.setattr(
        httpx, "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(capture), **kwargs),
    )
    case = {"photo": "p.jpg", "kind": "positive", "target": PLACE, "mission": MISSION}
    await judge_local(case, raw, "http://measure/v1", "qwen")
    monkeypatch.undo()

    # 서비스는 4단계에서 줄인 사진을 받는다
    server = fake_vlm(ok)
    await vlm_client.score(normalize_image(raw), PLACE, MISSION)
    sent = json.loads(server.requests[0].content)

    assert sent.pop("max_tokens") == 512  # 측정에는 없던 값. 답이 141토큰을 넘지 않아 결과에 영향 없음
    assert sent == measured[0]
    assert server.requests[0].url == "https://vlm.example.com/v1/chat/completions"
    assert server.requests[0].headers["Authorization"] == "Bearer test-key"


async def test_question_request_matches_measurement_script(fake_vlm, monkeypatch):
    # 객관식도 측정 스크립트(--format question)와 같은 요청이어야 dev 측정 결과가 서비스에 맞는다
    buf = io.BytesIO()
    Image.new("RGB", (1600, 1200), "white").save(buf, format="JPEG")
    raw = buf.getvalue()
    answer_json = ANSWER.model_dump_json()
    measured = []
    real_client = httpx.AsyncClient

    def capture(request):
        measured.append(json.loads(request.content))
        return _chat(answer_json)

    monkeypatch.setattr(
        httpx, "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(capture), **kwargs),
    )
    case = {"photo": "p.jpg", "kind": "positive", "target": PLACE, "mission": MISSION}
    await judge_local(case, raw, "http://measure/v1", "qwen", fmt="question")
    monkeypatch.undo()

    server = fake_vlm(lambda r: _chat(answer_json), vlm_format="question")
    await vlm_client.score(normalize_image(raw), PLACE, MISSION)
    sent = json.loads(server.requests[0].content)
    assert sent.pop("max_tokens") == 512
    assert sent == measured[0]


# 7. 관측 — Sentry에 시도 횟수·걸린 시간·상대 서버 상태를 남기고 키·주소는 남기지 않는다

@pytest.mark.parametrize(
    ("outcomes", "expected"),
    [
        pytest.param((status_(500), ok), {"upstream_status": 200, "last_error": None}, id="retry-then-ok"),
        # 두 번째가 연결 실패면 첫 시도의 500이 남지 않아야 한다
        pytest.param((status_(500), raise_(httpx.ConnectError)),
                     {"upstream_status": None, "last_error": "ConnectError"}, id="500-then-connect-error"),
        pytest.param((status_(500), lambda r: httpx.Response(200, content=b"not json")),
                     {"upstream_status": 200, "last_error": "JSONDecodeError"}, id="500-then-broken-json"),
    ],
)
async def test_sentry_context_describes_last_attempt_without_secrets(fake_vlm, monkeypatch, outcomes, expected):
    recorded = {}
    monkeypatch.setattr(sentry_sdk, "set_context", lambda name, value: recorded.update({name: value}))
    fake_vlm(*outcomes)
    try:
        await _score()
    except HTTPException:
        pass

    context = recorded["vlm"]
    assert context["attempts"] == 2
    assert {k: context[k] for k in expected} == expected
    assert context["elapsed_sec"] >= 0
    assert "test-key" not in str(context) and "vlm.example.com" not in str(context)


async def test_real_sentry_breadcrumb_hides_vlm_address(fake_vlm, monkeypatch, caplog):
    # Sentry가 httpx 호출을 자동으로 기록하는 기능을 실제로 켜고, 서비스와 같은 정리 함수를 거친 결과를 본다
    # httpx 로거를 INFO로 올려 요청 문장이 message로 들어오는 경우까지 본다 (PR #68 리뷰)
    monkeypatch.setattr(settings, "vlm_base_url", SELF_OK["vlm_base_url"])
    caplog.set_level("INFO", logger="httpx")
    kept = []

    def keep(crumb, hint):
        kept.append(sentry.scrub_breadcrumb(crumb, hint))
        return kept[-1]

    # 가짜 DSN이라 밖으로 보내지 않는다. 이 테스트 안에서만 켜고 끈다
    sentry_sdk.init(dsn="https://public@example.invalid/1", before_breadcrumb=keep,
                    default_integrations=False, integrations=[HttpxIntegration(), LoggingIntegration(level=logging.INFO, event_level=None)])
    try:
        fake_vlm(ok)
        await _score()
    finally:
        sentry_sdk.get_client().close()
        sentry_sdk.get_global_scope().set_client(None)

    http_crumbs = [c for c in kept if c.get("category") == "httplib"]
    log_crumbs = [c for c in kept if c.get("category") == "httpx"]
    assert http_crumbs and log_crumbs, "호출 기록과 로그 기록이 둘 다 남아야 이 테스트가 의미 있음"
    assert all(c["data"]["url"] == "[Filtered]" for c in http_crumbs)
    assert all("[Filtered]" in c["message"] for c in log_crumbs)
    assert "vlm.example.com" not in str(kept)
