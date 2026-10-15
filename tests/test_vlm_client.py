import inspect
import io
import json
import time

import anyio
import httpx
import pytest
import sentry_sdk
from fastapi import HTTPException
from PIL import Image

from app.core.config import Settings, settings
from app.main import app, lifespan
from app.photomissions import vlm_client
from app.photomissions.pipeline_verify import normalize_image
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
    assert summary == "verify 판정 백엔드: self (모델 qwen, 마감 60초, 기준 성공 80 / 재시도 40 / 랜드마크 50)"
    assert "secret-key" not in summary and "vlm.example.com" not in summary


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


RETRYABLE = [
    pytest.param(status_(500), id="500"),
    pytest.param(status_(503), id="503"),
    pytest.param(raise_(httpx.ConnectError), id="connect-error"),
    pytest.param(raise_(httpx.ConnectTimeout), id="connect-timeout"),
    pytest.param(raise_(httpx.RemoteProtocolError), id="connection-dropped"),
    pytest.param(lambda r: httpx.Response(200, content=b"not json"), id="broken-json"),
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


# 7. 관측 — Sentry에 시도 횟수·걸린 시간·상대 서버 상태를 남기고 키·주소는 남기지 않는다

async def test_sentry_context_records_attempts_without_secrets(fake_vlm, monkeypatch):
    recorded = {}
    monkeypatch.setattr(sentry_sdk, "set_context", lambda name, value: recorded.update({name: value}))
    fake_vlm(status_(500), ok)
    await _score()

    context = recorded["vlm"]
    assert context["attempts"] == 2
    assert context["upstream_status"] == 200
    assert context["elapsed_sec"] >= 0
    assert "test-key" not in str(context) and "vlm.example.com" not in str(context)
