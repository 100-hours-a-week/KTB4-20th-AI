"""RabbitMQ AI Worker의 메시지 처리 규칙 (계약 문서 4~5절). 브로커 없이 대역으로 확인한다."""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import httpx
import pytest
from aio_pika.exceptions import DeliveryError
from fastapi import HTTPException, status
from pamqp.commands import Basic
from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.core.time import SERVICE_TIMEZONE
from app.mq import errors
from app.mq.messages import (
    ERROR_KIND_HEADER,
    ERROR_MESSAGE_HEADER,
    FAILED_AT_HEADER,
    RETRY_COUNT_HEADER,
)
from app.mq.topology import JOB_QUEUES, JobType
from app.mq.worker import QueueWorker, process
from app.trips.fifo_gate import QueueFull

pytestmark = pytest.mark.anyio

SCHEDULE = JOB_QUEUES[JobType.SCHEDULE_GENERATE]
JUDGE = JOB_QUEUES[JobType.PHOTO_JUDGE]
NOW = datetime(2026, 11, 12, 12, 0, tzinfo=SERVICE_TIMEZONE)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@dataclass
class FakeMessage:
    body: bytes
    headers: dict[str, Any] = field(default_factory=dict)
    message_id: str | None = "request-message-1"
    correlation_id: str | None = "42"
    type: str | None = "SCHEDULE_GENERATE"
    content_type: str | None = "application/json"
    app_id: str | None = "planit-backend"
    timestamp: datetime | None = None
    redelivered: bool = False
    acked: bool = False

    async def ack(self) -> None:
        self.acked = True


def envelope(job_type: JobType = JobType.SCHEDULE_GENERATE, **overrides: Any) -> bytes:
    body = {
        "schemaVersion": 1,
        "jobId": "42",
        "jobType": job_type.value,
        "deadlineAt": (NOW + timedelta(minutes=28)).isoformat(),
        "request": {"region": "경주"},
    }
    body.update(overrides)
    return json.dumps(body).encode()


def message(retry_count: int = 0, **overrides: Any) -> FakeMessage:
    return FakeMessage(envelope(**overrides), headers={RETRY_COUNT_HEADER: retry_count})


def runner(result: Any = None, error: BaseException | None = None):
    calls: list[dict[str, Any]] = []

    async def run(queue, request):
        calls.append(request)
        if error is not None:
            raise error
        return result

    run.calls = calls  # type: ignore[attr-defined]
    return run


async def handle(incoming: FakeMessage, run, queue=SCHEDULE):
    return await process(incoming, queue, run, now=lambda: NOW)


def body_of(outgoing) -> dict[str, Any]:
    return json.loads(outgoing.message.body)


# --- 결과 ---


async def test_success_publishes_result_with_request_ids():
    incoming = message(retry_count=2)

    outgoing = await handle(incoming, runner({"status_code": 200, "data": {"places": []}}))

    assert (outgoing.exchange, outgoing.routing_key) == ("ai.result", "ai-result")
    assert body_of(outgoing) == {
        "schemaVersion": 1,
        "jobId": "42",
        "jobType": "SCHEDULE_GENERATE",
        "requestMessageId": "request-message-1",
        "status": "SUCCEEDED",
        "errorKind": None,
        "result": {"status_code": 200, "data": {"places": []}},
    }
    properties = outgoing.message
    assert properties.correlation_id == "42"
    assert properties.type == "SCHEDULE_GENERATE_RESULT"
    assert properties.app_id == "planit-ai"
    assert properties.message_id != "request-message-1"
    assert properties.headers[RETRY_COUNT_HEADER] == 2


async def test_expired_deadline_skips_ai_and_publish():
    run = runner({})
    incoming = message(deadlineAt=(NOW - timedelta(seconds=1)).isoformat())

    assert await handle(incoming, run) is None
    assert run.calls == []


# --- 재시도와 DLQ ---


@pytest.mark.parametrize("retry_count, stage", [(0, 1), (1, 2), (2, 3)])
async def test_retryable_failure_goes_to_next_retry_queue(retry_count, stage):
    incoming = message(retry_count=retry_count)
    incoming.headers["x-death"] = [{"reason": "expired"}]

    outgoing = await handle(incoming, runner(error=TimeoutError()))

    assert (outgoing.exchange, outgoing.routing_key) == ("ai.retry", f"ai-schedule-generation.retry.{stage}")
    assert outgoing.message.headers[RETRY_COUNT_HEADER] == stage
    assert "x-death" not in outgoing.message.headers
    assert outgoing.message.body == incoming.body
    assert outgoing.message.correlation_id == "42"


async def test_retryable_failure_after_three_retries_goes_to_dlq():
    outgoing = await handle(message(retry_count=3), runner(error=TimeoutError()))

    assert (outgoing.exchange, outgoing.routing_key) == ("ai.dlx", "ai-schedule-generation.dlq")
    headers = outgoing.message.headers
    assert headers[ERROR_KIND_HEADER] == "AI_TIMEOUT"
    assert headers[ERROR_MESSAGE_HEADER]
    assert datetime.fromisoformat(headers[FAILED_AT_HEADER]).utcoffset() == timedelta(hours=9)


async def test_non_retryable_failure_goes_straight_to_dlq():
    class Strict(BaseModel):
        region: int

    with pytest.raises(ValidationError) as invalid:
        Strict.model_validate({"region": "경주"})

    outgoing = await handle(message(retry_count=0), runner(error=invalid.value))

    assert outgoing.routing_key == "ai-schedule-generation.dlq"
    assert outgoing.message.headers[ERROR_KIND_HEADER] == "INVALID_REQUEST"


async def test_reprocess_message_failure_sends_failed_result():
    outgoing = await handle(message(retry_count=4), runner(error=TimeoutError()))

    assert outgoing.exchange == "ai.result"
    assert body_of(outgoing)["status"] == "FAILED"
    assert body_of(outgoing)["errorKind"] == "AI_TIMEOUT"
    assert body_of(outgoing)["result"] is None


@pytest.mark.parametrize(
    "body, kind",
    [
        (b"not json", "INVALID_MESSAGE"),
        (envelope(jobId=None), "INVALID_MESSAGE"),
        (envelope(jobType="PHOTO_JUDGE"), "INVALID_MESSAGE"),
        (envelope(deadlineAt="2026-11-12 12:00"), "INVALID_MESSAGE"),
        (envelope(schemaVersion=2), "UNSUPPORTED_SCHEMA_VERSION"),
    ],
)
async def test_unreadable_message_goes_to_dlq_with_original_ids(body, kind):
    run = runner({})
    incoming = FakeMessage(body)

    outgoing = await handle(incoming, run)

    assert run.calls == []
    assert outgoing.routing_key == "ai-schedule-generation.dlq"
    assert outgoing.message.headers[ERROR_KIND_HEADER] == kind
    assert outgoing.message.correlation_id == "42"
    assert outgoing.message.type == "SCHEDULE_GENERATE"


# --- 오류 분류 ---


def http_error(code: int, detail: str = "실패", cause: BaseException | None = None) -> HTTPException:
    error = HTTPException(status_code=code, detail=detail)
    error.__cause__ = cause
    return error


@pytest.mark.parametrize(
    "error, kind, retryable",
    [
        (http_error(status.HTTP_504_GATEWAY_TIMEOUT), "AI_TIMEOUT", True),
        (http_error(status.HTTP_429_TOO_MANY_REQUESTS), "AI_UNAVAILABLE", True),
        (http_error(status.HTTP_502_BAD_GATEWAY), "AI_UNAVAILABLE", True),
        (http_error(status.HTTP_503_SERVICE_UNAVAILABLE), "AI_UNAVAILABLE", True),
        (QueueFull(), "AI_UNAVAILABLE", True),
        (RuntimeError("db down"), "AI_UNAVAILABLE", True),
        (http_error(status.HTTP_422_UNPROCESSABLE_CONTENT), "INVALID_REQUEST", False),
    ],
)
def test_classify_schedule_errors(error, kind, retryable):
    classified = errors.classify(error, JobType.SCHEDULE_GENERATE)

    assert (classified.kind, classified.retryable) == (kind, retryable)


def image_status_error(code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://bucket.example/photo.jpg")
    return httpx.HTTPStatusError("실패", request=request, response=httpx.Response(code, request=request))


@pytest.mark.parametrize(
    "error, kind, retryable",
    [
        (http_error(502, "이미지 다운로드 실패", image_status_error(404)), "IMAGE_NOT_FOUND", False),
        (http_error(502, "이미지 다운로드 실패", image_status_error(403)), "IMAGE_NOT_FOUND", False),
        (http_error(502, "이미지 다운로드 실패", image_status_error(500)), "AI_UNAVAILABLE", True),
        (http_error(422, "이미지 파일 크기가 제한을 초과했습니다"), "UNSUPPORTED_IMAGE", False),
        (http_error(422, "허용되지 않은 이미지 호스트입니다"), "INVALID_REQUEST", False),
    ],
)
def test_classify_judge_image_errors(error, kind, retryable):
    classified = errors.classify(error, JobType.PHOTO_JUDGE)

    assert (classified.kind, classified.retryable) == (kind, retryable)


# --- 발행과 ACK ---


class FakeExchange:
    def __init__(self, outcomes: list[Any]):
        self.outcomes = outcomes
        self.published = 0

    async def publish(self, message, routing_key, mandatory, timeout):
        assert mandatory is True
        outcome = self.outcomes[min(self.published, len(self.outcomes) - 1)]
        self.published += 1
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def worker_with(exchange: FakeExchange, monkeypatch) -> QueueWorker:
    settings = Settings(mq_republish_max_retries=3, mq_republish_backoff_base_seconds=0.001)
    worker = QueueWorker(connection=None, queue=SCHEDULE, settings=settings)  # type: ignore[arg-type]
    worker._exchanges = {"ai.result": exchange, "ai.retry": exchange, "ai.dlx": exchange}
    restarts: list[bool] = []

    async def restart():
        restarts.append(True)

    monkeypatch.setattr(worker, "_restart", restart)
    worker.restarts = restarts  # type: ignore[attr-defined]
    return worker


async def outgoing_result():
    return await handle(message(), runner({"status_code": 200, "data": {}}))


async def test_ack_only_after_confirm(monkeypatch):
    exchange = FakeExchange([Basic.Ack()])
    worker = worker_with(exchange, monkeypatch)
    incoming = message()

    await worker._publish_and_ack(incoming, await outgoing_result())

    assert incoming.acked
    assert exchange.published == 1


async def test_republishes_then_acks_without_calling_ai_again(monkeypatch):
    exchange = FakeExchange([Basic.Nack(), TimeoutError(), Basic.Ack()])
    worker = worker_with(exchange, monkeypatch)
    incoming = message()

    await worker._publish_and_ack(incoming, await outgoing_result())

    assert incoming.acked
    assert exchange.published == 3
    assert worker.restarts == []


async def test_gives_up_after_republish_limit_and_restarts_channel(monkeypatch):
    exchange = FakeExchange([DeliveryError(None, None)])
    worker = worker_with(exchange, monkeypatch)
    incoming = message()

    await worker._publish_and_ack(incoming, await outgoing_result())
    await asyncio.sleep(0)

    assert not incoming.acked
    assert exchange.published == 4  # 최초 1회 + 재발행 3회
    assert worker.restarts == [True]


async def test_connection_error_leaves_message_for_redelivery(monkeypatch):
    exchange = FakeExchange([ConnectionResetError()])
    worker = worker_with(exchange, monkeypatch)
    incoming = message()

    await worker._publish_and_ack(incoming, await outgoing_result())

    assert not incoming.acked
    assert exchange.published == 1
    assert worker.restarts == []
