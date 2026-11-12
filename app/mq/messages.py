"""요청 메시지를 읽고, 결과·재시도·DLQ로 보낼 메시지를 만든다 (계약 문서 4절)."""

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from aio_pika import DeliveryMode, Message

from app.core.time import SERVICE_TIMEZONE
from app.mq.errors import INVALID_MESSAGE, UNSUPPORTED_SCHEMA_VERSION, AiJobError
from app.mq.topology import (
    DLX_EXCHANGE,
    RESULT_EXCHANGE,
    RESULT_ROUTING_KEY,
    RETRY_EXCHANGE,
    JobQueue,
    JobType,
)

SCHEMA_VERSION = 1
APP_ID = "planit-ai"
RETRY_COUNT_HEADER = "x-retry-count"
ERROR_KIND_HEADER = "x-error-kind"
ERROR_MESSAGE_HEADER = "x-error-message"
FAILED_AT_HEADER = "x-failed-at"
MAX_ERROR_MESSAGE_LENGTH = 500

# 다시 보낼 때 옮기지 않는 헤더. 실패 기록은 새로 붙이고, x-death 계열은 RabbitMQ가 관리한다
_DROPPED_HEADERS = (ERROR_KIND_HEADER, ERROR_MESSAGE_HEADER, FAILED_AT_HEADER, "x-death")
_DROPPED_HEADER_PREFIXES = ("x-first-death-", "x-last-death-")


class IncomingMessage(Protocol):
    """aio-pika 수신 메시지에서 쓰는 속성만 모은 모양. 테스트에서는 같은 속성을 가진 대역을 쓴다."""

    body: bytes
    headers: dict[str, Any]
    message_id: str | None
    correlation_id: str | None
    type: str | None
    content_type: str | None
    app_id: str | None
    timestamp: datetime | None


@dataclass(frozen=True)
class RequestEnvelope:
    schema_version: int
    job_id: str
    job_type: JobType
    deadline_at: datetime
    request: dict[str, Any]


@dataclass(frozen=True)
class Outgoing:
    exchange: str
    routing_key: str
    message: Message
    # 지표·로그용 처리 결과: succeeded, failed_result, retry, dlq
    outcome: str


def parse_envelope(body: bytes, queue: JobQueue) -> RequestEnvelope:
    """
    계약 문서 4.3절 요청 본문을 읽는다. 읽을 수 없거나 필수 값이 없으면 INVALID_MESSAGE,
    schemaVersion이 1이 아니면 UNSUPPORTED_SCHEMA_VERSION으로 AiJobError를 던진다.
    """
    try:
        raw = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise AiJobError(INVALID_MESSAGE, False, "본문을 JSON으로 읽지 못함") from e
    if not isinstance(raw, dict):
        raise AiJobError(INVALID_MESSAGE, False, "본문이 JSON 객체가 아님")

    schema_version = raw.get("schemaVersion")
    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        raise AiJobError(INVALID_MESSAGE, False, "schemaVersion 값이 없음")
    if schema_version != SCHEMA_VERSION:
        raise AiJobError(UNSUPPORTED_SCHEMA_VERSION, False, f"지원하지 않는 schemaVersion={schema_version}")

    job_id = raw.get("jobId")
    if not isinstance(job_id, str) or not job_id.isdigit():
        raise AiJobError(INVALID_MESSAGE, False, "jobId가 숫자 문자열이 아님")
    if raw.get("jobType") != queue.job_type.value:
        raise AiJobError(INVALID_MESSAGE, False, f"이 큐의 작업 종류가 아님 jobType={raw.get('jobType')}")
    deadline_at = _parse_deadline(raw.get("deadlineAt"))
    request = raw.get("request")
    if not isinstance(request, dict):
        raise AiJobError(INVALID_MESSAGE, False, "request가 JSON 객체가 아님")
    return RequestEnvelope(schema_version, job_id, queue.job_type, deadline_at, request)


def retry_count_of(message: IncomingMessage) -> int:
    """x-retry-count 헤더. 없거나 숫자가 아니면 0으로 본다."""
    value = (message.headers or {}).get(RETRY_COUNT_HEADER)
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0


def to_result(
    message: IncomingMessage,
    envelope: RequestEnvelope,
    *,
    result: dict[str, Any] | None,
    error_kind: str | None = None,
) -> Outgoing:
    """계약 문서 4.4절 결과 메시지. error_kind가 있으면 status=FAILED 결과다."""
    succeeded = error_kind is None
    body = {
        "schemaVersion": SCHEMA_VERSION,
        "jobId": envelope.job_id,
        "jobType": envelope.job_type.value,
        "requestMessageId": message.message_id,
        "status": "SUCCEEDED" if succeeded else "FAILED",
        "errorKind": error_kind,
        "result": result if succeeded else None,
    }
    outgoing = Message(
        json.dumps(body, ensure_ascii=False).encode("utf-8"),
        content_type="application/json",
        delivery_mode=DeliveryMode.PERSISTENT,
        message_id=str(uuid.uuid4()),
        correlation_id=envelope.job_id,
        type=f"{envelope.job_type.value}_RESULT",
        app_id=APP_ID,
        timestamp=_now(),
        headers={RETRY_COUNT_HEADER: retry_count_of(message)},
    )
    return Outgoing(RESULT_EXCHANGE, RESULT_ROUTING_KEY, outgoing, "succeeded" if succeeded else "failed_result")


def to_retry(message: IncomingMessage, queue: JobQueue, next_count: int) -> Outgoing:
    """요청 본문과 속성을 그대로 두고 x-retry-count만 올려 다음 단계 Retry Queue로 보낸다."""
    headers = _copied_headers(message)
    headers[RETRY_COUNT_HEADER] = next_count
    return Outgoing(RETRY_EXCHANGE, queue.retry_routing_key(next_count), _copy(message, headers), "retry")


def to_dlq(message: IncomingMessage, queue: JobQueue, error: AiJobError) -> Outgoing:
    """
    요청 본문과 속성을 그대로 두고 실패 원인 헤더를 붙여 요청 DLQ로 보낸다.
    본문을 읽지 못한 메시지도 correlation_id·type을 그대로 옮겨 백엔드가 작업을 찾을 수 있게 한다.
    routing key는 본문이 아니라 소비한 큐로 정한다.
    """
    headers = _copied_headers(message)
    headers[ERROR_KIND_HEADER] = error.kind
    headers[ERROR_MESSAGE_HEADER] = error.message[:MAX_ERROR_MESSAGE_LENGTH]
    headers[FAILED_AT_HEADER] = _now().isoformat()
    return Outgoing(DLX_EXCHANGE, queue.dlq_routing_key, _copy(message, headers), "dlq")


def _copy(message: IncomingMessage, headers: dict[str, Any]) -> Message:
    return Message(
        message.body,
        content_type=message.content_type or "application/json",
        delivery_mode=DeliveryMode.PERSISTENT,
        message_id=message.message_id,
        correlation_id=message.correlation_id,
        type=message.type,
        app_id=message.app_id,
        timestamp=message.timestamp,
        headers=headers,
    )


def _copied_headers(message: IncomingMessage) -> dict[str, Any]:
    return {
        name: value
        for name, value in (message.headers or {}).items()
        if name not in _DROPPED_HEADERS and not name.startswith(_DROPPED_HEADER_PREFIXES)
    }


def _parse_deadline(value: Any) -> datetime:
    if not isinstance(value, str):
        raise AiJobError(INVALID_MESSAGE, False, "deadlineAt 값이 없음")
    try:
        deadline = datetime.fromisoformat(value)
    except ValueError as e:
        raise AiJobError(INVALID_MESSAGE, False, f"deadlineAt 형식이 틀림 deadlineAt={value}") from e
    if deadline.tzinfo is None:
        raise AiJobError(INVALID_MESSAGE, False, f"deadlineAt에 시간대가 없음 deadlineAt={value}")
    return deadline


def _now() -> datetime:
    """서울 시간, 초 단위. x-failed-at은 +09:00으로 쓴다."""
    return datetime.now(SERVICE_TIMEZONE).replace(microsecond=0)
