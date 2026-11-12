"""
작업 큐를 소비해 AI를 부르고, 결과·재시도·DLQ로 발행한 뒤 요청을 ACK한다.

Worker 1개 = 채널 1개 = Consumer 1개이고 prefetch는 1이다. 다음 목적지 발행의 confirm을 받은 뒤에만 요청을 ACK하므로,
발행하지 못하거나 프로세스가 죽으면 요청은 작업 큐로 돌아가 다시 처리된다. 같은 작업이 두 번 처리될 수 있지만
백엔드가 작업 상태 조건부 UPDATE로 한 번만 반영한다.
"""

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

import aio_pika
from aio_pika.abc import (
    AbstractChannel,
    AbstractExchange,
    AbstractIncomingMessage,
    AbstractRobustConnection,
)
from aio_pika.exceptions import AMQPError, DeliveryError
from pamqp.commands import Basic

from app.core.config import Settings
from app.core.time import SERVICE_TIMEZONE
from app.mq import metrics
from app.mq.errors import AiJobError, classify
from app.mq.handlers import run_job
from app.mq.messages import (
    IncomingMessage,
    Outgoing,
    RequestEnvelope,
    parse_envelope,
    retry_count_of,
    to_dlq,
    to_result,
    to_retry,
)
from app.mq.topology import (
    JOB_QUEUES,
    MAX_AI_RETRIES,
    PUBLISH_EXCHANGES,
    REPROCESS_RETRY_COUNT,
    JobQueue,
    JobType,
)

logger = logging.getLogger(__name__)

JobRunner = Callable[[JobQueue, dict[str, Any]], Awaitable[dict[str, Any]]]


class PublishNacked(Exception):
    """브로커가 발행을 받지 않음(nack)."""


async def process(
    message: IncomingMessage,
    queue: JobQueue,
    run: JobRunner = run_job,
    now: Callable[[], datetime] = lambda: datetime.now(SERVICE_TIMEZONE),
) -> Outgoing | None:
    """
    요청 하나를 처리하고 다음에 보낼 메시지를 돌려준다. None이면 deadlineAt이 지나 AI를 부르지 않은 것이고, 발행 없이 ACK한다.
    그 작업은 백엔드 Stale Job Checker가 실패로 끝낸다.
    """
    retry_count = retry_count_of(message)
    try:
        envelope = parse_envelope(message.body, queue)
    except AiJobError as error:
        return to_dlq(message, queue, error)
    if now() > envelope.deadline_at:
        return None

    started = time.monotonic()
    try:
        result = await run(queue, envelope.request)
    except Exception as error:
        job_error = classify(error, queue.job_type)
        logger.warning(
            "AI 작업 실패 jobType=%s jobId=%s retryCount=%d errorKind=%s",
            queue.job_type, envelope.job_id, retry_count, job_error.kind, exc_info=error,
        )
        return route_failure(message, queue, envelope, retry_count, job_error)
    finally:
        metrics.PROCESSING_SECONDS.labels(queue.job_type.value).observe(time.monotonic() - started)
    return to_result(message, envelope, result=result)


def route_failure(
    message: IncomingMessage,
    queue: JobQueue,
    envelope: RequestEnvelope,
    retry_count: int,
    error: AiJobError,
) -> Outgoing:
    """계약 문서 5절. 05시 재처리 메시지(x-retry-count=4)는 오류 종류와 상관없이 FAILED 결과로 끝낸다."""
    if retry_count >= REPROCESS_RETRY_COUNT:
        return to_result(message, envelope, result=None, error_kind=error.kind)
    if error.retryable and retry_count < MAX_AI_RETRIES:
        return to_retry(message, queue, retry_count + 1)
    return to_dlq(message, queue, error)


class QueueWorker:
    """작업 큐 하나를 소비하는 Worker 1개."""

    def __init__(
        self,
        connection: AbstractRobustConnection,
        queue: JobQueue,
        settings: Settings,
        run: JobRunner = run_job,
    ):
        self._connection = connection
        self._queue = queue
        self._settings = settings
        self._run = run
        self._channel: AbstractChannel | None = None
        self._exchanges: dict[str, AbstractExchange] = {}
        self._amqp_queue: aio_pika.abc.AbstractQueue | None = None
        self._consumer_tag: str | None = None
        self._inflight = 0
        self._idle = asyncio.Event()
        self._idle.set()
        self._restarting = False

    async def start(self) -> None:
        """채널을 열고 Consumer를 등록한다. 큐나 Exchange가 없으면(백엔드가 아직 선언하지 않음) 예외로 시작을 멈춘다."""
        channel = await self._connection.channel(publisher_confirms=True, on_return_raises=True)
        await channel.set_qos(prefetch_count=1)
        self._amqp_queue = await channel.get_queue(self._queue.task_queue, ensure=True)
        self._exchanges = {name: await channel.get_exchange(name, ensure=True) for name in PUBLISH_EXCHANGES}
        self._channel = channel
        self._consumer_tag = await self._amqp_queue.consume(self._on_message, no_ack=False)
        metrics.ACTIVE_WORKERS.labels(self._queue.job_type.value).inc()

    async def stop(self, grace_seconds: float) -> None:
        """새 메시지를 받지 않게 한 뒤, 처리 중인 메시지가 끝날 때까지 기다렸다가 채널을 닫는다."""
        await self._cancel_consumer()
        try:
            await asyncio.wait_for(self._idle.wait(), grace_seconds)
        except TimeoutError:
            logger.warning("처리 중인 메시지를 기다리지 못하고 Worker를 닫음 jobType=%s", self._queue.job_type)
        await self._close_channel()

    async def _on_message(self, message: AbstractIncomingMessage) -> None:
        self._inflight += 1
        self._idle.clear()
        try:
            if message.redelivered:
                metrics.REDELIVERED.labels(self._queue.job_type.value).inc()
            outgoing = await process(message, self._queue, self._run)
            if outgoing is None:
                metrics.MESSAGES.labels(self._queue.job_type.value, "deadline_skipped").inc()
                await message.ack()
                return
            await self._publish_and_ack(message, outgoing)
        finally:
            self._inflight -= 1
            if self._inflight == 0:
                self._idle.set()

    async def _publish_and_ack(self, message: AbstractIncomingMessage, outgoing: Outgoing) -> None:
        """
        발행 confirm을 받은 뒤 요청을 ACK한다. nack·return·confirm 시간 초과면 AI를 다시 부르지 않고 같은 메시지를
        최대 mq_republish_max_retries번 다시 발행한다. 그래도 실패하면 채널을 닫아 요청을 작업 큐로 돌려보낸다.
        연결이 끊겼으면 ACK할 수 없으므로 재발행하지 않고 재전달에 맡긴다.
        """
        job_type = self._queue.job_type.value
        max_retries = self._settings.mq_republish_max_retries
        for attempt in range(max_retries + 1):
            try:
                await self._publish(outgoing)
                await message.ack()
                metrics.MESSAGES.labels(job_type, outgoing.outcome).inc()
                return
            except (DeliveryError, PublishNacked, TimeoutError) as error:
                metrics.PUBLISH_FAILURES.labels(job_type, _failure_kind(error)).inc()
                logger.warning(
                    "발행 실패 jobType=%s correlationId=%s exchange=%s attempt=%d",
                    job_type, message.correlation_id, outgoing.exchange, attempt + 1, exc_info=error,
                )
                if attempt == max_retries:
                    break
                backoff = self._settings.mq_republish_backoff_base_seconds * (2 ** attempt)
                await asyncio.sleep(backoff + random.uniform(0, backoff / 2))
            except (AMQPError, ConnectionError) as error:
                metrics.PUBLISH_FAILURES.labels(job_type, "connection").inc()
                logger.warning("연결이 끊겨 재전달에 맡김 jobType=%s correlationId=%s", job_type, message.correlation_id,
                               exc_info=error)
                return
        # 이 콜백 안에서 채널을 닫으면 자기 자신을 기다리게 되므로 따로 다시 연다
        asyncio.get_running_loop().create_task(self._restart())

    async def _publish(self, outgoing: Outgoing) -> None:
        confirmation = await self._exchanges[outgoing.exchange].publish(
            outgoing.message,
            routing_key=outgoing.routing_key,
            mandatory=True,
            timeout=self._settings.mq_confirm_timeout_seconds,
        )
        if not isinstance(confirmation, Basic.Ack):
            raise PublishNacked(type(confirmation).__name__)

    async def _restart(self) -> None:
        """Consumer를 취소하고 채널을 닫아 ACK하지 않은 요청을 작업 큐로 돌려보낸 뒤, 새 채널로 다시 소비한다."""
        if self._restarting:
            return
        self._restarting = True
        try:
            await self._cancel_consumer()
            await self._close_channel()
            await self.start()
            logger.warning("발행을 계속 실패해 Worker 채널을 다시 열었음 jobType=%s", self._queue.job_type)
        except Exception:
            logger.exception("Worker 채널을 다시 열지 못함 jobType=%s", self._queue.job_type)
        finally:
            self._restarting = False

    async def _cancel_consumer(self) -> None:
        if self._amqp_queue is not None and self._consumer_tag is not None:
            try:
                await self._amqp_queue.cancel(self._consumer_tag)
            except Exception:
                logger.debug("Consumer 취소 실패 jobType=%s", self._queue.job_type, exc_info=True)
            self._consumer_tag = None

    async def _close_channel(self) -> None:
        if self._channel is not None:
            metrics.ACTIVE_WORKERS.labels(self._queue.job_type.value).dec()
            try:
                await self._channel.close()
            except Exception:
                logger.debug("채널 닫기 실패 jobType=%s", self._queue.job_type, exc_info=True)
            self._channel = None


def _failure_kind(error: BaseException) -> str:
    if isinstance(error, TimeoutError):
        return "timeout"
    if isinstance(error, PublishNacked):
        return "nack"
    # on_return_raises=True면 라우팅할 큐가 없을 때 DeliveryError가 난다. nack도 같은 예외로 올 수 있다
    frame = getattr(error, "frame", None)
    return "return" if isinstance(frame, Basic.Return) else "nack"


class MqWorkerPool:
    """RabbitMQ 연결 하나와 큐별 Worker들. FastAPI lifespan에서 시작하고 종료한다."""

    def __init__(self, settings: Settings, run: JobRunner = run_job):
        self._settings = settings
        self._run = run
        self._connection: AbstractRobustConnection | None = None
        self._workers: list[QueueWorker] = []

    async def start(self) -> None:
        self._connection = await aio_pika.connect_robust(self._settings.rabbitmq_url)
        try:
            for job_type, count in self._worker_counts().items():
                for _ in range(count):
                    worker = QueueWorker(self._connection, JOB_QUEUES[job_type], self._settings, self._run)
                    await worker.start()
                    self._workers.append(worker)
        except BaseException:
            await self.stop()
            raise
        logger.info("RabbitMQ Worker 시작 %s", {job_type.value: count for job_type, count in self._worker_counts().items()})

    async def stop(self) -> None:
        """모든 Consumer를 먼저 취소해 새 메시지를 받지 않게 한 뒤, 처리 중인 메시지를 기다렸다가 연결을 닫는다."""
        grace = self._settings.mq_shutdown_grace_seconds
        await asyncio.gather(*(worker.stop(grace) for worker in self._workers), return_exceptions=True)
        self._workers.clear()
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    def _worker_counts(self) -> dict[JobType, int]:
        return {
            JobType.SCHEDULE_GENERATE: self._settings.mq_workers_schedule_generate,
            JobType.PHOTO_MISSION_GENERATE: self._settings.mq_workers_photo_mission_generate,
            JobType.PHOTO_JUDGE: self._settings.mq_workers_photo_judge,
        }
