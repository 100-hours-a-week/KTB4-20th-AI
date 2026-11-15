"""AI Worker 지표 (planit-docs ai/RABBITMQ_AI_WORKER_IMPLEMENTATION.md 7절). /metrics에 함께 노출된다."""

from prometheus_client import Counter, Gauge, Histogram

MESSAGES = Counter(
    "ai_worker_messages_total",
    "처리한 요청 메시지 수. outcome: succeeded, failed_result, retry, dlq, deadline_skipped",
    ["job_type", "outcome"],
)
REDELIVERED = Counter(
    "ai_worker_redelivered_total",
    "재전달된 요청 메시지 수. 늘면 결과 발행 실패나 Worker 비정상 종료를 의심한다",
    ["job_type"],
)
PUBLISH_FAILURES = Counter(
    "ai_worker_publish_failures_total",
    "다음 목적지 발행 실패 수. kind: nack, return, timeout, connection",
    ["job_type", "kind"],
)
PROCESSING_SECONDS = Histogram(
    "ai_worker_processing_seconds",
    "AI 호출 시간(초)",
    ["job_type"],
    buckets=(0.5, 1, 2.5, 5, 10, 20, 30, 60, 90, 120),
)
ACTIVE_WORKERS = Gauge("ai_worker_active", "지금 소비 중인 Worker 수", ["job_type"])
