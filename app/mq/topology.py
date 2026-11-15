"""계약 문서 3절의 Exchange·큐·routing key 이름. 선언은 백엔드만 하고 AI 서버는 이름으로 찾아 쓴다."""

from dataclasses import dataclass
from enum import StrEnum


class JobType(StrEnum):
    SCHEDULE_GENERATE = "SCHEDULE_GENERATE"
    PHOTO_MISSION_GENERATE = "PHOTO_MISSION_GENERATE"
    PHOTO_JUDGE = "PHOTO_JUDGE"


@dataclass(frozen=True)
class JobQueue:
    job_type: JobType
    task_queue: str
    routing_key: str
    # AI 호출 한 번을 기다리는 최대 시간(초). 넘으면 AI_TIMEOUT으로 재시도한다
    timeout_seconds: float

    def retry_routing_key(self, stage: int) -> str:
        """stage는 1부터 센다. 1·2·3단계는 5·10·20초 뒤 작업 큐로 돌아온다."""
        return f"{self.routing_key}.retry.{stage}"

    @property
    def dlq_routing_key(self) -> str:
        return f"{self.routing_key}.dlq"


REQUEST_EXCHANGE = "ai.request"
RETRY_EXCHANGE = "ai.retry"
DLX_EXCHANGE = "ai.dlx"
RESULT_EXCHANGE = "ai.result"
RESULT_ROUTING_KEY = "ai-result"

# AI Worker가 발행하는 Exchange. 소비는 작업 큐에서만 한다
PUBLISH_EXCHANGES = (RESULT_EXCHANGE, RETRY_EXCHANGE, DLX_EXCHANGE)

JOB_QUEUES: dict[JobType, JobQueue] = {
    JobType.SCHEDULE_GENERATE: JobQueue(
        JobType.SCHEDULE_GENERATE, "ai.schedule-generation.task", "ai-schedule-generation", 120.0,
    ),
    JobType.PHOTO_MISSION_GENERATE: JobQueue(
        JobType.PHOTO_MISSION_GENERATE, "ai.photomission-generation.task", "ai-photomission-generation", 110.0,
    ),
    JobType.PHOTO_JUDGE: JobQueue(
        JobType.PHOTO_JUDGE, "ai.photomission-verification.task", "ai-photomission-verification", 30.0,
    ),
}

# AI 처리 실패로 Retry Queue에 보낼 수 있는 최대 횟수. 3번째까지 쓰면 DLQ로 보낸다
MAX_AI_RETRIES = 3
# 포토미션 생성 05시 재처리 메시지의 x-retry-count. 이 메시지가 실패하면 DLQ 대신 FAILED 결과를 보낸다
REPROCESS_RETRY_COUNT = 4
