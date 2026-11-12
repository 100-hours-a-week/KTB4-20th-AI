"""작업 종류별로 기존 서비스 함수를 불러 AI 결과를 만든다. HTTP API와 같은 함수를 쓰고, 결과는 HTTP 응답 본문과 같은 모양이다."""

import asyncio
from typing import Any

from app.mq.topology import JobQueue, JobType
from app.photomissions.pipeline_generate import generate_missions
from app.photomissions.pipeline_verify import verify_photo
from app.photomissions.schemas import PhotoMissionGenerateRequest, VerifyRequest
from app.trips.router import run_place_selection
from app.trips.schemas.api import (
    E_Response_Status_Code,
    Place_Selection_Request,
    Place_Selection_Response_Data,
    Response,
)


async def run_job(queue: JobQueue, request: dict[str, Any]) -> dict[str, Any]:
    """
    request를 작업별 요청 모델로 검증한 뒤 AI를 부른다. 검증에 실패하면 pydantic ValidationError,
    queue.timeout_seconds를 넘기면 asyncio.TimeoutError를 던진다.
    """
    match queue.job_type:
        case JobType.SCHEDULE_GENERATE:
            return await _place_selection(queue, Place_Selection_Request.model_validate(request))
        case JobType.PHOTO_MISSION_GENERATE:
            generate_request = PhotoMissionGenerateRequest.model_validate(request)
            response = await asyncio.wait_for(generate_missions(generate_request), queue.timeout_seconds)
            return response.model_dump(mode="json", by_alias=True)
        case JobType.PHOTO_JUDGE:
            # 판정 요청에는 미션 ID(mission_id)가 더 들어오지만 판정에는 쓰지 않는다(HTTP API의 경로 변수와 같음)
            verify_request = VerifyRequest.model_validate(request)
            response = await asyncio.wait_for(verify_photo(verify_request), queue.timeout_seconds)
            return response.model_dump(mode="json", by_alias=True)


async def _place_selection(queue: JobQueue, request: Place_Selection_Request) -> dict[str, Any]:
    # place-selection은 동기 함수라 스레드에서 돌린다. wait_for가 끝나도 스레드는 멈추지 않으므로
    # RunPod 호출 자체의 timeout(runpod_model_timeout_seconds)이 실제 상한이다
    data: Place_Selection_Response_Data = await asyncio.wait_for(
        asyncio.to_thread(run_place_selection, request), queue.timeout_seconds,
    )
    response = Response[Place_Selection_Response_Data](status_code=E_Response_Status_Code.SUCCESS, data=data)
    return response.model_dump(mode="json", by_alias=True)
