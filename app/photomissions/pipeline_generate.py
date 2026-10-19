import asyncio
import math
import uuid

import httpx
from fastapi import HTTPException, status
from google.genai import errors

from app.core.config import Settings, settings
from app.photomissions.gemini_client import generate_structured
from app.photomissions.prompts import MISSION_SYSTEM_PROMPT
from app.photomissions.schemas import (
    Mission,
    MissionBatch,
    MissionDescription,
    MissionPlace,
    PhotoMissionGenerateRequest,
    PhotoMissionGenerateResponse,
    T_Preference,
)

MAX_PLACES_PER_CALL = 8  # 10곳 한 번 호출이 최대 19.81초(3번 측정). 여유를 두고 8곳으로 시작한다
# 이 서버 프로세스 전체에서 동시에 나가는 generate Gemini 호출 수. 요청마다가 아니라 모든 generate 요청이 나눠 쓴다
# (서버가 여러 대면 전체 동시 호출은 3 × 대수). 묶음 안 재시도가 끝날 때까지 자리를 잡고 있는다
MAX_CONCURRENT_CALLS = 3
_call_slots: asyncio.Semaphore | None = None


def start() -> None:
    # 앱 시작(lifespan) 때 만든다. 이벤트 루프가 바뀌는 테스트·스크립트도 실행마다 새로 만들게 한다
    global _call_slots
    _call_slots = asyncio.Semaphore(MAX_CONCURRENT_CALLS)


def close() -> None:
    global _call_slots
    _call_slots = None


def settings_errors(s: Settings) -> list[str]:
    # generate 설정 검사. 틀리면 앱이 시작되지 않는다 (verify 백엔드와 상관없이 항상 검사)
    deadline = s.generate_deadline_seconds
    if not math.isfinite(deadline) or deadline <= 0:  # inf면 마감이 사라지고 nan은 비교가 안 된다
        return ["GENERATE_DEADLINE_SECONDS는 0보다 큰 유한한 수여야 함"]
    return []


def check_settings(s: Settings) -> None:
    if errors := settings_errors(s):
        raise RuntimeError("generate 설정 오류: " + " / ".join(errors))


def new_mission_id() -> str:
    return f"ms_{uuid.uuid4().hex[:12]}"


def build_user_prompt(places: list[MissionPlace]) -> str:
    # 장소마다 번호를 붙인다. 모델은 이 번호를 number에 그대로 적어 돌려준다
    return "\n".join(
        f"{number}. 장소명: {place.displayName.text}\n"
        f"   반영된 취향: {', '.join(place.matched_preferences) or '없음'}"
        for number, place in enumerate(places, start=1)
    )


def covers_every_place(batch: MissionBatch, place_count: int) -> bool:
    # 번호 1..place_count가 빠짐·중복·없는 번호 없이 한 번씩 있어야 장소와 빠짐없이 짝지을 수 있다
    numbers = sorted(m.number for m in batch.missions)
    return numbers == list(range(1, place_count + 1))


async def write_descriptions(places: list[MissionPlace]) -> MissionBatch:
    # 1. 미션 문구 + scope 생성. 장소 전부를 한 번의 호출로 만든다. mission_id는 여기서 다루지 않는다.

    # description은 verify의 mission_description으로 재사용되므로, 사진만 보고 수행 여부를 판단할 수 있어야 한다.
    # scope 판정 기준(완료 규칙)은 MISSION_SYSTEM_PROMPT에 있다.

    # TODO: Retry-After 값(30초)은 baseline 측정 후 확정
    try:
        return await generate_structured(
            system=MISSION_SYSTEM_PROMPT,
            user=build_user_prompt(places),
            response_schema=MissionBatch,
            is_valid=lambda batch: covers_every_place(batch, len(places)),
        )
    except errors.APIError as e:
        if e.code == status.HTTP_429_TOO_MANY_REQUESTS:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Gemini 호출 한도 초과",
                headers={"Retry-After": "30"},
            ) from e
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="미션 문구 생성 실패",
        ) from e
    except httpx.TimeoutException as e:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="미션 문구 생성 시간 초과",
        ) from e
    except ValueError as e:  # 재시도 후에도 응답이 스키마와 맞지 않거나 장소와 짝지을 수 없음
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="미션 문구 생성 실패",
        ) from e


def pick_primary_category(matched_preferences: list[T_Preference]) -> T_Preference | None:
    # 대표 카테고리 
    return matched_preferences[0] if matched_preferences else None


def build_mission(place: MissionPlace, result: MissionDescription) -> Mission:
    # 2. 응답 조립 
    return Mission(
        mission_id=new_mission_id(),
        place_id=place.id,
        description=result.description,
        scope=result.scope,
        primary_category=pick_primary_category(place.matched_preferences),
    )


def split_places(places: list[MissionPlace]) -> list[list[MissionPlace]]:
    # 장소가 많으면 한 번의 호출이 백엔드 타임아웃(30초)을 넘는다(15곳 이상에서 자주 넘음, 2026-10-19 측정)
    # MAX_PLACES_PER_CALL곳 이하 묶음으로 고르게 나눈다. 순서를 지키고 이어진 장소끼리 묶는다(20곳 → 7+7+6)
    count = math.ceil(len(places) / MAX_PLACES_PER_CALL)
    size, extra = divmod(len(places), count)
    chunks, start = [], 0
    for index in range(count):
        end = start + size + (1 if index < extra else 0)
        chunks.append(places[start:end])
        start = end
    return chunks


def assemble_missions(
    chunks: list[list[MissionPlace]], batches: list[MissionBatch],
) -> list[Mission]:
    # 묶음마다 번호가 1부터 다시 붙는다. 모델이 돌려준 순서와 상관없이 번호로 장소와 짝지어 요청 순서대로 담는다
    missions = []
    for chunk, batch in zip(chunks, batches, strict=True):
        by_number = {m.number: m for m in batch.missions}
        missions += [build_mission(place, by_number[number]) for number, place in enumerate(chunk, start=1)]
    return missions


def pick_failure(failures: dict[int, HTTPException]) -> HTTPException:
    # 실제로 받은 묶음 실패 중 하나를 고른다. 취소되거나 시작 못 한 묶음의 결과는 추측하지 않는다
    # 429가 있으면 429(백엔드가 Retry-After 뒤에 다시 부르게), 아니면 묶음 번호가 가장 작은 실패. 여럿이면 번호가 작은 쪽
    ordered = [failures[index] for index in sorted(failures)]
    return next((e for e in ordered if e.status_code == status.HTTP_429_TOO_MANY_REQUESTS), ordered[0])


async def _write_chunks(chunks: list[list[MissionPlace]]) -> list[MissionBatch]:
    # 요청마다 작업자를 최대 MAX_CONCURRENT_CALLS개만 두고, 작업자가 묶음을 차례로 가져간다
    # (장소가 아주 많아도 태스크가 묶음 수만큼 늘지 않는다). 한 묶음이 실패하면 나머지는 취소하고 전체를 실패로 돌려준다
    if _call_slots is None:
        raise RuntimeError("pipeline_generate.start()를 먼저 불러야 함 (앱은 lifespan에서 부름)")
    slots = _call_slots
    batches: list[MissionBatch | None] = [None] * len(chunks)
    failures: dict[int, HTTPException] = {}
    todo = iter(range(len(chunks)))  # 작업자들이 나눠 쓴다. 이벤트 루프 하나에서 돌아 next()가 겹치지 않는다

    async def worker() -> None:
        for index in todo:
            async with slots:
                try:
                    batches[index] = await write_descriptions(chunks[index])
                except HTTPException as e:
                    failures[index] = e
                    raise

    try:
        async with asyncio.TaskGroup() as group:
            for _ in range(min(MAX_CONCURRENT_CALLS, len(chunks))):
                group.create_task(worker())
    except BaseExceptionGroup as failed:
        # 묶음 실패는 ExceptionGroup으로 묶여 온다. 그대로 두면 500이 되므로 HTTP 실패를 하나 골라 올린다
        # 프로그램 오류가 하나라도 섞여 있으면 502로 숨기지 않고 그대로 올린다
        if all(isinstance(e, HTTPException) for e in failed.exceptions):
            raise pick_failure(failures) from None
        raise
    return [batch for batch in batches if batch is not None]


async def generate_missions(
    request: PhotoMissionGenerateRequest,
) -> PhotoMissionGenerateResponse:
    # 장소를 묶음으로 나눠 동시에 만든다. 묶음 대기·호출·재시도를 모두 합쳐 GENERATE_DEADLINE_SECONDS 안에 끝낸다
    # 마감은 취소로 멈추는 방식이라 정확히 그 시각에 끊기지는 않고, Gemini 쪽에서 이미 시작된 생성 비용은 나갈 수 있다
    chunks = split_places(request.places)
    try:
        async with asyncio.timeout(settings.generate_deadline_seconds) as deadline:
            batches = await _write_chunks(chunks)
    except TimeoutError as e:
        if not deadline.expired():  # 마감이 아닌 다른 곳에서 난 TimeoutError는 그대로 올린다
            raise
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="미션 문구 생성 시간 초과",
        ) from e
    return PhotoMissionGenerateResponse(missions=assemble_missions(chunks, batches))
