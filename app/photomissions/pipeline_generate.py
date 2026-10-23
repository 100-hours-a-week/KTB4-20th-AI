import asyncio
import json
import logging
import math
import time
import uuid
from dataclasses import asdict, dataclass

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
_cleanups: set[asyncio.Task] = set()  # 부모가 먼저 빠져나간 정리 태스크도 끝날 때까지 붙잡아 둔다(이벤트 루프는 약한 참조만 가짐)
generate_log = logging.getLogger("uvicorn.error")  # verify와 같이 uvicorn 기본 설정에서 찍히는 로거


@dataclass
class ChunkTrace:
    # 묶음 하나의 기록. status: not_started(차례가 안 옴) / waiting(자리 대기) / running / ok / failed / cancelled
    places: int
    status: str = "not_started"
    wait_sec: float | None = None  # 동시 호출 자리를 기다린 시간
    call_sec: float | None = None  # 재시도를 포함해 Gemini를 부른 시간
    attempts: int = 0  # 실제 Gemini 시도 횟수
    error_status: int | None = None
    cancelled_while: str | None = None  # 취소됐을 때 waiting(자리 대기 중) / running(호출 중)
    error_kind: str | None = None  # HTTP 실패가 아닌 오류의 종류(예: 취소 정리 중 난 오류). 문장은 남기지 않는다


def start() -> None:
    # 앱 시작(lifespan) 때 만든다. 이벤트 루프가 바뀌는 테스트·스크립트도 실행마다 새로 만들게 한다
    # 설정도 여기서 검사한다. 앱 시작 검사를 거치지 않는 측정 스크립트도 잘못된 마감 값으로 돌지 않게
    check_settings(settings)
    global _call_slots
    _call_slots = asyncio.Semaphore(MAX_CONCURRENT_CALLS)


def close() -> None:
    # 남은 정리 태스크(_cleanups)는 기다리지 않는다.
    # 프로세스 종료 시 완료를 보장하지 않으며, 다음 start()는 새 세마포어를 만든다.
    global _call_slots
    _call_slots = None


def settings_errors(s: Settings) -> list[str]:
    # generate 설정 검사. 틀리면 앱이 시작되지 않는다 (verify 백엔드와 상관없이 항상 검사)
    deadline = s.generate_deadline_seconds
    if not math.isfinite(deadline) or deadline <= 0:  # inf면 마감이 사라지고 nan은 비교가 안 된다
        return ["GENERATE_DEADLINE_SECONDS는 0보다 큰 유한한 수여야 함"]
    return []


def check_settings(s: Settings) -> None:
    if problems := settings_errors(s):
        raise RuntimeError("generate 설정 오류: " + " / ".join(problems))


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


async def write_descriptions(places: list[MissionPlace], trace: dict | None = None) -> MissionBatch:
    # 1. 미션 문구 + scope 생성. 묶음 하나(장소 최대 MAX_PLACES_PER_CALL곳)를 한 번의 호출로 만든다. mission_id는 여기서 다루지 않는다.

    # description은 verify의 mission_description으로 재사용되므로, 사진만 보고 수행 여부를 판단할 수 있어야 한다.
    # scope 판정 기준(완료 규칙)은 MISSION_SYSTEM_PROMPT에 있다.

    # TODO: Retry-After 값(30초)은 baseline 측정 후 확정
    try:
        return await generate_structured(
            system=MISSION_SYSTEM_PROMPT,
            user=build_user_prompt(places),
            response_schema=MissionBatch,
            is_valid=lambda batch: covers_every_place(batch, len(places)),
            trace=trace,
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
    # 장소가 많으면 한 번의 호출이 길어진다(15곳 이상에서 당시 백엔드 타임아웃 30초를 자주 넘음, 2026-10-19 측정)
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


async def _stop_workers(workers: list[asyncio.Task]) -> list[BaseException]:
    # 남은 작업자를 취소하고 모두 끝날 때까지 기다린 뒤, 끝난 작업자의 예외를 전부 회수한다
    # (회수하지 않으면 "Task exception was never retrieved"로 남는다. 취소 정리 중 난 오류는 묶음 로그의 error_kind에 남는다.
    # 단 정리 중 부모가 한 번 더 취소돼 먼저 빠져나가면 요청 로그가 먼저 찍혀, 그 뒤에 난 오류는 회수만 되고 기록되지 않는다)
    for w in workers:
        w.cancel()
    await asyncio.wait(workers)
    return [w.exception() for w in workers if not w.cancelled() and w.exception() is not None]


def _collect(task: asyncio.Task) -> None:
    # 부모가 먼저 빠져나가 정리 태스크의 결과를 아무도 받지 않는 경우에도 결과를 회수해 둔다
    _cleanups.discard(task)
    if not task.cancelled():
        task.exception()


async def _write_chunks(chunks: list[list[MissionPlace]], traces: list[ChunkTrace]) -> list[MissionBatch]:
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
            trace, call = traces[index], {}
            trace.status, waiting = "waiting", time.monotonic()
            try:
                async with slots:
                    trace.status, started = "running", time.monotonic()
                    trace.wait_sec = round(started - waiting, 3)
                    try:
                        batches[index] = await write_descriptions(chunks[index], trace=call)
                    finally:
                        trace.call_sec = round(time.monotonic() - started, 3)
                        trace.attempts = call.get("attempts", 0)
                trace.status = "ok"
            except HTTPException as e:
                trace.status, trace.error_status = "failed", e.status_code
                failures[index] = e
                raise
            except asyncio.CancelledError:
                if trace.status == "waiting":  # 자리를 못 얻고 취소됐다. 혼잡 정도를 알 수 있게 기다린 시간을 남긴다
                    trace.wait_sec = round(time.monotonic() - waiting, 3)
                trace.status, trace.cancelled_while = "cancelled", trace.status
                raise
            except Exception as e:
                trace.status, trace.error_kind = "failed", type(e).__name__
                raise

    # TaskGroup은 쓰지 않는다. 3.12의 TaskGroup은 끝나기를 기다리는 중 자식이 실패하면 부모 태스크에 건 취소를
    # 거두지 않아, 바깥 취소·전체 마감과 묶음 실패가 겹치면 취소가 HTTP 실패로 바뀌거나 마감이 504가 되지 않는다.
    # 여기서는 부모에 취소를 걸지 않으므로, 부모가 받는 취소는 바깥 취소와 전체 마감뿐이다
    workers = [asyncio.create_task(worker()) for _ in range(min(MAX_CONCURRENT_CALLS, len(chunks)))]
    try:
        await asyncio.wait(workers, return_when=asyncio.FIRST_EXCEPTION)
    finally:
        # 묶음 실패·바깥 취소·마감이면 남은 작업자를 정리한다. 정리는 별도 태스크가 맡아, 정리 중 부모가 또 취소돼
        # 먼저 빠져나가도 작업자가 끝나고(자리 반납) 예외를 회수할 때까지 간다. 부모가 받던 취소·마감은 그대로 올라간다
        cleanup = asyncio.create_task(_stop_workers(workers))
        _cleanups.add(cleanup)
        cleanup.add_done_callback(_collect)
        raised = await asyncio.shield(cleanup)
    if bugs := [e for e in raised if not isinstance(e, HTTPException)]:
        # 프로그램 오류가 하나라도 있으면 502·429로 숨기지 않고 그대로 올린다
        raise BaseExceptionGroup("generate 묶음 처리 중 프로그램 오류", bugs)
    if failures:
        raise pick_failure(failures)
    return [batch for batch in batches if batch is not None]


async def generate_missions(
    request: PhotoMissionGenerateRequest,
) -> PhotoMissionGenerateResponse:
    # 장소를 묶음으로 나눠 동시에 만든다. 묶음 대기·호출·재시도를 모두 합쳐 GENERATE_DEADLINE_SECONDS 안에 끝낸다
    # 마감은 취소로 멈추는 방식이라 정확히 그 시각에 끊기지는 않고, Gemini 쪽에서 이미 시작된 생성 비용은 나갈 수 있다
    chunks = split_places(request.places)
    traces = [ChunkTrace(places=len(chunk)) for chunk in chunks]
    started = time.monotonic()
    final_status, outcome, error_kind = 500, "error", None
    try:
        try:
            async with asyncio.timeout(settings.generate_deadline_seconds) as deadline:
                batches = await _write_chunks(chunks, traces)
        except TimeoutError as e:
            if not deadline.expired():  # 마감이 아닌 다른 곳에서 난 TimeoutError는 그대로 올린다
                raise
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="미션 문구 생성 시간 초과",
            ) from e
        response = PhotoMissionGenerateResponse(missions=assemble_missions(chunks, batches))
        final_status, outcome = 200, "ok"
        return response
    except HTTPException as e:
        final_status, error_kind = e.status_code, e.detail if isinstance(e.detail, str) else type(e).__name__
        raise
    except asyncio.CancelledError:
        outcome = "cancelled"  # 응답을 보내지 않았으니 상태 코드를 지어내지 않는다
        raise
    except Exception as e:
        error_kind = type(e).__name__  # 프로그램 오류. 문장 대신 종류만 남긴다
        raise
    finally:
        # 마감·묶음 크기·동시 상한을 나중에 조정할 수 있게 요청마다 한 줄 남긴다. 장소 이름은 넣지 않는다
        fields = {
            "itinerary_id": request.itinerary_id,
            "places": len(request.places),
            "chunks": [asdict(trace) for trace in traces],
            "deadline_sec": settings.generate_deadline_seconds,
            "total_sec": round(time.monotonic() - started, 3),
            "final_status": final_status if outcome != "cancelled" else None,
            "outcome": outcome,
            "error_kind": error_kind,
        }
        generate_log.info("generate %s", json.dumps(fields, ensure_ascii=False))
