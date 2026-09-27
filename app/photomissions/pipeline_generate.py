import uuid

import httpx
from fastapi import HTTPException, status
from google.genai import errors

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


async def generate_missions(
    request: PhotoMissionGenerateRequest,
) -> PhotoMissionGenerateResponse:
    batch = await write_descriptions(request.places)
    # 모델이 돌려준 순서와 상관없이, 번호로 요청의 장소와 짝지어 요청 순서대로 담는다
    by_number = {m.number: m for m in batch.missions}
    missions = [
        build_mission(place, by_number[number])
        for number, place in enumerate(request.places, start=1)
    ]
    return PhotoMissionGenerateResponse(missions=missions)
