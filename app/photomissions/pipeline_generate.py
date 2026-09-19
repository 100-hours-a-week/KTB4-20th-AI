import asyncio
import uuid

from app.photomissions.gemini_client import generate_structured
from app.photomissions.prompts import MISSION_SYSTEM_PROMPT
from app.photomissions.schemas import (
    Mission,
    MissionDescription,
    MissionPlace,
    PhotoMissionGenerateRequest,
    PhotoMissionGenerateResponse,
    T_Preference,
)


def new_mission_id() -> str:
    return f"ms_{uuid.uuid4().hex[:12]}"


async def write_description(place: MissionPlace) -> MissionDescription:
    # 1. 미션 문구 + scope 생성. mission_id는 여기서 다루지 않는다.

    # description은 verify의 mission_description으로 재사용되므로, 사진만 보고 수행 여부를 판단할 수 있어야 한다.
    # scope 판정 기준(완료 규칙)은 MISSION_SYSTEM_PROMPT에 있다.

    user_prompt = (
        f"장소명: {place.displayName.text}\n"
        f"반영된 취향: {', '.join(place.matched_preferences) or '없음'}"
    )
    return await generate_structured(
        system=MISSION_SYSTEM_PROMPT,
        user=user_prompt,
        response_schema=MissionDescription,
    )


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
    # 장소별로 동시에 호출
    results = await asyncio.gather(
        *(write_description(place) for place in request.places)
    )
    # 장소와 결과를 순서에 맞춰 짝지어 Mission 객체를 만들고, 최종 응답에 담는다
    missions = [
        build_mission(place, result)
        for place, result in zip(request.places, results)
    ]
    return PhotoMissionGenerateResponse(missions=missions)


