import hashlib
import logging

from fastapi import APIRouter, Depends

from app.core.auth import verify_internal_token
from app.core.config import settings
from app.trips.schemas.api import (
    E_Response_Status_Code,
    Place_Selection_Request,
    Place_Selection_Response_Data,
    Response,
)
from app.trips.schemas.schemas import E_Breaker
from app.trips.services.model.selection import (
    model_slot_plan_seed,
    select_places_by_model,
)
from app.trips.services.place_selection import select_places_by_slot_plan
from app.trips.services.slot_plan import build_slot_plan

logger = logging.getLogger(__name__)

MIN_PLACES = 5  # 모델 경로가 이보다 적게 내면 규칙 경로로 되돌린다

router = APIRouter(
    prefix="/trips",
    tags=["trips"],
    dependencies=[Depends(verify_internal_token)],
)

@router.post("/place-selection", response_model=Response[Place_Selection_Response_Data])
def place_selection(request: Place_Selection_Request) -> Response[Place_Selection_Response_Data]:
    """
    TODO: v1에서는 하루 일정 생성만 가능합니다.
    v2에서 n-1박 n일 일정을 지원하려면, build_slot_plan(하루 단위)을 일자별로 반복 호출하는 것만으로는 부족합니다.

    [반복 호출 시 문제]
    1. 매일 같은 카테고리 배치: 입력(members)이 같아 R1, R2가 매일 동일하고, 3위 이하 카테고리는 배치되지 않음
    2. 매일 같은 장소 선택: select_places_by_slot_plan이 호출마다 평점순 상위 후보를 새로 조회함
    3. 첫날·마지막 날 구분 없음: 도착일, 출발일도 하루 전체 템플릿(오전~저녁 이후)을 사용함
    4. 저녁 이후 슬롯 매일 생성: 문항 10 점수가 같아 조건 충족 시 매일 밤 배치됨

    [추가 필요 요소]
    1. 일자 유형별 템플릿 (도착일, 종일, 출발일의 시간대 구성)
    2. 일자 간 카테고리 분배 규칙 (기획 결정 선행 필요)
    3. 여행 전체 장소 중복 제거
    4. 저녁 이후 슬롯 빈도 규칙
    """

    # 3. 회피조건 취합 (그룹 전체)
    deal_breakers: list[E_Breaker] = []
    for member in request.members:
        deal_breakers.extend(member.deal_breakers)

    places = None
    if settings.place_selection_mode == "model":
        # 학습한 모델 경로(ADR-0003). 동점 취향은 고정 순서로 처리한다. 어떤 이유로든 실패하면 규칙 경로로 되돌린다
        try:
            from app.trips.services.model.runtime import get_router

            model_plan = build_slot_plan(request.members, model_slot_plan_seed())
            # 첫 장소 무작위의 시드는 요청 내용에서 만든다(ADR-0007): 같은 요청은 같은 일정, 다른 요청은 다른 일정
            seed = int(hashlib.sha256(request.model_dump_json().encode("utf-8")).hexdigest()[:15], 16)
            places, route = select_places_by_model(
                model_plan, deal_breakers, request.region, request.members, get_router(),
                first_k=settings.model_first_place_top_k, seed=seed,
            )
            if len(places) < MIN_PLACES:
                raise ValueError(f"모델 경로 결과 {len(places)}곳 (< {MIN_PLACES})")
            logger.info("place-selection: model route=%s places=%d", route, len(places))
        except Exception:
            logger.exception("모델 경로 실패, 규칙 경로로 되돌림")
            places = None

    if places is None:
        # 1. 하루 슬롯 구조 결정 (시간대별 카테고리, 저녁 이후 슬롯 여부)
        slot_plan = build_slot_plan(request.members)

        # 3. slot plan 순서대로 장소 선택 (후보 부족 시 강등, 보충 포함)
        places = select_places_by_slot_plan(slot_plan, deal_breakers, request.region, request.members)

    return Response[Place_Selection_Response_Data](
        status_code=E_Response_Status_Code.SUCCESS,
        data=Place_Selection_Response_Data(places=places),
    )

# @router.post("/course-recommendation", response_model=Response[Course_Recommendation_Response_Data])
# def course_recommendation(request: Course_Recommendation_Request) -> Response[Course_Recommendation_Response_Data]:

#     courses=[] # TODO: requests를 사용해 courses 만들기
#     # 1. ??
#     # 2. ?? ...
#     return Response[Course_Recommendation_Response_Data](
#         status_code=E_Response_Status_Code.SUCCESS,
#         data=Course_Recommendation_Response_Data(courses=courses),
#     )

# @router.post("/precheck", response_model=Response[Precheck_Response_Data])
# def precheck(request: Precheck_Request) -> Response[Precheck_Response_Data]:
#     courses=[] # TODO: requests를 사용해 precheck 만들기
#     # 1. ??
#     # 2. ?? ...
#     has_issue=True
#     affected_place_ids=[]
#     reasons=[]
#     return Response[Precheck_Response_Data](
#         status_code=E_Response_Status_Code.SUCCESS,
#         data=Precheck_Response_Data(
#             has_issue=has_issue,
#             affected_place_ids=affected_place_ids,
#             reasons=reasons,
#         ),
#     )