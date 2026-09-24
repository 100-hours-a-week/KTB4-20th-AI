from fastapi import APIRouter, Depends

from app.core.auth import verify_internal_token
from app.trips.schemas.api import (
    E_Response_Status_Code,
    Place_Selection_Request,
    Place_Selection_Response_Data,
    Response,
)
from app.trips.schemas.schemas import E_Breaker
from app.trips.services.assign_time_slot import assign_time_slots
from app.trips.services.place_selection import select_places
from app.trips.services.preference import calculate_group_preference
from app.trips.services.slot_allocation import allocate_slots

router = APIRouter(
    prefix="/trips",
    tags=["trips"],
    dependencies=[Depends(verify_internal_token)],
)

@router.post("/place-selection", response_model=Response[Place_Selection_Response_Data])
def place_selection(request: Place_Selection_Request) -> Response[Place_Selection_Response_Data]:
    """
    TODO: v1에서는 하루 일정 생성만 가능하므로 현재 로직이 문제가 없습니다.
    v2, v3에서는 시작일, 종료일에 기간이 생깁니다.
    현재는 "슬롯 배분" 함수가 카테고리 별로 몇 개의 장소가 필요한지 dict[E_Preference, float]로 알 수 있지만
    v2, v3에서 카테고리별 장소가 렌덤하게 배치되어도 무방한지, 만약 안된다면 어떻게 배치할지 논의가 선행되어야 합니다.
    선행된 논의를 바탕으로 슬롯 배분이 일자별로 dict[E_Preference, float]를 가진 "배열"로 변경될 수도 있기 때문입니다.
    """

    # 1. 취향 판정 로직
    preferences = calculate_group_preference(request.members)

    # 2. 슬롯 배분 (카테고리별 필요 개수 산출)
    slot_counts = allocate_slots(request.start_date, request.end_date, preferences)

    # 3. 회피조건 취합 (그룹 전체)
    deal_breakers: list[E_Breaker] = []
    for member in request.members:
        deal_breakers.extend(member.deal_breakers)

    # 4. 최종 장소 선택 (카테고리별 딕셔너리 반환)
    places_by_category = select_places(preferences, slot_counts, deal_breakers, request.region)

    # 5. 시간대 슬롯 배정
    slot_result = assign_time_slots(places_by_category, preferences)

    return Response[Place_Selection_Response_Data](
        status_code=E_Response_Status_Code.SUCCESS,
        data=Place_Selection_Response_Data(places=slot_result),
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