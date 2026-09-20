from fastapi import APIRouter, Depends

from app.core.auth import verify_internal_token
from app.trips.services.preference import calculate_group_preference
from app.trips.services.slot_allocation import allocate_slots_by_category
from app.trips.services.place_selection import select_places
from app.trips.schema.api import Response, E_Response_Status_Code, Place_Selection_Request, Place_Selection_Response_Data, Course_Recommendation_Request, Course_Recommendation_Response_Data, Precheck_Request, Precheck_Response_Data

router = APIRouter(
    prefix="/trips",
    tags=["trips"],
    dependencies=[Depends(verify_internal_token)],
)

@router.post("/place-selection", response_model=Response[Place_Selection_Response_Data])
def place_selection(request: Place_Selection_Request) -> Response[Place_Selection_Response_Data]:
    # 1. 취향 판정 로직
    preferences = calculate_group_preference(request.members)

    # 2. 슬롯 배분 (카테고리별 필요 개수 산출)
    slot_counts = allocate_slots_by_category(request.start_date, request.end_date, preferences)

    # 3. 최종 장소 선택
    selected_places = select_places(preferences, slot_counts)

    return Response[Place_Selection_Response_Data](
        status_code=E_Response_Status_Code.SUCCESS,
        data=Place_Selection_Response_Data(places=selected_places),
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