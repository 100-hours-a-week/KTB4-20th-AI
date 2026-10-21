import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.auth import verify_internal_token
from app.trips.schemas.api import (
    E_Response_Status_Code,
    Place_Selection_Request,
    Place_Selection_Response_Data,
    Precheck_Response_Data,
    Response,
)
from app.trips.schemas.schemas import E_Breaker
from app.trips.services.external_factors import outdoor_excluded_days
from app.trips.services.precheck_service import PrecheckUnavailable, build_precheck
from app.trips.services.trip_orchestrator import build_trip_plans

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/trips",
    tags=["trips"],
    dependencies=[Depends(verify_internal_token)],
)

@router.post("/place-selection", response_model=Response[Place_Selection_Response_Data])
def place_selection(request: Place_Selection_Request) -> Response[Place_Selection_Response_Data]:
    """
    최대 8박 9일의 일정을 3개 안으로 돌려준다 (ADR-0008). places[안][일자][장소]이고, 안은 [0] 규칙(평점 상위 중 시드로 고름, Spring이 재정렬하는 원본),
    [1] 모델 경로의 취향 안, [2] 모델 경로의 다양성 안이다. 모델 경로가 실패하면 그 안을 규칙 변형으로 채운다.

    external_factors(precheck 응답)는 요청의 지역과 여행 날짜에 해당하는 has_issue만 읽고, 신호가 있는 날은 야외 후보를 뺀다.
    """

    # 회피조건 취합 (그룹 전체)
    deal_breakers: list[E_Breaker] = []
    for member in request.members:
        deal_breakers.extend(member.deal_breakers)

    num_days = (request.end_date - request.start_date).days + 1
    signal_days = outdoor_excluded_days(request.external_factors, request.region, request.start_date, num_days)
    places = build_trip_plans(request, deal_breakers, num_days, signal_days)

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

@router.post("/precheck", response_model=Response[Precheck_Response_Data])
def precheck() -> Response[Precheck_Response_Data]:
    """
    BE가 여행 출발 하루 전 06시에 보내는 트리거 (ADR-0009). 요청 본문은 없다.
    5개 지역의 내일부터 9일 치 날씨 위험을 돌려준다(단기·중기 예보). 응답은 BE가 place-selection의 external_factors로 그대로 전달한다.
    기상청 호출이 일부 실패하면 성공한 날짜·지역만 담고, 전부 실패하거나 인증키가 없으면 503이다.
    """
    try:
        data = build_precheck()
    except PrecheckUnavailable as error:
        logger.error("precheck 불가: %s", error)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from None
    return Response[Precheck_Response_Data](status_code=E_Response_Status_Code.SUCCESS, data=data)
