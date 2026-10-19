from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from app.core.time import get_service_today
from app.trips.schemas.schemas import E_Region, Member_Survey, Place


class E_Response_Status_Code(int, Enum):
    SUCCESS = 200
    INVALID_INPUT = 422
    SERVER_ERROR = 500

class Response[T](BaseModel):
    status_code: E_Response_Status_Code
    data: T

# --- trips/place-selection ---
MAX_TRIP_DAYS = 9  # 최대 8박 9일 (ADR-0008)


class External_Reason(BaseModel):
    """precheck 응답의 사유. 장소 선정은 has_issue만 읽으므로 모르는 값도 거부하지 않는다."""
    category: str
    detail: str


class External_Region_Forecast(BaseModel):
    region: str  # precheck 표기(예: REGION-SEOUL). E_Region으로의 매핑은 services/external_factors.py
    has_issue: bool
    reasons: list[External_Reason] = Field(default_factory=list)


class External_Forecast_Day(BaseModel):
    date: date
    regions: list[External_Region_Forecast] = Field(default_factory=list)


class External_Factors(BaseModel):
    """precheck 응답을 BE가 그대로 전달한 값(5개 지역 전체). 요청의 지역과 여행 날짜에 해당하는 것만 쓴다 (ADR-0008)."""
    generated_at: datetime | None = None
    forecasts: list[External_Forecast_Day] = Field(default_factory=list)


class Place_Selection_Request(BaseModel):
    region: E_Region
    start_date: date
    end_date: date
    members: list[Member_Survey] = Field(min_length=1, max_length=8)
    external_factors: External_Factors | None = None

    @field_validator('start_date')
    @classmethod
    def validate_start_date_not_in_past(cls, start_date_value: date) -> date:
        if start_date_value < get_service_today():
            raise ValueError("시작일은 오늘자 이후여야 합니다.")
        return start_date_value

    @field_validator('end_date')
    @classmethod
    def validate_end_date_not_before_start_date(cls, end_date_value: date, validation_info: ValidationInfo) -> date:
        start_date_value = validation_info.data.get("start_date")
        if start_date_value is None:
            return end_date_value
        if end_date_value < start_date_value:
            raise ValueError("종료일은 시작일 이후여야 합니다.")
        if (end_date_value - start_date_value).days + 1 > MAX_TRIP_DAYS:
            raise ValueError(f"여행 기간은 최대 {MAX_TRIP_DAYS}일(8박 9일)입니다.")
        return end_date_value

    @field_validator('members')
    @classmethod
    def validate_user_id_unique(cls, members_value: list[Member_Survey]) -> list[Member_Survey]:
        user_ids = [member.user.user_id for member in members_value]
        if len(user_ids) != len(set(user_ids)):
            raise ValueError('user_id는 중복될 수 없음')
        return members_value


class Place_Selection_Response_Data(BaseModel):
    # places[안][일자][장소]. 안: [0] 규칙(Spring이 재정렬하는 원본), [1] 취향, [2] 다양성 (ADR-0008)
    places: list[list[list[Place]]]



# --- trips/precheck ---

class E_Precheck_Reason_Category(str, Enum):
    """이상치 사유 대분류"""
    WEATHER_RISK = "weather_risk"
    CLOSURE_RISK = "closure_risk"


class E_Weather_Detail(str, Enum):
    """날씨 위험 상세 (ADR-0009). 단기 구간은 예보 양, 중기 구간은 강수 확률·문구로 판단하므로 이름이 다르다."""
    HEAVY_RAIN = "heavy_rain"
    HEAVY_SNOW = "heavy_snow"
    RAIN_LIKELY = "rain_likely"
    SNOW_LIKELY = "snow_likely"


class Precheck_Reason(BaseModel):
    category: E_Precheck_Reason_Category
    detail: str


class Precheck_Response_Data(BaseModel):
    """precheck 응답. 요청의 external_factors(External_Factors)와 같은 모양이라 BE가 그대로 전달한다.
    External_Factors를 상속하지 않는 것은 generated_at을 필수로 두면서 부모의 기본값 있는 필드를 덮어쓰면 타입 검사기가 거부하기 때문이다.
    generated_at은 사용한 예보 중 가장 오래된 것의 수신 시각(UTC)이다."""
    generated_at: datetime
    forecasts: list[External_Forecast_Day]


# --- trips/course-recommendation ---

class E_Course_Type(str, Enum):
    SHORTEST = "shortest"
    PREFERENCE = "preference"
    DIVERSITY = "diversity"


class Course(BaseModel):
    course_type: E_Course_Type
    places: list[Place]

class Course_Recommendation_Request(BaseModel):
    region: E_Region
    start_date: date
    end_date: date
    members: list[Member_Survey]
    affected_place_ids: list[str] = Field(default_factory=list)
    reasons: list[Precheck_Reason] = Field(default_factory=list)


class Course_Recommendation_Response_Data(BaseModel):
    courses: list[Course]