from datetime import date
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
class Place_Selection_Request(BaseModel):
    region: E_Region
    start_date: date
    end_date: date
    members: list[Member_Survey] = Field(min_length=2, max_length=8)

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
        return end_date_value

    @field_validator('members')
    @classmethod
    def validate_user_id_unique(cls, members_value: list[Member_Survey]) -> list[Member_Survey]:
        user_ids = [member.user.user_id for member in members_value]
        if len(user_ids) != len(set(user_ids)):
            raise ValueError('user_id는 중복될 수 없음')
        return members_value


class Place_Selection_Response_Data(BaseModel):
    places: list[Place]



# --- trips/precheck ---

class E_Precheck_Reason_Category(str, Enum):
    """이상치 사유 대분류"""
    WEATHER_RISK = "weather_risk"
    CLOSURE_RISK = "closure_risk"


class Precheck_Reason(BaseModel):
    category: E_Precheck_Reason_Category
    detail: str

class Precheck_Request(BaseModel):
    region: E_Region


class Precheck_Response_Data(BaseModel):
    has_issue: bool
    affected_place_ids: list[str]
    reasons: list[Precheck_Reason]


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