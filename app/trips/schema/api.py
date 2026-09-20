from enum import Enum
from typing import Generic, TypeVar
from pydantic import BaseModel, Field
from schemas import Location, Member_Survey, Place

T = TypeVar("T")

class E_Response_Status_Code(int, Enum):
    SUCCESS = 200
    INVALID_INPUT = 422
    SERVER_ERROR = 500

class Response(BaseModel, Generic[T]):
    status_code: E_Response_Status_Code
    data: T

# --- trips/place-selection ---
class Place_Selection_Request(BaseModel):
    location: Location
    start_date: str
    end_date: str
    members: list[Member_Survey]


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
    location: Location


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
    location: Location
    start_date: str
    end_date: str
    members: list[Member_Survey]
    affected_place_ids: list[str] = Field(default_factory=list)
    reasons: list[Precheck_Reason] = Field(default_factory=list)


class Course_Recommendation_Response_Data(BaseModel):
    courses: list[Course]