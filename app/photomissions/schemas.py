from typing import Literal

from pydantic import BaseModel, Field

# TODO: place-selection 쪽 타입 확정되면 대체
type T_Preference = str


# generate

class DisplayName(BaseModel):
    text: str
    languageCode: str


class MissionPlace(BaseModel):
    id: str  # place-selection 응답의 Place.id
    displayName: DisplayName  # place-selection 응답의 Place.displayName
    selected_for: list[str]  # 빈 배열 허용
    matched_preferences: list[T_Preference]  # 빈 배열 허용


class PhotoMissionGenerateRequest(BaseModel):
    itinerary_id: str
    places: list[MissionPlace] = Field(min_length=1)


class Mission(BaseModel):
    mission_id: str
    place_id: str
    description: str
    scope: Literal["PERSONAL", "GROUP"]
    primary_category: T_Preference | None  # matched_preferences[0], 비어 있으면 null


class PhotoMissionGenerateResponse(BaseModel):
    missions: list[Mission]


# verify

class Coordinates(BaseModel):
    latitude: float
    longitude: float


class VerifyRequest(BaseModel):
    image_url: str
    place_id: str
    place_name: str
    place_coordinates: Coordinates
    mission_description: str
    photo_coordinates: Coordinates | None = None  # BE가 업로드 시 사진 EXIF에서 뽑은 좌표, 없으면 null


class DetectedLabel(BaseModel):
    name: str  # 사진에서 알아본 것 (일반명사)
    matched: bool  # 미션 조건과 관계있는지


class VerifyResponse(BaseModel):
    result: Literal["success", "retry", "fail"]
    reason: Literal["location_mismatch"] | None = None
    match_score: float = Field(ge=0, le=100)
    detected_labels: list[DetectedLabel]  # 빈 배열 허용 (GPS 조기 반려 시 빈 배열)
    landmark_confidence: float | None = Field(ge=0, le=100)  # 인식 신뢰도가 임계값 미만이면 null
    retry_hint: str | None  # retry가 아니면 null


class VlmResult(BaseModel):
    # 필드 순서가 생성 순서다. 관찰 → 인식 → 판단 → 조언 차례

    detected_labels: list[DetectedLabel]  # 관찰: 사진에 무엇이 보이는가
    landmark_confidence: float = Field(ge=0, le=100)  # 인식: 그것이 목표 장소인가
    match_score: float = Field(ge=0, le=100)  # 판단: 미션대로 찍혔는가
    retry_hint: str  # 조언: 항상 생성. 노출 여부는 7번(build_response)이 정한다


class MissionDescription(BaseModel):
    description: str
    scope: Literal["PERSONAL", "GROUP"]  # 한 사람 사진이 다른 사람 몫을 대신할 수 있으면 GROUP
