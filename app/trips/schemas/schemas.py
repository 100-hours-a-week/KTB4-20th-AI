from enum import Enum

from pydantic import BaseModel, Field


# Enum
class E_Breaker(str, Enum):
    """회피조건. 전부 '사용자가 절대 하고 싶지 않은 것'이라는 같은 성격.
    OUTDOOR_ACTIVITY만 회피 대상(야외)이 너무 많아,
    여집합(실내)으로 구현(DIRECT/INVERTED_EXCLUDE_MAP에서 처리)."""
    NOISY_PLACE = "시끄러운_곳"
    OUTDOOR_ACTIVITY = "야외_활동"
    DRINKING = "술자리"
    SEAFOOD = "해산물"

class E_Google_Place_Type(str, Enum):
    """Google Places type. 312개 전수 검토 중 확정 93개 발췌.
    Food and Drink(159개)는 전체 Enum화 대신 별도 상수 목록으로 관리 예정."""
    # HISTORY_CULTURE
    MUSEUM = "museum"
    ART_GALLERY = "art_gallery"
    ART_MUSEUM = "art_museum"
    HISTORY_MUSEUM = "history_museum"
    CONCERT_HALL = "concert_hall"
    OPERA_HOUSE = "opera_house"
    CHURCH = "church"
    BUDDHIST_TEMPLE = "buddhist_temple"
    HINDU_TEMPLE = "hindu_temple"
    MOSQUE = "mosque"
    SHINTO_SHRINE = "shinto_shrine"
    SYNAGOGUE = "synagogue"
    # NATURE_HEALING
    PARK = "park"
    NATIONAL_PARK = "national_park"
    BEACH = "beach"
    # ACTIVITY
    AMUSEMENT_PARK = "amusement_park"
    NIGHT_CLUB = "night_club"
    KARAOKE = "karaoke"
    DANCE_HALL = "dance_hall"
    VIDEO_ARCADE = "video_arcade"
    COMEDY_CLUB = "comedy_club"
    LIVE_MUSIC_VENUE = "live_music_venue"
    AMUSEMENT_CENTER = "amusement_center"
    ARENA = "arena"
    STADIUM = "stadium"
    WATER_PARK = "water_park"
    # CONVENIENCE_SHOPPING
    SHOPPING_MALL = "shopping_mall"
    MARKET = "market"
    # FOOD(대표값만 — 전체 159개는 별도 관리)
    RESTAURANT = "restaurant"
    CAFE = "cafe"
    BAKERY = "bakery"
    SEAFOOD_RESTAURANT = "seafood_restaurant"
    BAR = "bar"
    BAR_AND_GRILL = "bar_and_grill"
    COCKTAIL_BAR = "cocktail_bar"
    SPORTS_BAR = "sports_bar"
    WINE_BAR = "wine_bar"
    LOUNGE_BAR = "lounge_bar"
    HOOKAH_BAR = "hookah_bar"
    OYSTER_BAR_RESTAURANT = "oyster_bar_restaurant"
    PUB = "pub"
    IRISH_PUB = "irish_pub"
    GASTROPUB = "gastropub"
    BREWPUB = "brewpub"
    # 기타(제외 판정용)
    TOURIST_ATTRACTION = "tourist_attraction"
    POINT_OF_INTEREST = "point_of_interest"

class E_Preference(str, Enum):
    """5개 취향 카테고리. Google types 직접 그룹핑으로 정의함."""
    HISTORY_CULTURE = "HISTORY_CULTURE"
    NATURE_HEALING = "NATURE_HEALING"
    FOOD = "FOOD"
    ACTIVITY = "ACTIVITY"
    CONVENIENCE_SHOPPING = "CONVENIENCE_SHOPPING"

# --- 공통 값 객체 ---

class Display_Name(BaseModel):
    """장소 표시 이름. Google Places 응답의 displayName(LocalizedText) 그대로."""
    text: str
    languageCode: str


class Location(BaseModel):
    """위경도 좌표. Google Places 응답의 location(LatLng) 그대로."""
    latitude: float
    longitude: float

# --- 도메인 모델 ---
class User(BaseModel):
    user_id: str

class GooglePlaceData(BaseModel):
    """Google Places API 원본 필드만. Google 스펙이 바뀔 때만 변경.
    TODO: Google Places API, DB Entity 사이 타입을 적절히 반영할 것.
    Google Places API 우선, 그리고 DB Entity 차선"""
    id: str
    displayName: Display_Name
    location: Location
    types: list[E_Google_Place_Type]
    rating: float
    userRatingCount: int
    editorialSummary: str | None = None


class Place(GooglePlaceData):
    """우리 서비스가 계산해서 덧붙이는 필드. 매칭 로직이 바뀔 때만 변경."""
    selected_for: list[str] = Field(default_factory=list)
    matched_preferences: list[E_Preference] = Field(default_factory=list)


class Member_Survey(BaseModel):
    """구성원별 설문 응답 + 회피조건 + 필수방문지."""
    user: User
    survey_result: list[int] = Field(min_length=15, max_length=15)
    deal_breakers: list[E_Breaker] = Field(default_factory=list)
    must_visit: list[Place] = Field(default_factory=list)