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
    """Google Places type. 312개 전수 검토 중 확정 93개 반영."""
    # HISTORY_CULTURE (29개)
    MUSEUM = "museum"
    ART_GALLERY = "art_gallery"
    ART_MUSEUM = "art_museum"
    AUDITORIUM = "auditorium"
    CASTLE = "castle"
    CULTURAL_LANDMARK = "cultural_landmark"
    HISTORICAL_PLACE = "historical_place"
    HISTORY_MUSEUM = "history_museum"
    MONUMENT = "monument"
    PERFORMING_ARTS_THEATER = "performing_arts_theater"
    SCULPTURE = "sculpture"
    AMPHITHEATRE = "amphitheatre"
    CONCERT_HALL = "concert_hall"
    CULTURAL_CENTER = "cultural_center"
    HISTORICAL_LANDMARK = "historical_landmark"
    OPERA_HOUSE = "opera_house"
    PHILHARMONIC_HALL = "philharmonic_hall"
    CHURCH = "church"
    BUDDHIST_TEMPLE = "buddhist_temple"
    HINDU_TEMPLE = "hindu_temple"
    MOSQUE = "mosque"
    SHINTO_SHRINE = "shinto_shrine"
    SYNAGOGUE = "synagogue"
    MASSAGE = "massage"
    MASSAGE_SPA = "massage_spa"
    SAUNA = "sauna"
    SPA = "spa"
    WELLNESS_CENTER = "wellness_center"
    YOGA_STUDIO = "yoga_studio"

    # NATURE_HEALING (21개)
    BEACH = "beach"
    ISLAND = "island"
    LAKE = "lake"
    MOUNTAIN_PEAK = "mountain_peak"
    NATURE_PRESERVE = "nature_preserve"
    RIVER = "river"
    SCENIC_SPOT = "scenic_spot"
    WOODS = "woods"
    AQUARIUM = "aquarium"
    BOTANICAL_GARDEN = "botanical_garden"
    CITY_PARK = "city_park"
    GARDEN = "garden"
    HIKING_AREA = "hiking_area"
    NATIONAL_PARK = "national_park"
    OBSERVATION_DECK = "observation_deck"
    PARK = "park"
    PICNIC_GROUND = "picnic_ground"
    STATE_PARK = "state_park"
    WILDLIFE_PARK = "wildlife_park"
    WILDLIFE_REFUGE = "wildlife_refuge"
    ZOO = "zoo"

    # ACTIVITY (34개)
    ADVENTURE_SPORTS_CENTER = "adventure_sports_center"
    AMUSEMENT_CENTER = "amusement_center"
    AMUSEMENT_PARK = "amusement_park"
    BARBECUE_AREA = "barbecue_area"
    BOWLING_ALLEY = "bowling_alley"
    COMEDY_CLUB = "comedy_club"
    CYCLING_PARK = "cycling_park"
    DANCE_HALL = "dance_hall"
    FERRIS_WHEEL = "ferris_wheel"
    GO_KARTING_VENUE = "go_karting_venue"
    KARAOKE = "karaoke"
    LIVE_MUSIC_VENUE = "live_music_venue"
    MINIATURE_GOLF_COURSE = "miniature_golf_course"
    NIGHT_CLUB = "night_club"
    OFF_ROADING_AREA = "off_roading_area"
    PAINTBALL_CENTER = "paintball_center"
    ROLLER_COASTER = "roller_coaster"
    SKATEBOARD_PARK = "skateboard_park"
    VIDEO_ARCADE = "video_arcade"
    WATER_PARK = "water_park"
    ARENA = "arena"
    FISHING_CHARTER = "fishing_charter"
    FISHING_PIER = "fishing_pier"
    FISHING_POND = "fishing_pond"
    GOLF_COURSE = "golf_course"
    ICE_SKATING_RINK = "ice_skating_rink"
    INDOOR_GOLF_COURSE = "indoor_golf_course"
    RACE_COURSE = "race_course"
    SKI_RESORT = "ski_resort"
    SPORTS_ACTIVITY_LOCATION = "sports_activity_location"
    SPORTS_COMPLEX = "sports_complex"
    STADIUM = "stadium"
    SWIMMING_POOL = "swimming_pool"
    TENNIS_COURT = "tennis_court"

    # CONVENIENCE_SHOPPING (9개)
    CLOTHING_STORE = "clothing_store"
    COSMETICS_STORE = "cosmetics_store"
    DEPARTMENT_STORE = "department_store"
    FARMERS_MARKET = "farmers_market"
    FLEA_MARKET = "flea_market"
    GIFT_SHOP = "gift_shop"
    JEWELRY_STORE = "jewelry_store"
    MARKET = "market"
    SHOPPING_MALL = "shopping_mall"

    # FOOD(대표값 — 전체 159개는 별도 상수 목록 관리 예정)
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

    # 기타(제외 판정용 참고)
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