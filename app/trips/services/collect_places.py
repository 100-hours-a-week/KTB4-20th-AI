import math
import time

import requests

from app.core.config import settings
from app.trips.schemas.schemas import E_Google_Place_Type, E_Preference

BATCH_SIZE = 20  # Nearby Search 1회 최대 결과 수
SLEEP_SECONDS = 0.5 # 레이트리밋 대기
MAX_RETRIES = 3

REGIONS = {
    "서울": {"center": (37.5665, 126.9780), "radius_m": 18000},
    "경주": {"center": (35.8367, 129.2110), "radius_m": 13000},
}

ALLOCATION = {
    "서울": {
        E_Preference.HISTORY_CULTURE: 100,
        E_Preference.NATURE_HEALING: 100,
        E_Preference.FOOD: 100,
        E_Preference.ACTIVITY: 100,
        E_Preference.CONVENIENCE_SHOPPING: 100,
    },
    "경주": {
        E_Preference.HISTORY_CULTURE: 150,
        E_Preference.NATURE_HEALING: 87,
        E_Preference.FOOD: 88,
        E_Preference.ACTIVITY: 88,
        E_Preference.CONVENIENCE_SHOPPING: 87,
    },
}

CATEGORY_TYPE_MAP: dict[E_Preference, list[E_Google_Place_Type]] = {
    E_Preference.HISTORY_CULTURE: [
        E_Google_Place_Type.MUSEUM,
        E_Google_Place_Type.ART_GALLERY,
        E_Google_Place_Type.ART_MUSEUM,
        E_Google_Place_Type.AUDITORIUM,
        E_Google_Place_Type.CASTLE,
        E_Google_Place_Type.CULTURAL_LANDMARK,
        E_Google_Place_Type.HISTORICAL_PLACE,
        E_Google_Place_Type.HISTORY_MUSEUM,
        E_Google_Place_Type.MONUMENT,
        E_Google_Place_Type.PERFORMING_ARTS_THEATER,
        E_Google_Place_Type.SCULPTURE,
        E_Google_Place_Type.AMPHITHEATRE,
        E_Google_Place_Type.CONCERT_HALL,
        E_Google_Place_Type.CULTURAL_CENTER,
        E_Google_Place_Type.HISTORICAL_LANDMARK,
        E_Google_Place_Type.OPERA_HOUSE,
        E_Google_Place_Type.PHILHARMONIC_HALL,
        E_Google_Place_Type.CHURCH,
        E_Google_Place_Type.BUDDHIST_TEMPLE,
        E_Google_Place_Type.HINDU_TEMPLE,
        E_Google_Place_Type.MOSQUE,
        E_Google_Place_Type.SHINTO_SHRINE,
        E_Google_Place_Type.SYNAGOGUE,
        E_Google_Place_Type.MASSAGE,
        E_Google_Place_Type.MASSAGE_SPA,
        E_Google_Place_Type.SAUNA,
        E_Google_Place_Type.SPA,
        E_Google_Place_Type.WELLNESS_CENTER,
        E_Google_Place_Type.YOGA_STUDIO,
    ],
    E_Preference.NATURE_HEALING: [
        E_Google_Place_Type.BEACH,
        E_Google_Place_Type.ISLAND,
        E_Google_Place_Type.LAKE,
        E_Google_Place_Type.MOUNTAIN_PEAK,
        E_Google_Place_Type.NATURE_PRESERVE,
        E_Google_Place_Type.RIVER,
        E_Google_Place_Type.SCENIC_SPOT,
        E_Google_Place_Type.WOODS,
        E_Google_Place_Type.AQUARIUM,
        E_Google_Place_Type.BOTANICAL_GARDEN,
        E_Google_Place_Type.CITY_PARK,
        E_Google_Place_Type.GARDEN,
        E_Google_Place_Type.HIKING_AREA,
        E_Google_Place_Type.NATIONAL_PARK,
        E_Google_Place_Type.OBSERVATION_DECK,
        E_Google_Place_Type.PARK,
        E_Google_Place_Type.PICNIC_GROUND,
        E_Google_Place_Type.STATE_PARK,
        E_Google_Place_Type.WILDLIFE_PARK,
        E_Google_Place_Type.WILDLIFE_REFUGE,
        E_Google_Place_Type.ZOO,
    ],
    E_Preference.ACTIVITY: [
        E_Google_Place_Type.ADVENTURE_SPORTS_CENTER,
        E_Google_Place_Type.AMUSEMENT_CENTER,
        E_Google_Place_Type.AMUSEMENT_PARK,
        E_Google_Place_Type.BARBECUE_AREA,
        E_Google_Place_Type.BOWLING_ALLEY,
        E_Google_Place_Type.COMEDY_CLUB,
        E_Google_Place_Type.CYCLING_PARK,
        E_Google_Place_Type.DANCE_HALL,
        E_Google_Place_Type.FERRIS_WHEEL,
        E_Google_Place_Type.GO_KARTING_VENUE,
        E_Google_Place_Type.KARAOKE,
        E_Google_Place_Type.LIVE_MUSIC_VENUE,
        E_Google_Place_Type.MINIATURE_GOLF_COURSE,
        E_Google_Place_Type.NIGHT_CLUB,
        E_Google_Place_Type.OFF_ROADING_AREA,
        E_Google_Place_Type.PAINTBALL_CENTER,
        E_Google_Place_Type.ROLLER_COASTER,
        E_Google_Place_Type.SKATEBOARD_PARK,
        E_Google_Place_Type.VIDEO_ARCADE,
        E_Google_Place_Type.WATER_PARK,
        E_Google_Place_Type.ARENA,
        E_Google_Place_Type.FISHING_CHARTER,
        E_Google_Place_Type.FISHING_PIER,
        E_Google_Place_Type.FISHING_POND,
        E_Google_Place_Type.GOLF_COURSE,
        E_Google_Place_Type.ICE_SKATING_RINK,
        E_Google_Place_Type.INDOOR_GOLF_COURSE,
        E_Google_Place_Type.RACE_COURSE,
        E_Google_Place_Type.SKI_RESORT,
        E_Google_Place_Type.SPORTS_ACTIVITY_LOCATION,
        E_Google_Place_Type.SPORTS_COMPLEX,
        E_Google_Place_Type.STADIUM,
        E_Google_Place_Type.SWIMMING_POOL,
        E_Google_Place_Type.TENNIS_COURT,
    ],
    E_Preference.CONVENIENCE_SHOPPING: [
        E_Google_Place_Type.CLOTHING_STORE,
        E_Google_Place_Type.COSMETICS_STORE,
        E_Google_Place_Type.DEPARTMENT_STORE,
        E_Google_Place_Type.FARMERS_MARKET,
        E_Google_Place_Type.FLEA_MARKET,
        E_Google_Place_Type.GIFT_SHOP,
        E_Google_Place_Type.JEWELRY_STORE,
        E_Google_Place_Type.MARKET,
        E_Google_Place_Type.SHOPPING_MALL,
    ],
    E_Preference.FOOD: [
        E_Google_Place_Type.RESTAURANT,
        E_Google_Place_Type.CAFE,
        E_Google_Place_Type.BAKERY,
        E_Google_Place_Type.SEAFOOD_RESTAURANT,
        E_Google_Place_Type.BAR,
        E_Google_Place_Type.BAR_AND_GRILL,
        E_Google_Place_Type.COCKTAIL_BAR,
        E_Google_Place_Type.SPORTS_BAR,
        E_Google_Place_Type.WINE_BAR,
        E_Google_Place_Type.LOUNGE_BAR,
        E_Google_Place_Type.HOOKAH_BAR,
        E_Google_Place_Type.OYSTER_BAR_RESTAURANT,
        E_Google_Place_Type.PUB,
        E_Google_Place_Type.IRISH_PUB,
        E_Google_Place_Type.GASTROPUB,
        E_Google_Place_Type.BREWPUB,
    ],
}

# count_places
def count_places() -> int:
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM places")
            result = cursor.fetchone()
        return result[0]
    finally:
        conn.close()

# insert_place_category, insert_place_type
from app.trips.services.db import get_connection


def insert_place_category(place_row_id: int, category: str) -> None:
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO place_categories (place_id, category) VALUES (%s, %s)",
                (place_row_id, category),
            )
        conn.commit()
    finally:
        conn.close()


def insert_place_type(place_row_id: int, type_value: str) -> None:
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO place_types (place_id, type) VALUES (%s, %s)",
                (place_row_id, type_value),
            )
        conn.commit()
    finally:
        conn.close()

# insert_place
from datetime import datetime


def insert_place(place_data: dict) -> int:
    now = datetime.now()

    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO places (
                    google_place_id, name, rating, user_rating_count,
                    editorial_summary, latitude, longitude,
                    created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    place_data["id"],
                    place_data.get("displayName", {}).get("text"),
                    place_data.get("rating"),
                    place_data.get("userRatingCount"),
                    place_data.get("editorialSummary", {}).get("text")
                        if place_data.get("editorialSummary") else None,
                    place_data.get("location", {}).get("latitude"),
                    place_data.get("location", {}).get("longitude"),
                    now,
                    now,
                ),
            )
            new_id = cursor.lastrowid
        conn.commit()
        return new_id
    finally:
        conn.close()

# place_exists

def place_exists(google_place_id: str) -> bool:
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT 1 FROM places WHERE google_place_id = %s LIMIT 1",
                (google_place_id,),
            )
            result = cursor.fetchone()
        return result is not None
    finally:
        conn.close()

# call_nearby_search
NEARBY_SEARCH_URL = "https://places.googleapis.com/v1/places:searchNearby"

FIELD_MASK = (
    "places.id,places.displayName,places.location,"
    "places.types,places.rating,places.userRatingCount,places.editorialSummary"
)


def call_nearby_search(
    included_types: list[str],
    center: tuple[float, float],
    radius_m: int,
    max_result_count: int,
) -> dict:
    latitude, longitude = center

    body = {
        "includedTypes": included_types,
        "maxResultCount": max_result_count,
        "locationRestriction": {
            "circle": {
                "center": {"latitude": latitude, "longitude": longitude},
                "radius": float(radius_m),
            }
        },
    }

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": settings.google_places_api_key,
        "X-Goog-FieldMask": FIELD_MASK,
    }

    response = requests.post(NEARBY_SEARCH_URL, json=body, headers=headers, timeout=10)
    response.raise_for_status()
    return response.json()

# call_nearby_search_with_retry
def call_nearby_search_with_retry(
    included_types: list[str],
    center: tuple[float, float],
    radius_m: int,
    max_result_count: int,
) -> dict | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return call_nearby_search(included_types, center, radius_m, max_result_count)
        except requests.exceptions.RequestException as e:
            wait = 2 ** attempt  # 지수 백오프: 2초, 4초, 8초
            print(f"호출 실패({attempt}/{MAX_RETRIES}): {e} — {wait}초 후 재시도")
            time.sleep(wait)

    print(f"{MAX_RETRIES}회 재시도 후에도 실패 — 이 배치는 건너뜀")
    return None

def collect_places() -> None:
    # 1. REGIONS와 ALLOCATION을 이중 for문으로 순회한다 (region, category, count)
    for region_name, region_info in REGIONS.items():
        center = region_info["center"]
        radius_m = region_info["radius_m"]

        for category, count in ALLOCATION[region_name].items():
            included_types = [t.value for t in CATEGORY_TYPE_MAP[category]]

            num_batches = math.ceil(count / BATCH_SIZE)
            collected = 0

            # 2. count를 BATCH_SIZE(20)로 나눠 몇 번 호출해야 하는지 계산한다.
            #    (예: 150개 필요하면 8번 호출: 20*7=140 + 마지막 10개)
            for batch_idx in range(num_batches):
                remaining = count - collected
                this_batch_size = min(BATCH_SIZE, remaining)
                if this_batch_size <= 0:
                    break

                # 3. 반복마다 Nearby Search API를 호출한다.
                #    - includedTypes: category에 대응하는 Google types 목록
                #    - locationRestriction: region의 center, radius
                #    - maxResultCount: 이번 회차에 필요한 개수(마지막 회차는 나머지만)
                #    - 호출 후 time.sleep()으로 레이트리밋 대기
                response = call_nearby_search_with_retry(
                    included_types=included_types,
                    center=center,
                    radius_m=radius_m,
                    max_result_count=this_batch_size,
                )
                time.sleep(SLEEP_SECONDS)

                if response is None:
                    continue  # 이 배치는 건너뛰고 다음 배치로

                # 4. 응답으로 받은 장소들마다:
                #    - google_place_id가 이미 DB에 있는지 SELECT로 확인
                #    - 없으면 INSERT INTO places (Prepared Statement)
                #    - INSERT INTO place_categories (해당 category)
                #    - INSERT INTO place_types (응답의 types 배열 전부)
                for place_data in response.get("places", []):
                    google_place_id = place_data["id"]

                    if place_exists(google_place_id):
                        continue

                    place_row_id = insert_place(place_data)
                    insert_place_category(place_row_id, category.value)

                    for type_value in place_data.get("types", []):
                        insert_place_type(place_row_id, type_value)

                    collected += 1

            print(f"{region_name} - {category.value}: {collected}/{count} 수집 완료")

    # 5. 전체 반복이 끝나면 SELECT COUNT(*) FROM places로 최종 개수 로그 출력
    total = count_places()
    print(f"전체 수집 완료: {total}개")