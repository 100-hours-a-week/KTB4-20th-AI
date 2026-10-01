import time
from pathlib import Path
from typing import cast

import requests

from app.core.config import settings
from app.trips.constants import HEX_GRID_POINTS, HEX_RADIUS_M, COLLECTION_AREA_TO_REGION
from app.trips.schemas.schemas import E_Google_Place_Type, E_Preference

BATCH_SIZE = 20  # Nearby Search 1회 최대 결과 수
SLEEP_SECONDS = 0.5 # 레이트리밋 대기
MAX_RETRIES = 3 # 호출 자체가 실패했을 경우, 최대 3회까지 재호출

REGIONS = {
    "서울": {"center": (37.5665, 126.9780), "radius_km": 18},
    "경주": {"center": (35.8367, 129.2110), "radius_km": 13},
    "부산": {"center": (35.1796, 129.0756), "radius_km": 15},
    "전주": {"center": (35.8181703, 127.1535972), "radius_km": 8},
    "제주시권": {"center": (33.499461, 126.530167), "radius_km": 15},
    "서귀포권": {"center": (33.2541, 126.5601), "radius_km": 15},
}

# 고정 반경 적용
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
        E_Google_Place_Type.BOTANICAL_GARDEN,
        E_Google_Place_Type.CITY_PARK,
        E_Google_Place_Type.GARDEN,
        E_Google_Place_Type.HIKING_AREA,
        E_Google_Place_Type.NATIONAL_PARK,
        E_Google_Place_Type.OBSERVATION_DECK,
        E_Google_Place_Type.PARK,
        E_Google_Place_Type.PICNIC_GROUND,
        E_Google_Place_Type.STATE_PARK,
        E_Google_Place_Type.WILDLIFE_REFUGE,
    ],
    E_Preference.ACTIVITY: [
        E_Google_Place_Type.ADVENTURE_SPORTS_CENTER,
        E_Google_Place_Type.AMUSEMENT_CENTER,
        E_Google_Place_Type.AMUSEMENT_PARK,
        E_Google_Place_Type.AQUARIUM,
        E_Google_Place_Type.ZOO,
        E_Google_Place_Type.WILDLIFE_PARK,
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
        E_Google_Place_Type.MASSAGE,
        E_Google_Place_Type.MASSAGE_SPA,
        E_Google_Place_Type.SAUNA,
        E_Google_Place_Type.SPA,
        E_Google_Place_Type.WELLNESS_CENTER,
        E_Google_Place_Type.YOGA_STUDIO,
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
            cursor.execute("SELECT COUNT(*) FROM ai_places")
            result = cursor.fetchone()
        return result[0]
    finally:
        conn.close()

# insert_place_region, insert_place_category, insert_place_type
from app.trips.services.db import get_connection

def insert_place_region(place_row_id: int, region: str) -> None:
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO ai_place_regions (place_id, region) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE id = id",
                (place_row_id, region),
            )
        conn.commit()
    finally:
        conn.close()

def insert_place_category(place_row_id: int, category: str) -> None:
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO ai_place_categories (place_id, category) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE id = id",
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
                "INSERT INTO ai_place_types (place_id, type) VALUES (%s, %s)",
                (place_row_id, type_value),
            )
        conn.commit()
    finally:
        conn.close()

# insert_place
import json
import re
from datetime import UTC, date, datetime

# 컬럼명이 응답 key를 snake_case로 바꾼 것과 같은 scalar field 41개
SCALAR_FIELD_NAMES = [
    "rating", "userRatingCount",
    "adrFormatAddress", "businessStatus", "formattedAddress", "shortFormattedAddress",
    "googleMapsUri", "iconBackgroundColor", "iconMaskBaseUri", "movedPlace", "movedPlaceId",
    "primaryType", "pureServiceAreaBusiness", "utcOffsetMinutes",
    "internationalPhoneNumber", "nationalPhoneNumber", "priceLevel", "websiteUri",
    "allowsDogs", "curbsidePickup", "delivery", "dineIn", "goodForChildren", "goodForGroups",
    "goodForWatchingSports", "liveMusic", "menuForChildren", "outdoorSeating", "reservable",
    "restroom", "servesBeer", "servesBreakfast", "servesBrunch", "servesCocktails",
    "servesCoffee", "servesDessert", "servesDinner", "servesLunch", "servesVegetarianFood",
    "servesWine", "takeout",
]

# 구조 그대로 JSON 컬럼에 저장하는 field 21개
JSON_FIELD_NAMES = [
    "addressComponents", "addressDescriptor", "attributions", "consumerAlert",
    "containingPlaces", "subDestinations", "postalAddress", "entrances",
    "navigationPoints", "transitStation", "priceRange", "regularOpeningHours",
    "currentOpeningHours", "regularSecondaryOpeningHours", "currentSecondaryOpeningHours",
    "generativeSummary", "reviewSummary", "neighborhoodSummary",
    "evChargeAmenitySummary", "evChargeOptions", "fuelOptions",
]

# 컬럼명 → 응답 내 key 경로 (객체에서 값을 꺼내 펼치는 컬럼 33개)
NESTED_COLUMN_PATHS: dict[str, tuple[str, ...]] = {
    "name": ("displayName", "text"),
    "editorial_summary": ("editorialSummary", "text"),
    "latitude": ("location", "latitude"),
    "longitude": ("location", "longitude"),
    "primary_type_display_name": ("primaryTypeDisplayName", "text"),
    "google_maps_type_label": ("googleMapsTypeLabel", "text"),
    "wheelchair_accessible_parking": ("accessibilityOptions", "wheelchairAccessibleParking"),
    "wheelchair_accessible_entrance": ("accessibilityOptions", "wheelchairAccessibleEntrance"),
    "wheelchair_accessible_restroom": ("accessibilityOptions", "wheelchairAccessibleRestroom"),
    "wheelchair_accessible_seating": ("accessibilityOptions", "wheelchairAccessibleSeating"),
    "parking_free_lot": ("parkingOptions", "freeParkingLot"),
    "parking_paid_lot": ("parkingOptions", "paidParkingLot"),
    "parking_free_street": ("parkingOptions", "freeStreetParking"),
    "parking_paid_street": ("parkingOptions", "paidStreetParking"),
    "parking_valet": ("parkingOptions", "valetParking"),
    "parking_free_garage": ("parkingOptions", "freeGarageParking"),
    "parking_paid_garage": ("parkingOptions", "paidGarageParking"),
    "payment_accepts_credit_cards": ("paymentOptions", "acceptsCreditCards"),
    "payment_accepts_debit_cards": ("paymentOptions", "acceptsDebitCards"),
    "payment_accepts_cash_only": ("paymentOptions", "acceptsCashOnly"),
    "payment_accepts_nfc": ("paymentOptions", "acceptsNfc"),
    "plus_code_global": ("plusCode", "globalCode"),
    "plus_code_compound": ("plusCode", "compoundCode"),
    "time_zone_id": ("timeZone", "id"),
    "google_maps_directions_uri": ("googleMapsLinks", "directionsUri"),
    "google_maps_place_uri": ("googleMapsLinks", "placeUri"),
    "google_maps_write_review_uri": ("googleMapsLinks", "writeAReviewUri"),
    "google_maps_reviews_uri": ("googleMapsLinks", "reviewsUri"),
    "google_maps_photos_uri": ("googleMapsLinks", "photosUri"),
    "viewport_low_latitude": ("viewport", "low", "latitude"),
    "viewport_low_longitude": ("viewport", "low", "longitude"),
    "viewport_high_latitude": ("viewport", "high", "latitude"),
    "viewport_high_longitude": ("viewport", "high", "longitude"),
}


def convert_to_snake_case(camel_case_name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", camel_case_name).lower()


def get_nested_value(data: dict, *key_path: str):
    current_value = data
    for key in key_path:
        if not isinstance(current_value, dict):
            return None
        current_value = current_value.get(key)
    return current_value


def build_opening_date(opening_date_data: dict | None) -> date | None:
    if not opening_date_data:
        return None
    try:
        return date(
            opening_date_data["year"], opening_date_data["month"], opening_date_data["day"]
        )
    except (KeyError, ValueError):
        return None


def build_place_row(place_data: dict) -> dict:
    now = datetime.now(UTC)
    place_row = {
        "google_place_id": place_data["id"],
        "resource_name": place_data.get("name"),
        "created_at": now,
        "updated_at": now,
        "opening_date": build_opening_date(place_data.get("openingDate")),
    }
    for field_name in SCALAR_FIELD_NAMES:
        place_row[convert_to_snake_case(field_name)] = place_data.get(field_name)
    for column_name, key_path in NESTED_COLUMN_PATHS.items():
        place_row[column_name] = get_nested_value(place_data, *key_path)
    for field_name in JSON_FIELD_NAMES:
        field_value = place_data.get(field_name)
        place_row[convert_to_snake_case(field_name)] = (
            json.dumps(field_value, ensure_ascii=False) if field_value is not None else None
        )
    return place_row


def insert_place(place_data: dict) -> int:
    place_row = build_place_row(place_data)
    column_names = ", ".join(place_row.keys())
    placeholders = ", ".join(["%s"] * len(place_row))

    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                f"INSERT INTO ai_places ({column_names}) VALUES ({placeholders})",
                tuple(place_row.values()),
            )
            new_id = cursor.lastrowid
        conn.commit()
        return new_id
    finally:
        conn.close()

def insert_place_reviews(place_row_id: int, review_list: list[dict]) -> None:
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            for review in review_list:
                publish_time = review.get("publishTime")
                cursor.execute(
                    """
                    INSERT INTO ai_place_reviews (
                        place_id, review_name, relative_publish_time_description, rating,
                        review_text, original_text, author_display_name, author_uri,
                        author_photo_uri, publish_time, flag_content_uri, google_maps_uri
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        place_row_id,
                        review.get("name"),
                        review.get("relativePublishTimeDescription"),
                        review.get("rating"),
                        get_nested_value(review, "text", "text"),
                        get_nested_value(review, "originalText", "text"),
                        get_nested_value(review, "authorAttribution", "displayName"),
                        get_nested_value(review, "authorAttribution", "uri"),
                        get_nested_value(review, "authorAttribution", "photoUri"),
                        datetime.fromisoformat(publish_time) if publish_time else None,
                        review.get("flagContentUri"),
                        review.get("googleMapsUri"),
                    ),
                )
        conn.commit()
    finally:
        conn.close()

def get_place_row_id(google_place_id: str) -> int | None:
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT id FROM ai_places WHERE google_place_id = %s LIMIT 1",
                (google_place_id,),
            )
            result = cursor.fetchone()
        return result[0] if result is not None else None
    finally:
        conn.close()

# call_nearby_search
NEARBY_SEARCH_URL = "https://places.googleapis.com/v1/places:searchNearby"

FIELD_MASK = "*"


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
        "languageCode": "ko",
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

PROGRESS_FILE = Path(__file__).resolve().parent.parent / "data" / "collect_progress.txt"

def load_completed() -> set[str]:
    try:
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            return {line.strip() for line in f}
    except FileNotFoundError:
        return set()

def mark_completed(key: str) -> None:
    with open(PROGRESS_FILE, 'a', encoding='utf-8') as f:
        f.write(key + '\n')

def collect_places() -> None:
    # 지금까지 완료한 것 호출, 전체 호출 횟수 초기화
    completed = load_completed() # 지금까지 완료된 장소
    total_calls = 0 # 전체 호출 횟수
    consecutive_parse_failures = 0
    consecutive_call_failures = 0
    MAX_CONSECUTIVE_FAILURES = 5  # 연속 5번 이상하면 진짜 문제로 판단

    # 1. 반복문 1: HEX_GRID_POINTS의 지역을 순회
    for region_name, points in HEX_GRID_POINTS.items():
        region_value = COLLECTION_AREA_TO_REGION[region_name].value

        # 반복문 2: 지역의 세부 좌표 배열(points)를 순서대로 순회
        for i, point in enumerate(points):
            center = cast(tuple[float, float], tuple(point)) # 세부 좌표

            # 반복문 3: 5개의 취향 카테고리 순회
            for category in E_Preference:
                key = f"{region_name}_{i}_{category.value}" # 이 조합의 고유 식별자
                if key in completed:
                    continue # 이미 끝난 조합이면 건너뜀 (API 호출 자체를 안 함)

                included_types = [t.value for t in CATEGORY_TYPE_MAP[category]]

                # 3. 반복마다 Nearby Search API를 호출한다.
                #    - includedTypes: category에 대응하는 Google types 목록
                #    - locationRestriction: region의 center, radius
                #    - maxResultCount: 이번 회차에 필요한 개수(마지막 회차는 나머지만)
                #    - 호출 후 time.sleep()으로 레이트리밋 대기
                response = call_nearby_search_with_retry(
                    included_types=included_types,
                    center=center,
                    radius_m=HEX_RADIUS_M,
                    max_result_count=BATCH_SIZE
                )
                total_calls += 1
                time.sleep(SLEEP_SECONDS)

                if response is None:
                    consecutive_call_failures += 1
                    print(f"호출 실패, 완료 기록 안 함: {key} ({consecutive_call_failures}/{MAX_CONSECUTIVE_FAILURES})")
                    if consecutive_call_failures >= MAX_CONSECUTIVE_FAILURES:
                        print("연속 호출 실패, 수집을 중단합니다. 키, 권한, 네트워크를 확인하세요.")
                        return
                    continue
                consecutive_call_failures = 0

                if response is not None:
                    try:
                        # 4. 응답으로 받은 장소들마다:
                        #    - google_place_id가 이미 DB에 있는지 SELECT로 확인
                        #    - 없으면 INSERT INTO ai_places (Prepared Statement)
                        #    - INSERT INTO ai_place_categories (해당 category)
                        #    - INSERT INTO ai_place_types (응답의 types 배열 전부)
                        for place_data in response.get("places", []):
                            google_place_id = place_data["id"]

                            place_row_id = get_place_row_id(google_place_id)
                            if place_row_id is None:
                                place_row_id = insert_place(place_data)
                                for type_value in place_data.get("types", []):
                                    insert_place_type(place_row_id, type_value)
                                insert_place_reviews(place_row_id, place_data.get("reviews", []))

                            insert_place_category(place_row_id, category.value)
                            insert_place_region(place_row_id, region_value)
                    except (KeyError, AttributeError, TypeError) as e:
                        consecutive_parse_failures += 1
                        print(f"응답 구조 이상({key}): {e} — 이 조합은 건너뜀")
                        print(f"연속 이상 횟수: {consecutive_parse_failures}/{MAX_CONSECUTIVE_FAILURES}")

                        if consecutive_parse_failures >= MAX_CONSECUTIVE_FAILURES:
                            print("연속으로 응답 구조 이상 발생 — API 스펙이 바뀌었을 가능성, 수집을 중단합니다.")
                            print(f"마지막 응답 원본: {response}")
                            return  # 진짜 여기서 전체 종료
                        
                        continue # mark_completed(key)를 건너뛰고 다음 category로 이동

                    mark_completed(key)  # 성공한 조합만 완료로 기록 (실패 조합은 위의 continue로 건너뜀)

        print(f"{region_name} 완료 (누적 호출: {total_calls}회)")

    total = count_places()
    print(f"전체 수집 완료: {total}개 (총 호출 {total_calls}회)")

if __name__ == "__main__":
    collect_places()