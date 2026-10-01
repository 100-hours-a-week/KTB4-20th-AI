from app.trips.constants import (
    HEX_GRID_POINTS,
    HEX_RADIUS_M,
    NIGHT_ACTIVITY_TYPES,
    REGION_TO_COLLECTION_AREAS,
)
from app.trips.schemas.schemas import (
    Display_Name,
    E_Breaker,
    E_Google_Place_Type,
    E_Preference,
    E_Region,
    Editorial_Summary,
    Location,
    Member_Survey,
    Place,
)
from app.trips.services.db import get_connection
from app.trips.services.preference import find_matched_members
from app.trips.services.slot_plan import E_Time_Slot, Slot_Plan

# DB 조회 기준 가중치 (RATINGS_WEIGHT는 ratings 가중치, USER_RATING_COUNT_WEIGHT는 userRatingCount 가중치
RATINGS_WEIGHT = 0.6
USER_RATING_COUNT_WEIGHT = 0.4

# 제외 항목 (목록에 있으면 제외)
DIRECT_EXCLUDE_MAP: dict[E_Breaker, set[str]] = {
    E_Breaker.SEAFOOD: {"seafood_restaurant"},
    E_Breaker.NOISY_PLACE: {
        "night_club", "amusement_park", "karaoke", "dance_hall",
        "video_arcade", "comedy_club", "live_music_venue",
        "amusement_center", "arena", "stadium", "water_park",
    },
    E_Breaker.RELIGIOUS_FACILITY: {
        "church", "buddhist_temple", "hindu_temple",
        "mosque", "shinto_shrine", "synagogue",
    },
    E_Breaker.ANIMAL_FACILITY: {
        "aquarium", "zoo", "wildlife_park", "wildlife_refuge",
    },
    E_Breaker.HEIGHT_AVERSION: {
        "observation_deck", "ferris_wheel", "roller_coaster",
    },
    E_Breaker.OUTDOOR_ACTIVITY: {
        # NATURE_HEALING(18개) - 전부 야외
        "beach", "island", "lake", "mountain_peak", "nature_preserve",
        "river", "scenic_spot", "woods", "botanical_garden", "city_park",
        "garden", "hiking_area", "national_park", "observation_deck",
        "park", "picnic_ground", "state_park", "wildlife_refuge",
        # ACTIVITY(37개 중 야외 확정분)
        "adventure_sports_center", "amusement_park", "zoo", "wildlife_park",
        "barbecue_area", "cycling_park", "ferris_wheel", "off_roading_area",
        "roller_coaster", "skateboard_park", "water_park",
        "fishing_charter", "fishing_pier", "fishing_pond",
        "golf_course", "race_course", "ski_resort",
        # 애매했던 것 중 야외로 확정
        "go_karting_venue", "miniature_golf_course", "paintball_center",
        "arena", "sports_activity_location", "sports_complex",
        "stadium", "tennis_court",
    },
}

def get_DB_places_by_category(
    category: E_Preference,
    limit: int,
    region: E_Region,
    excluded_types: set[str] | None = None,
    required_types: set[str] | None = None,
) -> list[Place]:
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            # 1단계: 필터링·정렬된 장소 목록 조회 (id만 걸러내기용 JOIN, type 컬럼은 안 가져옴)
            query = """
                SELECT DISTINCT p.id, p.google_place_id, p.name, p.rating,
                    p.user_rating_count, p.editorial_summary,
                    p.latitude, p.longitude
                FROM ai_places p
                JOIN ai_place_categories pc ON p.id = pc.place_id
                WHERE pc.category = %s
                    AND p.rating IS NOT NULL
                    AND p.user_rating_count IS NOT NULL
            """
            params: list = [category.value]

            # 지역 필터: E_Region -> 수집지역명 리스트 -> 각 지역의 중심좌표+반경으로 OR 조건 구성
            collection_areas = REGION_TO_COLLECTION_AREAS[region]
            point_conditions = []
            for area_name in collection_areas:
                for point in HEX_GRID_POINTS[area_name]:
                    lat, lon = point
                    point_conditions.append(
                        "ST_Distance_Sphere(POINT(p.longitude, p.latitude), POINT(%s, %s)) <= %s"
                    )
                    params.extend([lon, lat, HEX_RADIUS_M])

            query += f" AND ({' OR '.join(point_conditions)})"

            if excluded_types:
                query += """
                    AND p.id NOT IN (
                        SELECT place_id FROM ai_place_types
                        WHERE type IN ({})
                    )
                """.format(", ".join(["%s"] * len(excluded_types)))
                params.extend(excluded_types)

            # 포함 필터: 지정한 type 중 하나 이상을 가진 장소만 조회 (예: 저녁 이후 슬롯의 야간형 ACTIVITY)
            if required_types:
                query += """
                    AND p.id IN (
                        SELECT place_id FROM ai_place_types
                        WHERE type IN ({})
                    )
                """.format(", ".join(["%s"] * len(required_types)))
                params.extend(required_types)

            query += f"""
                ORDER BY (p.rating * {RATINGS_WEIGHT}
                    + LOG(p.user_rating_count + 1) * {USER_RATING_COUNT_WEIGHT}) DESC
                LIMIT %s
            """
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            if not rows:
                return []

            # 2단계: 위에서 뽑힌 장소들의 전체 type 목록을 별도로 조회
            place_ids = [row[0] for row in rows]
            placeholders = ", ".join(["%s"] * len(place_ids))
            cursor.execute(
                f"SELECT place_id, type FROM ai_place_types WHERE place_id IN ({placeholders})",
                place_ids,
            )
            type_rows = cursor.fetchall()

        # place_id별로 type들을 묶음
        types_by_place_id: dict[int, list[str]] = {}
        for place_id, type_value in type_rows:
            types_by_place_id.setdefault(place_id, []).append(type_value)

        return [
            Place(
                id=row[1],
                displayName=Display_Name(text=row[2], languageCode="ko"),
                location=Location(latitude=row[6], longitude=row[7]),
                types=[
                    E_Google_Place_Type(t)
                    for t in types_by_place_id.get(row[0], [])
                    if t in E_Google_Place_Type._value2member_map_
                ],
                rating=row[3],
                userRatingCount=row[4],
                editorialSummary=Editorial_Summary(text=row[5], languageCode="ko") if row[5] else None,
            )
            for row in rows
        ]
    finally:
        conn.close()

NIGHT_ACTIVITY_TYPE_VALUES: set[str] = {place_type.value for place_type in NIGHT_ACTIVITY_TYPES}

# 주간 카테고리 1개에서 필요할 수 있는 최대 장소 수 (주간 슬롯 전체 수)
DAYTIME_SLOT_COUNT = 3

def _build_fallback_order(
    short_category: E_Preference, daytime_category_ranking: tuple[E_Preference, ...]
) -> list[E_Preference]:
    """방식 A: 부족한 카테고리의 다음 순위부터 순차 보충. 마지막 순위 다음은 1위부터 다시 탐색."""
    short_index = daytime_category_ranking.index(short_category)
    return list(daytime_category_ranking[short_index + 1:]) + list(daytime_category_ranking[:short_index])

def select_places_by_slot_plan(
    slot_plan: Slot_Plan,
    deal_breakers: list[E_Breaker],
    region: E_Region,
    members: list[Member_Survey],
) -> list[Place]:
    direct_excluded: set[str] = set()
    for breaker in deal_breakers:
        direct_excluded |= DIRECT_EXCLUDE_MAP.get(breaker, set())

    # 카테고리별 후보 목록 (조회는 필요한 카테고리만, 카테고리당 1회)
    daytime_pools: dict[E_Preference, list[Place]] = {}

    def get_daytime_pool(category: E_Preference) -> list[Place]:
        if category not in daytime_pools:
            excluded_types = direct_excluded | NIGHT_ACTIVITY_TYPE_VALUES if category == E_Preference.ACTIVITY else direct_excluded
            daytime_pools[category] = get_DB_places_by_category(
                category, DAYTIME_SLOT_COUNT, region, excluded_types=excluded_types,
            )
        return daytime_pools[category]

    food_slot_count = sum(1 for item in slot_plan.items if item.category == E_Preference.FOOD)
    food_pool = get_DB_places_by_category(
        E_Preference.FOOD, food_slot_count, region, excluded_types=direct_excluded,
    )

    has_evening_slot = any(item.time_slot == E_Time_Slot.EVENING for item in slot_plan.items)
    evening_pool = get_DB_places_by_category(
        E_Preference.ACTIVITY, 1, region,
        excluded_types=direct_excluded, required_types=NIGHT_ACTIVITY_TYPE_VALUES,
    ) if has_evening_slot else []

    matched_members_by_category: dict[E_Preference, list[str]] = {}

    def assign(place: Place, category: E_Preference) -> Place:
        if category not in matched_members_by_category:
            matched_members_by_category[category] = find_matched_members(members, category)
        place.matched_preferences = [category]
        place.selected_for = matched_members_by_category[category]
        return place

    result: list[Place] = []
    for item in slot_plan.items:
        # 저녁 이후: 야간형 후보가 없으면 슬롯 제거 (5슬롯 강등)
        if item.time_slot == E_Time_Slot.EVENING:
            if evening_pool:
                result.append(assign(evening_pool.pop(0), E_Preference.ACTIVITY))
            continue

        # 점심, 저녁: FOOD 고정, 보충 없음
        if item.category == E_Preference.FOOD:
            if food_pool:
                result.append(assign(food_pool.pop(0), E_Preference.FOOD))
            continue

        # 주간: 해당 카테고리 후보가 없으면 방식 A로 다음 순위 카테고리에서 보충
        for category in [item.category, *_build_fallback_order(item.category, slot_plan.daytime_category_ranking)]:
            pool = get_daytime_pool(category)
            if pool:
                result.append(assign(pool.pop(0), category))
                break

    return result