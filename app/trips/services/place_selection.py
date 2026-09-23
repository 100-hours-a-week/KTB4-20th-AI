from app.trips.services.db import get_connection
from app.trips.schemas.schemas import E_Region, E_Preference, E_Breaker, Place, E_Google_Place_Type, Display_Name, Location, LocalizedText

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
}

# 제외 항목 (목록에 있는 것만 통과, 야외 활동이 많아서 여집합으로 계산)
INVERTED_EXCLUDE_MAP: dict[E_Breaker, set[str]] = {
    E_Breaker.OUTDOOR_ACTIVITY: {
        "museum", "art_gallery", "art_museum", "history_museum",
        "concert_hall", "opera_house",
        "church", "buddhist_temple", "hindu_temple", "mosque", "shinto_shrine", "synagogue",
    },
}

# 지역
REGION_TO_COLLECTION_AREAS: dict[E_Region, list[str]] = {
    E_Region.SEOUL: ["서울"],
    E_Region.BUSAN: ["부산"],
    E_Region.JEJU: ["제주시권", "서귀포권"],  # 하나의 선택지가 두 수집 지역에 대응
    E_Region.GYEONGJU: ["경주"],
    E_Region.JEONJU: ["전주"],
}

def get_DB_places_by_category(
    category: E_Preference,
    preference_score: float,
    limit: int,
    excluded_types: set[str] | None = None,
    allowed_types: set[str] | None = None,
) -> list[Place]:
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            # 1단계: 필터링·정렬된 장소 목록 조회 (id만 걸러내기용 JOIN, type 컬럼은 안 가져옴)
            query = """
                SELECT DISTINCT p.id, p.google_place_id, p.name, p.rating,
                       p.user_rating_count, p.editorial_summary,
                       p.latitude, p.longitude
                FROM places p
                JOIN place_categories pc ON p.id = pc.place_id
                WHERE pc.category = %s
            """
            params: list = [category.value]

            if excluded_types or allowed_types:
                query += """
                    AND p.id NOT IN (
                        SELECT place_id FROM place_types
                        WHERE type IN ({})
                    )
                """.format(", ".join(["%s"] * len(excluded_types))) if excluded_types else ""
                if excluded_types:
                    params.extend(excluded_types)

            if allowed_types:
                query += """
                    AND p.id IN (
                        SELECT place_id FROM place_types
                        WHERE type IN ({})
                    )
                """.format(", ".join(["%s"] * len(allowed_types)))
                params.extend(allowed_types)

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
                f"SELECT place_id, type FROM place_types WHERE place_id IN ({placeholders})",
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
                editorialSummary=LocalizedText(text=row[5], languageCode="ko") if row[5] else None,
            )
            for row in rows
        ]
    finally:
        conn.close()

def select_places(
    preferences: dict[E_Preference, float],
    slot_counts: dict[E_Preference, int],
    deal_breakers: list[E_Breaker],
) -> dict[E_Preference, list[Place]]:
    direct_excluded: set[str] = set()
    inverted_allowed: set[str] = set()  # "이 목록에 있는 것만 통과"(여집합 방식)
    has_inverted = False

    for breaker in deal_breakers:
        direct_excluded |= DIRECT_EXCLUDE_MAP.get(breaker, set())
        if breaker in INVERTED_EXCLUDE_MAP:
            inverted_allowed |= INVERTED_EXCLUDE_MAP[breaker]
            has_inverted = True

    result: dict[E_Preference, list[Place]] = {c: [] for c in E_Preference}

    for category, count in slot_counts.items():
        if count <= 0:
            continue
        db_places = get_DB_places_by_category(
            category, preferences[category], count,
            excluded_types=direct_excluded,
            allowed_types=inverted_allowed if has_inverted else None,
        )
        result[category] = db_places

    return result