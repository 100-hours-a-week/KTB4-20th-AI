from app.trips.schemas.schemas import E_Region, E_Preference, E_Breaker, Place

ALPHA = 0.7
BETA = 0.3

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

INVERTED_EXCLUDE_MAP: dict[E_Breaker, set[str]] = {
    E_Breaker.OUTDOOR_ACTIVITY: {
        "museum", "art_gallery", "art_museum", "history_museum",
        "concert_hall", "opera_house",
        "church", "buddhist_temple", "hindu_temple", "mosque", "shinto_shrine", "synagogue",
    },
}

REGION_TO_COLLECTION_AREAS: dict[E_Region, list[str]] = {
    E_Region.SEOUL: ["서울"],
    E_Region.BUSAN: ["부산"],
    E_Region.JEJU: ["제주시권", "서귀포권"],  # 하나의 선택지가 두 수집 지역에 대응
    E_Region.GYEONGJU: ["경주"],
    E_Region.JEONJU: ["전주"],
}

def get_DB_places_by_category(category: E_Preference, preference_score: float, limit: int) -> list[Place]:
    # TODO: DB 쿼리로 직접 처리
    #   ORDER By (preference_score * ALPHA + rating * BETA) DESC
    #   LIMIT limit
    return []

def select_places(preferences: dict[E_Preference, float], slot_counts: dict[E_Preference, int]) -> list[Place]:
    # 1. 결과를 담을 빈 리스트 result 선언
    result = []

    # 2. slot_counts를 순회하면서 category, count 선언
    for category, count in slot_counts.items():
        # (edge case) count가 0이면 제외
        if count == 0: continue

        # DB로부터 정렬·제한된 결과 호출
        called_places = get_DB_places_by_category(category, preferences[category], count)

        # result에 이어붙인다.(extend)
        result.extend(called_places)

    return result