from app.trips.schema.schemas import E_Preference, Place

ALPHA = 0.7
BETA = 0.3

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