from app.trips.schemas.schemas import E_Preference, Place


def assign_time_slots(places_by_category: dict[E_Preference, list[Place]], preferences: dict[E_Preference, float]) -> list[Place]:    
    # 1. 취향 순위로 non-food 카테고리 정렬(1위부터)
    ranked_categories = sorted(
        (c for c in places_by_category if c not in (E_Preference.FOOD, E_Preference.ACTIVITY)),
        key=lambda c: preferences[c],
        reverse=True,
    )

    # 2. 'FOOD'가 아닌 장소(비음식점)들을 rank 기준으로 순서대로 추가
    non_food_places = []
    for category in ranked_categories:
        non_food_places.extend(places_by_category[category])

    # 3. 'FOOD' 장소 목록, 'ACTIVITY' 장소 목록 구분
    food_places = places_by_category.get(E_Preference.FOOD, [])
    activity_places = places_by_category.get(E_Preference.ACTIVITY, [])

    # 4. result 초기화
    result: list[Place] = []

    # 5. 순서대로 삽입
    if len(non_food_places) > 0:
        result.append(non_food_places[0]) # 오전 - 1위 카테고리부터
    if len(food_places) > 0:
        result.append(food_places[0]) # 점심
    if len(non_food_places) > 1:
        result.append(non_food_places[1]) # 오후1
    if len(non_food_places) > 2:
        result.append(non_food_places[2]) # 오후2
    if len(food_places) > 1:
        result.append(food_places[1]) # 저녁
    if len(activity_places) > 0:
        result.append(activity_places[0]) # 저녁이후

    return result