from app.trips.schemas.schemas import E_Preference, Place

def assign_time_slots(places_by_category: dict[E_Preference, list[Place]]) -> list[Place]:
    # 1. 'FOOD'가 아닌 장소(비음식점)들을 순서대로 추가
    non_food_places = []
    for category, places in places_by_category.items():
        if category != 'FOOD' and category != 'ACTIVITY':
            non_food_places.extend(places)
            
    # 2. 비음식점 리스트로 초기화
    result = non_food_places.copy()

    # 3. 'FOOD' 장소는 점심, 저녁에 삽입
    food_places = places_by_category.get(E_Preference.FOOD, [])
    if len(food_places) > 0:
        result.insert(1, food_places[0])
    if len(food_places) > 1:
        result.insert(4, food_places[1])

    # 4. 'ACTIVITY' 장소는 있을 경우 마지막에 삽입
    activity_places = places_by_category.get(E_Preference.ACTIVITY, [])
    if len(activity_places) > 1:
        result.append(activity_places[0])

    return result