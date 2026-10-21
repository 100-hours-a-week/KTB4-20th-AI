"""안 하나의 취향 반영도를 센다 (ADR-0012).

낮 활동 장소만 센다. 식당(FOOD)과 저녁 이후 장소는 취향 카테고리로 고른 곳이 아니라서 어느 쪽에도 넣지 않는다.
  preference_count: 매칭 카테고리가 그룹 취향 상위 2개에 속하는 곳
  discovery_count:  속하지 않는 곳. 3안이 섞은 3·4위 카테고리와, 회피 조건으로 후보가 모자라 다음 순위로 보충한 곳이 모두 들어간다

저녁 이후 장소는 type으로 알아본다. 후보 조회가 저녁 이후 풀에는 야간 type을 요구하고 낮 ACTIVITY 풀에서는 야간 type을 제외하므로(규칙·모델 경로 모두),
야간 type을 가진 ACTIVITY 장소는 저녁 이후 슬롯에서만 나온다.
"""

from collections.abc import Collection

from app.trips.schemas.schemas import E_Preference, Place
from app.trips.services.place_selection import NIGHT_ACTIVITY_TYPE_VALUES


def is_evening_place(place: Place) -> bool:
    return E_Preference.ACTIVITY in place.matched_preferences and any(
        place_type.value in NIGHT_ACTIVITY_TYPE_VALUES for place_type in place.types
    )


def count_taste_and_discovery(trip: list[list[Place]], taste_categories: Collection[E_Preference]) -> tuple[int, int]:
    """(preference_count, discovery_count). 매칭 카테고리가 없는 장소는 세지 않는다."""
    preference = discovery = 0
    for day in trip:
        for place in day:
            if not place.matched_preferences:
                continue
            category = place.matched_preferences[0]
            if category == E_Preference.FOOD or is_evening_place(place):
                continue
            if category in taste_categories:
                preference += 1
            else:
                discovery += 1
    return preference, discovery
