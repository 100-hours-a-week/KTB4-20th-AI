import json
from pathlib import Path

from app.trips.schemas.schemas import E_Google_Place_Type, E_Preference, E_Region

# 수집 격자 반경(m): hex grid 각 중심점 기준 반경
HEX_RADIUS_M = 2883

# 수집 지역별 hex grid 중심점 좌표 목록 ([위도, 경도])
_HEX_GRID_PATH = Path(__file__).resolve().parent / "data" / "hex_grid_points.json"
with open(_HEX_GRID_PATH, encoding="utf-8") as hex_grid_file:
    HEX_GRID_POINTS: dict[str, list[list[float]]] = json.load(hex_grid_file)

# 요청 지역 → 수집 지역명 매핑
REGION_TO_COLLECTION_AREAS: dict[E_Region, list[str]] = {
    E_Region.SEOUL: ["서울"],
    E_Region.BUSAN: ["부산"],
    E_Region.JEJU: ["제주시권", "서귀포권"],  # 하나의 선택지가 두 수집 지역에 대응
    E_Region.GYEONGJU: ["경주"],
    E_Region.JEONJU: ["전주"],
}

# 수집 지역명 → 요청 지역 매핑 (REGION_TO_COLLECTION_AREAS의 역방향)
COLLECTION_AREA_TO_REGION: dict[str, E_Region] = {
    collection_area: region
    for region, collection_areas in REGION_TO_COLLECTION_AREAS.items()
    for collection_area in collection_areas
}

# 카테고리별 문항 인덱스 매핑 (survey_result의 위치 기반 접근에 사용)
CATEGORY_INDEX_MAP_FOR_SLOT: dict[E_Preference, tuple[int, int]] = {
    E_Preference.HISTORY_CULTURE: (0, 3),
    E_Preference.NATURE_HEALING: (3, 6),
    E_Preference.FOOD: (6, 9),
    E_Preference.ACTIVITY: (10, 12),
    E_Preference.CONVENIENCE_SHOPPING: (12, 15),
}

# 야간형 ACTIVITY 문항(10번) 인덱스 범위: 저녁 이후 칸 생성 판정에만 사용함
NIGHT_ACTIVITY_INDEX_RANGE: tuple[int, int] = (9, 10)

# 야간형 ACTIVITY type: 문항 10(클럽, 노래방, 라이브 공연장 등 밤 문화)에 대응
# 저녁 이후 칸 후보로만 사용하며, 나머지 ACTIVITY type은 주간형으로 분류함
NIGHT_ACTIVITY_TYPES: frozenset[E_Google_Place_Type] = frozenset({
    E_Google_Place_Type.NIGHT_CLUB,
    E_Google_Place_Type.DANCE_HALL,
    E_Google_Place_Type.KARAOKE,
    E_Google_Place_Type.COMEDY_CLUB,
    E_Google_Place_Type.LIVE_MUSIC_VENUE,
})

# 저녁 이후 칸 생성 임계값: 문항 10 그룹 점수가 이 값을 "초과"하면 생성함
# 3은 5점 척도의 "보통"이며, 부동소수점 오차로 미세하게 초과하는 경우는 생성으로 허용함
EVENING_SLOT_THRESHOLD: float = 3.0