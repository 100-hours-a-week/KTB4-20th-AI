"""여행 미션 코드에서 학습용 취향 벡터를 만든다 (M1 T1-4, ADR-0005).

취향은 여행 전에 정해진 값이어야 하므로 방문 행동이 아니라 `TRAVEL_MISSION`(사전 선언 미션 코드)으로만 만든다.
방문 행동으로 만들면 정답이 입력에 섞인다(누수).

카테고리는 서비스 설문의 5개 중 FOOD를 뺀 4개다. FOOD에 대응하는 미션 코드가 없다.
미매핑 코드(4 시티투어, 22 SNS 인생샷, 24 신규 여행지 발굴, 25 반려동물 동반, 26 인플루언서 따라하기)는 무시한다.
"""

MISSION_CATEGORIES: dict[str, frozenset[str]] = {
    "HISTORY_CULTURE": frozenset({"3", "6", "12", "13"}),  # 역사 유적지, 문화예술/공연/전시, 드라마 촬영지, 종교/성지
    "NATURE_HEALING": frozenset({"8", "10", "21", "27", "28"}),  # 캠핑, 온천/스파, Well-ness, 친환경, 등반
    "ACTIVITY": frozenset({"2", "5", "7", "9", "11"}),  # 테마파크, 야외 스포츠, 유흥, 축제, 체험
    "CONVENIENCE_SHOPPING": frozenset({"1", "23"}),  # 쇼핑, 호캉스
}
IGNORED_MISSIONS = frozenset({"4", "22", "24", "25", "26"})


def mission_codes(raw: str) -> set[str]:
    # "2;22;24;6;8;"처럼 끝에 구분자가 붙어 있다
    return {code.strip() for code in (raw or "").split(";") if code.strip()}


def preference_vector(raw_mission: str) -> dict[str, int]:
    codes = mission_codes(raw_mission)
    return {category: int(bool(codes & members)) for category, members in MISSION_CATEGORIES.items()}


def has_preference(vector: dict[str, int]) -> bool:
    # 서비스 입력은 최소 하나가 1이어서, 모두 0인 여행은 학습에서 뺀다
    return any(vector.values())


def unknown_codes(raw_mission: str) -> set[str]:
    # 매핑표에도 무시 목록에도 없는 코드. 새 코드가 생겼는지 점검하는 용도
    known = IGNORED_MISSIONS.union(*MISSION_CATEGORIES.values())
    return mission_codes(raw_mission) - known
