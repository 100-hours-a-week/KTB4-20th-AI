"""방문 기록을 하루 단위 시퀀스로 묶는다 (M1 T1-3, T1-5, ADR-0005).

하루 = 같은 `TRAVEL_ID`, 같은 `VISIT_START_YMD`. 순서는 `VISIT_ORDER`를 따른다.
집·친구집·사무실·숙소(방문지 유형 21~24)는 뺀다. 정답 신호는 만족도·재방문 의향·추천 의향이 모두 4 이상인 방문이다.
"""

import collections

from scripts.aihub_common import region_of

NON_PLACE_TYPES = frozenset({"21", "22", "23", "24"})  # 집, 친구/친지집, 사무실, 숙소
SCORE_FIELDS = ("DGSTFN", "REVISIT_INTENTION", "RCMDTN_INTENTION")  # 만족도, 재방문 의향, 추천 의향
MIN_SCORE = 4
SERVICE_REGIONS = ("서울", "부산", "경주", "전주", "제주")


def is_place_visit(visit: dict) -> bool:
    return visit["VISIT_AREA_TYPE_CD"] not in NON_PLACE_TYPES


def is_satisfied(visit: dict) -> bool:
    # 세 값이 모두 채워져 있고 모두 4 이상이어야 정답으로 인정한다
    for field in SCORE_FIELDS:
        raw = (visit.get(field) or "").strip()
        if not raw:
            return False
        try:
            if float(raw) < MIN_SCORE:
                return False
        except ValueError:
            return False
    return True


def group_days(visits: list[dict]) -> dict[tuple[str, str], list[dict]]:
    # 집·숙소와 정답이 아닌 방문을 뺀 뒤 (여행, 방문일)로 묶어 방문 순서로 정렬한다
    days: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for v in visits:
        if is_place_visit(v) and is_satisfied(v):
            days[(v["TRAVEL_ID"], v["VISIT_START_YMD"])].append(v)
    for rows in days.values():
        rows.sort(key=lambda r: int(r["VISIT_ORDER"]))
    return days


def day_region(rows: list[dict]) -> str:
    # 하루의 첫 방문 중 서비스 지역 주소로 판정되는 방문의 지역. 없으면 "기타"
    for r in rows:
        region = region_of(r["ROAD_NM_ADDR"] or r["LOTNO_ADDR"] or "")
        if region:
            return region
    return "기타"


def _float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def visit_record(v: dict) -> dict:
    # 학습 예제에 남길 칸만 고른다. 좌표는 숫자가 아니거나 없으면 None이다
    return {
        "visit_area_id": v["VISIT_AREA_ID"],
        "order": int(v["VISIT_ORDER"]),
        "name": v["VISIT_AREA_NM"].strip(),
        "road_addr": v["ROAD_NM_ADDR"].strip(),
        "lotno_addr": v["LOTNO_ADDR"].strip(),
        "x": _float_or_none(v["X_COORD"]),
        "y": _float_or_none(v["Y_COORD"]),
        "type_cd": v["VISIT_AREA_TYPE_CD"],
        "poi_id": v["POI_ID"].strip(),
        "dgstfn": float(v["DGSTFN"]),
        "revisit_intention": float(v["REVISIT_INTENTION"]),
        "rcmdtn_intention": float(v["RCMDTN_INTENTION"]),
    }
