"""로그 방문지와 DB 장소의 정렬 검증 정답 쌍을 만든다 (M2 T2-3, ADR-0003, ADR-0005).

같은 장소로 볼 수 있는 쌍: 좌표가 반경 안이고 이름이 비슷한 (로그 방문지, DB 장소).
같은 장소가 임베딩 공간에서 가까운지(검색 성공률)와 같은 코드를 받는지(코드 일치율)를 이 쌍으로 잰다.

주의: 쌍을 이름 유사도로 골랐기 때문에 이름을 쓰는 임베딩에 유리한 편향이 있다. 결과 해석에 함께 적는다.
엄격 기준만 쓴다(유사도 0.8 이상). 이름이 한쪽에 포함되는 것만으로는 쌍으로 보지 않는다.
"""

import collections
import difflib
import math
import re
import unicodedata

from scripts.semantic_id.place_text import log_visit_text
from scripts.semantic_id.text import normalize_name

RADIUS_M = 300
MIN_SIMILARITY = 0.8
_CELL = 100  # 격자 한 칸 = 위도·경도 0.01도(약 1km)


def match_key(name: str) -> str:
    # 이름 비교용 키: 정규화 후 소문자, 공백·구두점·"본점/지점" 제거
    text = unicodedata.normalize("NFC", normalize_name(name)).lower()
    return re.sub(r"[\s\(\)\[\]\-_.·,]|본점|지점", "", text)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def build_pairs(places: list[dict], visits: list[dict]) -> list[dict]:
    """places: {google_place_id, name, lat, lng}, visits: days.jsonl의 방문 기록(name, x=경도, y=위도 …).

    방문마다 반경 안에서 이름이 가장 비슷한 DB 장소 하나를 고르고, (로그 텍스트, DB 장소)가 같은 쌍은 합친다.
    """
    grid: dict[tuple[int, int], list[tuple[str, str, float, float]]] = collections.defaultdict(list)
    for p in places:
        cell = (round(p["lat"] * _CELL), round(p["lng"] * _CELL))
        grid[cell].append((p["google_place_id"], match_key(p["name"]), p["lat"], p["lng"]))

    pairs: dict[tuple[str, str], dict] = {}
    for v in visits:
        if v.get("x") is None or v.get("y") is None:
            continue
        key = match_key(v["name"])
        if not key:
            continue
        cy, cx = round(v["y"] * _CELL), round(v["x"] * _CELL)
        best: tuple[float, str, float] | None = None
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for gid, pkey, lat, lng in grid.get((cy + dy, cx + dx), []):
                    dist = haversine_m(v["y"], v["x"], lat, lng)
                    if dist > RADIUS_M:
                        continue
                    sim = difflib.SequenceMatcher(None, key, pkey).ratio()
                    if best is None or sim > best[0]:
                        best = (sim, gid, dist)
        if best is None or best[0] < MIN_SIMILARITY:
            continue
        text = log_visit_text(v)
        k = (text, best[1])
        if k not in pairs:
            pairs[k] = {"log_text": text, "google_place_id": best[1], "similarity": round(best[0], 3),
                        "distance_m": round(best[2], 1), "visits": 0}
        pairs[k]["visits"] += 1
    return sorted(pairs.values(), key=lambda r: (r["google_place_id"], r["log_text"]))
