"""임베딩에 넣을 장소 텍스트를 만든다 (M2 T2-1, ADR-0003).

DB 장소와 로그 방문지가 같은 장소일 때 같은 코드를 받으려면 양쪽에서 같은 규칙으로 같은 칸만 써야 한다.
쓰는 칸: 정규화한 이름, 시도·시군구, 서비스 카테고리 라벨. 좌표와 평점 같은 값은 넣지 않는다.

  이름 | 시도 시군구 | 카테고리     예: "경복궁 | 서울 종로구 | 역사·문화"

값이 없으면 그 칸을 뺀다. 로그 방문의 약 17.5%는 주소가 없고, 유형 코드 6개는 카테고리 라벨이 없다.
"""

import unicodedata

from scripts.semantic_id.text import category_for_type, normalize_name

SEPARATOR = " | "

# 서비스 카테고리 → 텍스트에 쓰는 한글 라벨. 영문 코드보다 한국어 이름과 같은 언어라 임베딩이 가깝게 본다
CATEGORY_LABELS = {
    "FOOD": "음식",
    "NATURE_HEALING": "자연·힐링",
    "HISTORY_CULTURE": "역사·문화",
    "ACTIVITY": "액티비티",
    "CONVENIENCE_SHOPPING": "편의·쇼핑",
}
CATEGORY_ORDER = tuple(CATEGORY_LABELS)  # 여러 카테고리일 때 이 순서로 쓴다

# 시도 이름: DB는 "서울특별시"처럼 정식 이름이고 로그는 "서울"처럼 줄인 이름이 섞여 있어 줄인 이름으로 맞춘다
PROVINCES = {
    "서울특별시": "서울", "서울": "서울",
    "부산광역시": "부산", "부산": "부산",
    "대구광역시": "대구", "대구": "대구",
    "인천광역시": "인천", "인천": "인천",
    "광주광역시": "광주", "광주": "광주",
    "대전광역시": "대전", "대전": "대전",
    "울산광역시": "울산", "울산": "울산",
    "세종특별자치시": "세종", "세종": "세종",
    "경기도": "경기", "경기": "경기",
    "강원특별자치도": "강원", "강원도": "강원", "강원": "강원",
    "충청북도": "충북", "충북": "충북",
    "충청남도": "충남", "충남": "충남",
    "전북특별자치도": "전북", "전라북도": "전북", "전북": "전북",
    "전라남도": "전남", "전남": "전남",
    "경상북도": "경북", "경북": "경북",
    "경상남도": "경남", "경남": "경남",
    "제주특별자치도": "제주", "제주도": "제주", "제주": "제주",
}


# 시도가 빠진 주소("대한민국 경주시 …")를 위한 시군 → 시도. DB 주소에서 시도 없이 나타난 시군만 둔다
CITY_PROVINCE = {
    "경주시": "경북", "제주시": "제주", "서귀포시": "제주", "전주시": "전북", "완주군": "전북",
    "고양시": "경기", "김해시": "경남", "양산시": "경남", "양주시": "경기", "과천시": "경기",
    "구리시": "경기", "부천시": "경기", "안양시": "경기", "남양주시": "경기", "광명시": "경기",
}
_SUFFIXES = ("시", "군", "구")


def _city_after(tokens: list[str]) -> str:
    # 시도 바로 뒤의 시군구. 일반구(전주시 완산구)가 있으면 함께 쓴다
    if not tokens or not tokens[0].endswith(_SUFFIXES):
        return ""
    city = tokens[0]
    if city.endswith("시") and len(tokens) > 1 and tokens[1].endswith("구"):
        return f"{city} {tokens[1]}"
    return city


def _city_before(tokens: list[str]) -> str:
    # 주소가 거꾸로 적힌 경우("… 동안구 안양시 경기도")의 시도 앞쪽 시군구
    for i in range(len(tokens) - 1, -1, -1):
        if tokens[i].endswith(_SUFFIXES):
            if tokens[i].endswith("시") and i > 0 and tokens[i - 1].endswith("구"):
                return f"{tokens[i]} {tokens[i - 1]}"
            return tokens[i]
    return ""


def district_of(address: str) -> str:
    """주소에서 "시도 시군구"를 뽑는다. 못 뽑으면 빈 문자열.

    "대한민국 전북특별자치도 전주시 완산구 태조로 51" → "전북 전주시 완산구"
    "서울 종로구 사직로 161" → "서울 종로구"
    "대한민국 경주시 황남동" (시도 없음) → "경북 경주시"
    "… 동안구 안양시 경기도 대한민국" (거꾸로 적힘) → "경기 안양시 동안구"
    """
    text = unicodedata.normalize("NFC", address or "").replace(",", " ")
    tokens = [t for t in text.split() if t not in ("대한민국", "KR")]
    idx = next((i for i, t in enumerate(tokens) if t in PROVINCES), None)
    if idx is None:
        if tokens and tokens[0] in CITY_PROVINCE:
            return f"{CITY_PROVINCE[tokens[0]]} {_city_after(tokens)}"
        return ""
    province = PROVINCES[tokens[idx]]
    after = tokens[idx + 1:]
    while after and after[0] in PROVINCES:  # "서울특별시 KR 서울특별시 금천구"처럼 시도가 반복된 주소
        after = after[1:]
    city = _city_after(after) or _city_before(tokens[:idx])
    return f"{province} {city}" if city else province


def category_text(categories: list[str]) -> str:
    known = [c for c in CATEGORY_ORDER if c in set(categories)]
    return ", ".join(CATEGORY_LABELS[c] for c in known)


def compose(name: str, district: str, categories: list[str]) -> str:
    parts = [normalize_name(name), district, category_text(categories)]
    return SEPARATOR.join(p for p in parts if p)


def db_place_text(name: str, formatted_address: str, categories: list[str]) -> str:
    # DB 장소: ai_places.name, ai_places.formatted_address, ai_place_categories.category(여러 개 가능)
    return compose(name, district_of(formatted_address), categories)


def log_visit_text(visit: dict) -> str:
    # 로그 방문지: days.jsonl의 방문 기록(name, road_addr, lotno_addr, type_cd)
    address = visit.get("road_addr") or visit.get("lotno_addr") or ""
    category = category_for_type(visit.get("type_cd", ""))
    return compose(visit.get("name", ""), district_of(address), [category] if category else [])
