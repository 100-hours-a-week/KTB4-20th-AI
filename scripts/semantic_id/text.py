"""장소 텍스트를 만드는 공통 규칙 (M1 T1-6, T1-7). DB 장소와 로그 방문지에 같은 규칙을 쓴다.

같은 장소가 같은 코드를 받으려면 임베딩 입력이 양쪽에서 같은 방식으로 만들어져야 한다(ADR-0003).
"""

import re
import unicodedata

HANGUL = re.compile(r"[가-힣]")
_BRACKETS = str.maketrans({"（": "(", "）": ")", "［": "[", "］": "]"})
_BRACKET_PART = re.compile(r"\s*[(\[]([^)\]]*)[)\]]\s*")


def _has_hangul(text: str) -> bool:
    return bool(HANGUL.search(text))


def normalize_name(name: str) -> str:
    """이름에서 영문 병기와 대안 표기를 정리한다. 한글이 없으면 공백만 정리한다.

    1. NFC로 맞추고 전각 괄호를 반각으로 바꾸고 공백을 하나로 줄인다
    2. "/"로 나뉜 대안 표기가 있으면 한글이 있는 첫 부분을 쓴다
       예: "카페 아래헌 / Cafe Arae-Heon / カフェアレホン" → "카페 아래헌"
    3. 괄호 안 내용은, 바깥에 한글이 있으면 영문 병기나 보조 설명으로 보고 뺀다 ("스타벅스 (Starbucks) 서울역점" → "스타벅스 서울역점",
       "관악산둘레길(안양구간)" → "관악산둘레길"). 로그 이름에는 보조 설명이 거의 없어, 남기면 양쪽 텍스트가 어긋난다
       바깥에 한글이 없고 안에 한글이 있으면 괄호 글자만 뺀다 ("Gymboxx Fitness (평촌점)" → "Gymboxx Fitness 평촌점")
    4. 브랜드 약어가 섞인 이름("AK& 홍대", "버거킹 평촌금성GS점")은 한글이 같이 있어도 그대로 둔다
    """
    text = unicodedata.normalize("NFC", name).translate(_BRACKETS)
    text = re.sub(r"\s+", " ", text).strip()
    if not text or not _has_hangul(text):
        return text

    parts = [p.strip() for p in re.split(r"\s*/\s*", text) if p.strip()]
    chosen = next((p for p in parts if _has_hangul(p)), text)

    if _BRACKET_PART.search(chosen):
        outside = _BRACKET_PART.sub(" ", chosen)
        if _has_hangul(outside):
            chosen = outside  # 괄호 안은 영문 병기
        else:
            chosen = _BRACKET_PART.sub(lambda m: " " + m.group(1) + " ", chosen)  # 안쪽 한글을 살린다
    cleaned = re.sub(r"\s+", " ", chosen).strip()
    return cleaned or text


# 방문지 유형 코드 → 서비스 카테고리 (T1-7)
# 서비스 지역 방문 중 DB 장소와 엄격히 일치한 11,272건에서 유형 코드별로 DB 카테고리 분포를 잰 결과로 정했다.
# 한 카테고리가 70% 이상이면 그 카테고리로 두고, 갈리거나 표본이 작은 코드는 라벨을 비운다.
#   11 식당/카페 → FOOD 97%                       4 상업지구 → CONVENIENCE_SHOPPING 96%
#   2 역사/유적/종교 → HISTORY_CULTURE 94%         3 문화시설 → HISTORY_CULTURE 81%
#   5 레저/스포츠 → ACTIVITY 84%                   1 자연관광지 → NATURE_HEALING 77%
#   6 테마시설 → ACTIVITY 64%(NATURE_HEALING 28%)  → 70% 미만이지만 정의상 ACTIVITY와 가장 가까워 둔다
# 라벨을 비우는 코드:
#   7 산책로(NATURE_HEALING 47% / HISTORY_CULTURE 48%), 8 축제(일치 32건), 9 역·터미널·휴게소(교통 정류),
#   10 상점(FOOD 50% / CONVENIENCE_SHOPPING 47%), 12 기타, 13 체험 활동 관광지(NATURE_HEALING 38%로 갈림)
TYPE_CODE_CATEGORY: dict[str, str] = {
    "11": "FOOD",
    "4": "CONVENIENCE_SHOPPING",
    "2": "HISTORY_CULTURE",
    "3": "HISTORY_CULTURE",
    "5": "ACTIVITY",
    "6": "ACTIVITY",
    "1": "NATURE_HEALING",
}


def category_for_type(type_cd: str) -> str | None:
    return TYPE_CODE_CATEGORY.get((type_cd or "").strip())
