"""기상청 예보 값을 날씨 위험(has_issue, detail)으로 바꾼다 (ADR-0009).

단기 구간은 예보의 양으로, 중기 구간은 강수 확률과 날씨 문구로 판단한다(중기예보에는 강수량·적설량이 없다).
기준값은 아래 상수다. 운영하면서 조정한다.
"""

import re

from app.trips.schemas.api import E_Weather_Detail

HEAVY_RAIN_DAY_MM = 30.0  # 그 날 1시간 강수량 합계
HEAVY_RAIN_HOUR_MM = 15.0  # 어느 한 시간의 강수량
HEAVY_SNOW_DAY_CM = 5.0  # 그 날 신적설 합계
LIKELY_PRECIPITATION_PERCENT = 60  # 중기: 오전·오후 중 큰 강수 확률
BELOW_ONE = 0.5  # "1mm 미만", "1cm 미만"

_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def parse_amount(text: object) -> float | None:
    """단기예보의 PCP·SNO 문자열을 숫자로 바꾼다. 모르는 형식은 None.

    "강수없음"·"적설없음" → 0, "0" → 0, "1mm 미만" → 0.5, "1.0mm" → 1.0, "30.0~50.0mm" → 30.0(아래 끝), "50.0mm 이상" → 50.0
    """
    value = str(text).strip()
    if not value:
        return None
    if value in ("강수없음", "적설없음"):
        return 0.0
    if "미만" in value:
        return BELOW_ONE
    match = _NUMBER.search(value)
    return float(match.group()) if match else None


def judge_short_day(rain_mm: list[float], snow_cm: list[float]) -> list[E_Weather_Detail]:
    details: list[E_Weather_Detail] = []
    if sum(rain_mm) >= HEAVY_RAIN_DAY_MM or max(rain_mm, default=0.0) >= HEAVY_RAIN_HOUR_MM:
        details.append(E_Weather_Detail.HEAVY_RAIN)
    if sum(snow_cm) >= HEAVY_SNOW_DAY_CM:
        details.append(E_Weather_Detail.HEAVY_SNOW)
    return details


def judge_mid_day(probabilities: list[int], phrases: list[str]) -> list[E_Weather_Detail]:
    if max(probabilities, default=0) < LIKELY_PRECIPITATION_PERCENT:
        return []
    details: list[E_Weather_Detail] = []
    if any("비" in phrase or "소나기" in phrase for phrase in phrases):
        details.append(E_Weather_Detail.RAIN_LIKELY)
    if any("눈" in phrase for phrase in phrases):
        details.append(E_Weather_Detail.SNOW_LIKELY)
    return details
