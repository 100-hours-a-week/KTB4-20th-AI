"""요청의 외부 요인(precheck 응답)에서 이 여행에 해당하는 신호만 골라낸다 (ADR-0008).

신호는 has_issue만 읽는다. 지역·날짜가 없거나 모르는 표기는 신호 없음으로 본다(예보 범위 밖 날짜, 새로 추가된 지역이 요청 실패로 이어지지 않게).
"""

from datetime import date, timedelta

from app.trips.schemas.api import External_Factors
from app.trips.schemas.schemas import E_Region

PRECHECK_REGION_TO_REGION: dict[str, E_Region] = {
    "REGION-SEOUL": E_Region.SEOUL,
    "REGION-BUSAN": E_Region.BUSAN,
    "REGION-GYEONGJU": E_Region.GYEONGJU,
    "REGION-JEJU": E_Region.JEJU,
    "REGION-JEONJU": E_Region.JEONJU,
}


def outdoor_excluded_days(
    external_factors: External_Factors | None, region: E_Region, start_date: date, num_days: int,
) -> frozenset[int]:
    """야외 후보를 빼야 하는 일자 번호(0부터)."""
    if external_factors is None:
        return frozenset()
    trip_days = {start_date + timedelta(days=index): index for index in range(num_days)}
    days: set[int] = set()
    for forecast in external_factors.forecasts:
        index = trip_days.get(forecast.date)
        if index is None:
            continue
        if any(PRECHECK_REGION_TO_REGION.get(entry.region) == region and entry.has_issue for entry in forecast.regions):
            days.add(index)
    return frozenset(days)
