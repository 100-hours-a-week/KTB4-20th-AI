"""precheck: 5개 지역의 내일부터 9일 치 날씨 위험을 판정한다 (ADR-0009).

BE가 여행 출발 하루 전 06시에 트리거를 보낸다. 응답은 여행과 무관하게 같으므로 기상청 예보를 짧게 캐시해서 몰리는 트리거가 같은 예보를 한 번만 받게 한다.
날짜마다 단기예보(그날 예보가 12시간 이상 있을 때)를, 없으면 중기예보를 쓴다. 호출이 일부 실패하면 성공한 날짜·지역만 응답하고(빠진 값은 place-selection이
신호 없음으로 본다), 전부 실패하면 PrecheckUnavailable이다. 실패를 has_issue false로 채우지 않는다.
"""

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

from app.core.config import settings
from app.core.time import SERVICE_TIMEZONE
from app.trips.schemas.api import (
    E_Precheck_Reason_Category,
    External_Forecast_Day,
    External_Reason,
    External_Region_Forecast,
    Precheck_Response_Data,
)
from app.trips.services import kma_client
from app.trips.services.kma_client import KmaError, MidDay, ShortDay
from app.trips.services.weather_risk import judge_mid_day, judge_short_day

logger = logging.getLogger(__name__)

FORECAST_DAYS = 9  # 내일부터 9일. 최대 여행 기간(8박 9일)과 같다
SHORT_FULL_DAY_HOURS = 12  # 이 시각 수 이상 예보가 있는 날짜만 단기예보로 판정한다(마지막 날은 일부 시간만 있다)


@dataclass(frozen=True)
class PrecheckRegion:
    name: str  # place-selection의 external_factors와 같은 표기 (services/external_factors.py)
    short_grids: tuple[tuple[int, int], ...]  # 단기예보 격자(nx, ny). 지역별 수집 지점의 중심을 precheck.convert_to_grid로 바꾼 값
    mid_reg_id: str  # 중기예보 시·도 구역 코드


# 제주는 제주시권과 서귀포권 두 격자를 보고 하나라도 이슈면 true다
REGIONS: tuple[PrecheckRegion, ...] = (
    PrecheckRegion("REGION-SEOUL", ((60, 127),), "11B00000"),
    PrecheckRegion("REGION-BUSAN", ((98, 76),), "11H20000"),
    PrecheckRegion("REGION-GYEONGJU", ((100, 90),), "11H10000"),
    PrecheckRegion("REGION-JEJU", ((53, 38), (53, 33)), "11G00000"),
    PrecheckRegion("REGION-JEONJU", ((63, 89),), "11F10000"),
)


class PrecheckUnavailable(Exception):
    """판정할 수 있는 날짜·지역이 하나도 없음(키 없음, 기상청 전부 실패)."""


class ForecastCache:
    """키별로 값을 짧게 보관한다. 같은 키를 동시에 요청하면 한 번만 받는다. 실패는 저장하지 않는다."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._entries: dict[tuple, tuple[float, datetime, Any]] = {}
        self._locks: dict[tuple, threading.Lock] = {}
        self._guard = threading.Lock()

    def get[T](self, key: tuple, fetch: Callable[[], T], ttl: float) -> tuple[T, datetime]:
        """(값, 받은 시각 UTC)."""
        with self._guard:
            lock = self._locks.setdefault(key, threading.Lock())
        with lock:
            entry = self._entries.get(key)
            if entry is not None and self._clock() < entry[0]:
                return entry[2], entry[1]
            value = fetch()
            fetched_at = datetime.now(UTC)
            self._entries[key] = (self._clock() + ttl, fetched_at, value)
            return value, fetched_at

    def clear(self) -> None:
        with self._guard:
            self._entries.clear()


_cache = ForecastCache()


def _short_reasons(days: dict[date, ShortDay], day: date) -> tuple[bool, list]:
    entry = days.get(day)
    if entry is None:
        return False, []
    return True, judge_short_day(entry.rain_mm, entry.snow_cm)


def build_precheck(now: datetime | None = None, cache: ForecastCache | None = None) -> Precheck_Response_Data:
    if not settings.kma_api_key.strip():
        raise PrecheckUnavailable("기상청 인증키(KMA_API_KEY)가 설정되지 않음")
    current = (now or datetime.now(SERVICE_TIMEZONE)).astimezone(SERVICE_TIMEZONE)  # 서비스 시간대(Asia/Seoul)의 지금
    cache = cache or _cache
    ttl = settings.precheck_cache_ttl_seconds
    dates = [current.date() + timedelta(days=offset) for offset in range(1, FORECAST_DAYS + 1)]

    def load[T](key: tuple, fetch: Callable[[], T]) -> tuple[T, datetime] | None:
        try:
            return cache.get(key, fetch, ttl)
        except KmaError as error:
            logger.warning("precheck: 기상청 %s 실패: %s", key[0], error)  # 메시지에 URL과 키는 없다
            return None

    grids = {grid for region in REGIONS for grid in region.short_grids}
    with ThreadPoolExecutor(max_workers=8) as pool:
        short_futures = {
            grid: pool.submit(load, ("short", *grid), lambda grid=grid: kma_client.fetch_short(grid[0], grid[1], current)) for grid in grids
        }
        mid_futures = {
            region.mid_reg_id: pool.submit(load, ("mid", region.mid_reg_id), lambda reg_id=region.mid_reg_id: kma_client.fetch_mid(reg_id, current))
            for region in REGIONS
        }
        short_loaded = {grid: future.result() for grid, future in short_futures.items()}
        mid_loaded = {reg_id: future.result() for reg_id, future in mid_futures.items()}

    fetched_times = [entry[1] for entry in (*short_loaded.values(), *mid_loaded.values()) if entry is not None]
    forecasts: list[External_Forecast_Day] = []
    for day in dates:
        regions: list[External_Region_Forecast] = []
        for region in REGIONS:
            shorts: list[dict[date, ShortDay]] = [entry[0] for grid in region.short_grids if (entry := short_loaded[grid]) is not None]
            mid_entry = mid_loaded[region.mid_reg_id]
            mid_days: dict[date, MidDay] = mid_entry[0] if mid_entry else {}
            full = [d for d in shorts if day in d and len(d[day].hours) >= SHORT_FULL_DAY_HOURS]
            partial = [d for d in shorts if day in d]
            if full:
                details = [x for d in full for x in _short_reasons(d, day)[1]]
            elif day in mid_days:
                details = judge_mid_day(mid_days[day].probabilities, mid_days[day].phrases)
            elif partial:
                details = [x for d in partial for x in _short_reasons(d, day)[1]]
            else:
                continue  # 이 날짜·지역은 판정할 자료가 없다
            unique = list(dict.fromkeys(details))
            regions.append(External_Region_Forecast(
                region=region.name, has_issue=bool(unique),
                reasons=[External_Reason(category=E_Precheck_Reason_Category.WEATHER_RISK.value, detail=d.value) for d in unique],
            ))
        if regions:
            forecasts.append(External_Forecast_Day(date=day, regions=regions))
    if not forecasts:
        raise PrecheckUnavailable("기상청 예보를 하나도 받지 못함")
    return Precheck_Response_Data(generated_at=min(fetched_times).replace(microsecond=0), forecasts=forecasts)
