"""기상청(공공데이터포털) 단기·중기 예보 호출 (ADR-0009).

단기예보(VilageFcstInfoService_2.0/getVilageFcst): 격자(nx, ny)별 시간 단위 예보. 발표 후 약 3.5일 뒤까지.
중기예보(MidFcstInfoService/getMidLandFcst): 시·도 구역(regId)별 날짜 단위 강수 확률과 날씨 문구. 06시 발표분은 4일 뒤부터, 18시 발표분은 5일 뒤부터 10일 뒤까지.

인증키가 URL 쿼리에 들어가고 httpx 예외 문구에 URL이 들어갈 수 있으므로, 예외는 종류만 남기고 원문을 붙이지 않는다(`from None`).
"""

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import httpx

from app.core.config import settings
from app.trips.services.weather_risk import parse_amount

SHORT_URL = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
MID_URL = "https://apis.data.go.kr/1360000/MidFcstInfoService/getMidLandFcst"
SHORT_BASE_HOURS = (2, 5, 8, 11, 14, 17, 20, 23)  # 단기예보 발표 시각
SHORT_PUBLISH_DELAY = timedelta(minutes=15)  # 발표 후 약 10분 뒤부터 받을 수 있다
MID_ISSUE_HOURS = (6, 18)  # 중기예보 발표 시각
ROWS_PER_PAGE = 1500  # 단기예보는 한 번에 1,016건 안팎이라 한 쪽이면 충분하지만 넘으면 다음 쪽을 읽는다
MAX_PAGES = 4

_MID_FIELD = re.compile(r"^(rnSt|wf)(\d+)(?:Am|Pm)?$")


class KmaError(Exception):
    """기상청 호출이나 응답 해석에 실패함. 메시지에는 URL과 키를 넣지 않는다."""


@dataclass
class ShortDay:
    rain_mm: list[float] = field(default_factory=list)
    snow_cm: list[float] = field(default_factory=list)
    hours: set[str] = field(default_factory=set)  # 이 날짜에 예보가 있는 시각(HHMM). 많을수록 온전한 하루다


@dataclass
class MidDay:
    probabilities: list[int] = field(default_factory=list)
    phrases: list[str] = field(default_factory=list)


def _ymd(text: str) -> date:
    return date(int(text[0:4]), int(text[4:6]), int(text[6:8]))


def _recent(candidates: list[datetime], limit: datetime, count: int = 2) -> list[datetime]:
    return sorted((c for c in candidates if c <= limit), reverse=True)[:count]


def short_base_candidates(now: datetime) -> list[tuple[str, str]]:
    """받을 수 있는 가장 최근 발표 시각 둘(최신, 직전). (base_date, base_time)."""
    reference = now - SHORT_PUBLISH_DELAY
    candidates = [
        datetime(day.year, day.month, day.day, hour, tzinfo=now.tzinfo)
        for day in (reference.date() - timedelta(days=1), reference.date()) for hour in SHORT_BASE_HOURS
    ]
    return [(c.strftime("%Y%m%d"), c.strftime("%H00")) for c in _recent(candidates, reference)]


def mid_issue_candidates(now: datetime) -> list[datetime]:
    """가장 최근 중기예보 발표 둘(최신, 직전)."""
    candidates = [
        datetime(day.year, day.month, day.day, hour, tzinfo=now.tzinfo)
        for day in (now.date() - timedelta(days=1), now.date()) for hour in MID_ISSUE_HOURS
    ]
    return _recent(candidates, now)


def _response(url: str, params: dict) -> dict:
    try:
        http = httpx.get(
            url, params={"serviceKey": settings.kma_api_key, "dataType": "JSON", **params}, timeout=settings.kma_timeout_seconds,
        )
    except httpx.HTTPError as error:
        raise KmaError(f"기상청 호출 실패: {type(error).__name__}") from None
    if http.status_code != 200:
        raise KmaError(f"기상청 HTTP {http.status_code}")
    try:
        return http.json()["response"]
    except (ValueError, KeyError, TypeError):  # 인증 오류 등은 JSON이 아니라 XML로 오기도 한다
        raise KmaError("기상청 응답 형식 오류") from None


def _items(response: dict) -> tuple[list[dict], int]:
    """(항목 목록, 전체 건수). 자료가 없으면 빈 목록."""
    try:
        code = response["header"]["resultCode"]
        if code == "03":  # NO_DATA
            return [], 0
        if code != "00":
            raise KmaError(f"기상청 결과 코드 {code}")
        body = response.get("body") or {}
        items = body.get("items")
        if not isinstance(items, dict):
            return [], 0
        item = items.get("item", [])
        return (item if isinstance(item, list) else [item]), int(body.get("totalCount", 0))
    except (KeyError, TypeError, ValueError):
        raise KmaError("기상청 응답 형식 오류") from None


def fetch_short(nx: int, ny: int, now: datetime) -> dict[date, ShortDay]:
    """격자의 단기예보를 날짜별 1시간 강수량·신적설 목록으로 돌려준다."""
    for base_date, base_time in short_base_candidates(now):
        items: list[dict] = []
        for page in range(1, MAX_PAGES + 1):
            page_items, total = _items(_response(SHORT_URL, {
                "numOfRows": ROWS_PER_PAGE, "pageNo": page, "base_date": base_date, "base_time": base_time, "nx": nx, "ny": ny,
            }))
            items += page_items
            if not page_items or len(items) >= total:
                break
        if items:
            break
    else:
        raise KmaError("단기예보 자료 없음")
    days: dict[date, ShortDay] = {}
    try:
        for item in items:
            day = days.setdefault(_ymd(item["fcstDate"]), ShortDay())
            day.hours.add(item["fcstTime"])
            if item["category"] == "PCP" and (amount := parse_amount(item["fcstValue"])) is not None:
                day.rain_mm.append(amount)
            elif item["category"] == "SNO" and (amount := parse_amount(item["fcstValue"])) is not None:
                day.snow_cm.append(amount)
    except (KeyError, TypeError, ValueError):
        raise KmaError("단기예보 항목 형식 오류") from None
    return days


def fetch_mid(reg_id: str, now: datetime) -> dict[date, MidDay]:
    """시·도 구역의 중기예보를 날짜별 강수 확률·문구로 돌려준다(오전·오후는 한 날짜로 합침)."""
    for issued in mid_issue_candidates(now):
        items, _ = _items(_response(MID_URL, {
            "numOfRows": 10, "pageNo": 1, "regId": reg_id, "tmFc": issued.strftime("%Y%m%d%H%M"),
        }))
        if items:
            break
    else:
        raise KmaError("중기예보 자료 없음")
    days: dict[date, MidDay] = {}
    try:
        for key, value in items[0].items():
            match = _MID_FIELD.match(key)
            if not match or value is None:
                continue
            day = days.setdefault(issued.date() + timedelta(days=int(match.group(2))), MidDay())
            if match.group(1) == "rnSt":
                day.probabilities.append(int(value))
            else:
                day.phrases.append(str(value))
    except (TypeError, ValueError):
        raise KmaError("중기예보 항목 형식 오류") from None
    return days
