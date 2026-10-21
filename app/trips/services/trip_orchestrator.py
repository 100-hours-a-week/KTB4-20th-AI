"""여행 전체 3개 안을 만든다 (ADR-0008).

  [0] 규칙 경로(평점순). Spring이 재정렬하는 원본이라 항상 규칙으로 만든다.
  [1] 모델 경로의 취향 안, [2] 모델 경로에 3위·4위 카테고리를 더한 다양성 안.
      모델 경로가 실패하거나 그 안이 유효하지 않으면 그 안만 규칙 변형으로 채운다. 어느 경우에도 3개 안이 나온다.

[0]의 DB 조회와 [1], [2]의 후보 조회·모델 호출은 동시에 시작한다. 시드는 요청의 지역, 날짜, 구성원, 이 여행에 해당하는 날씨 신호로만 만든다
(precheck의 generated_at이나 다른 지역의 신호가 바뀌어도 같은 여행 조건이면 같은 일정이 나온다).
"""

import hashlib
import json
import logging
import random
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from app.core.config import settings
from app.trips.schemas.api import Place_Selection_Request
from app.trips.schemas.schemas import E_Breaker, Place
from app.trips.services.model.trip_selection import select_trip_by_model
from app.trips.services.slot_plan import (
    E_Time_Slot,
    Trip_Slot_Plan,
    build_trip_slot_plan,
)
from app.trips.services.trip_rule import select_trip_by_slot_plan

logger = logging.getLogger(__name__)

Trip = list[list[Place]]


def derive_seed(base: int, label: str) -> int:
    return int(hashlib.sha256(f"{base}:{label}".encode()).hexdigest()[:15], 16)


def trip_seed(request: Place_Selection_Request, outdoor_excluded_days: frozenset[int]) -> int:
    payload = {
        "region": request.region.value,
        "start_date": request.start_date.isoformat(),
        "end_date": request.end_date.isoformat(),
        "members": [member.model_dump(mode="json") for member in request.members],
        "outdoor_excluded_days": sorted(outdoor_excluded_days),
    }
    return int(hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:15], 16)


def is_valid_model_trip(trip_plan: Trip_Slot_Plan, trip: Trip) -> bool:
    """일자마다 저녁 이후를 뺀 슬롯을 모두 채웠을 때만 유효하다. 하루 일정의 기준(5곳 이상)을 여행 전체로 늘린 것이다."""
    if len(trip) != len(trip_plan.days):
        return False
    return all(
        len(day_places) >= sum(1 for item in day_plan.items if item.time_slot != E_Time_Slot.EVENING)
        for day_plan, day_places in zip(trip_plan.days, trip, strict=True)
    )


def build_trip_plans(
    request: Place_Selection_Request, deal_breakers: list[E_Breaker], num_days: int, outdoor_excluded_days: frozenset[int],
) -> list[Trip]:
    base = trip_seed(request, outdoor_excluded_days)
    members, region = request.members, request.region
    # 세 안이 같은 카테고리 순위를 공유한다. 동점 순서도 요청 시드로 정해서 같은 요청은 같은 순위가 나온다
    plan = build_trip_slot_plan(members, num_days, random.Random(derive_seed(base, "ties")))
    diverse_plan = build_trip_slot_plan(members, num_days, random.Random(derive_seed(base, "ties")), diversity=True)

    def rule_trip(trip_plan: Trip_Slot_Plan, variant_label: str | None = None) -> Trip:
        variant_seed = None if variant_label is None else derive_seed(base, variant_label)
        return select_trip_by_slot_plan(trip_plan, deal_breakers, region, members, outdoor_excluded_days, variant_seed)

    def model_trip(trip_plan: Trip_Slot_Plan, label: str, diversity: bool) -> Trip:
        from app.trips.services.model.runtime import get_router

        trip, route = select_trip_by_model(
            trip_plan, deal_breakers, region, members, get_router(), outdoor_excluded_days, diversity=diversity,
            first_k=settings.model_first_place_top_k, seed=derive_seed(base, label),
        )
        if not is_valid_model_trip(trip_plan, trip):
            raise ValueError(f"모델 경로 결과가 유효하지 않음: 일자별 {[len(day) for day in trip]}곳")
        logger.info("place-selection: model plan=%s route=%s places=%d", label, route, sum(len(day) for day in trip))
        return trip

    steps: list[tuple[str, Callable[[], Trip], Callable[[], Trip]]] = [
        ("plan1", lambda: model_trip(plan, "plan1", False), lambda: rule_trip(plan, "plan1")),
        ("plan2", lambda: model_trip(diverse_plan, "plan2", True), lambda: rule_trip(diverse_plan, "plan2")),
    ]
    with ThreadPoolExecutor(max_workers=3) as pool:
        rule_future = pool.submit(rule_trip, plan)
        futures = [(label, pool.submit(primary), fallback) for label, primary, fallback in steps]
        results = [rule_future.result()]
        for label, future, fallback in futures:
            try:
                results.append(future.result())
            except Exception:
                logger.exception("모델 경로 실패, 규칙 변형으로 채움: %s", label)
                results.append(fallback())
    return results
