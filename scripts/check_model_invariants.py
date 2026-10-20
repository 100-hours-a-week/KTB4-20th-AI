"""모델 경로의 결과가 4개 invariant를 지키는지 조합 전체로 확인한다 (ADR-0003, ADR-0005 전제 조건, ADR-0006, ADR-0008).

tests/trips의 조합 테스트(7,680건: 지역 5 × 회피조건 부분집합 × 대표 설문)와 같은 입력으로 모델 경로(select_trip_by_model, 하루 일정)를 직접 부른다.
라우터를 거치면 모델 경로가 실패해도 규칙 경로가 대신 응답해 통과처럼 보이므로, 여기서는 모델 경로(여행 전체용 select_trip_by_model, 하루 일정)만 호출한다.

  invariant 1  장소 수 5 또는 6
  invariant 2  2·5번째가 FOOD
  invariant 3  회피 조건 type을 가진 장소 없음
  invariant 4  장소 중복 없음
  ADR-0006     FOOD가 아닌 슬롯은 취향 상위 두 카테고리(후보 부족 시 다음 순위 추가), 6번째는 ACTIVITY

  uv run --with onnxruntime --with numpy python -m scripts.check_model_invariants --model-dir "$V/m3/bundle_a" --workers 4
"""

import argparse
import collections
import copy
import hashlib
import json
import multiprocessing
import random
import time
from itertools import combinations
from pathlib import Path

from app.trips.schemas.schemas import E_Breaker, E_Preference, E_Region, Member_Survey
from app.trips.services.model.inference import InferenceRouter, LocalEngine
from app.trips.services.model.trip_selection import select_trip_by_model
from app.trips.services.place_selection import DIRECT_EXCLUDE_MAP
from app.trips.services.slot_plan import build_trip_slot_plan

SURVEYS = json.loads((Path(__file__).resolve().parents[1] / "tests/trips/data/representative_surveys.json").read_text(encoding="utf-8"))
_router: InferenceRouter | None = None
_diversity = False
FIRST_K = 30  # ADR-0007: 첫 장소를 모델 상위 30곳 중 시드 무작위로 고른 상태에서도 불변식이 지켜져야 한다


def hash_seed(case) -> int:
    return int(hashlib.sha256(repr(case).encode("utf-8")).hexdigest()[:15], 16)


def init_worker(model_dir: str, diversity: bool = False) -> None:
    global _router, _diversity
    _router = InferenceRouter(None, LocalEngine(Path(model_dir), max_concurrency=1))
    _diversity = diversity


def all_cases() -> list[tuple[str, tuple[str, ...], int]]:
    subsets = [c for n in range(len(E_Breaker) + 1) for c in combinations(E_Breaker, n)]
    return [(r.value, tuple(b.value for b in s), i) for r in E_Region for s in subsets for i in range(len(SURVEYS))]


def check(case: tuple[str, tuple[str, ...], int]) -> tuple[tuple, list[str], str]:
    region_value, breaker_values, survey_index = case
    members_raw = copy.deepcopy(SURVEYS[survey_index]["members"])
    members_raw[0]["deal_breakers"] = list(breaker_values)
    members = [Member_Survey.model_validate(m) for m in members_raw]
    breakers = [E_Breaker(b) for b in breaker_values]
    plan = build_trip_slot_plan(members, 1, random.Random(hash_seed(case)), _diversity)  # 하루 일정이라 하루 전체 템플릿
    problems: list[str] = []
    try:
        trace: dict = {}
        trip, route = select_trip_by_model(
            plan, breakers, E_Region(region_value), members, _router, diversity=_diversity,
            trace=trace, first_k=FIRST_K, seed=hash_seed(case),
        )
        places = trip[0]
    except Exception as error:  # noqa: BLE001  모델 경로 자체가 실패한 경우도 위반으로 센다
        return case, [f"예외 {type(error).__name__}: {error}"], "error"
    if len(places) not in (5, 6):
        problems.append(f"장소 수 {len(places)}")
    elif any(E_Preference.FOOD not in places[i].matched_preferences for i in (1, 4)):
        problems.append("2·5번째가 FOOD가 아님")
    excluded: set[str] = set()
    for b in breakers:
        excluded |= DIRECT_EXCLUDE_MAP.get(b, set())
    if any(excluded & {t.value for t in p.types} for p in places):
        problems.append("회피 type 포함")
    if len({p.id for p in places}) != len(places):
        problems.append("장소 중복")
    allowed = {E_Preference(c) for c in trace["daytime_categories"]["daytime"]}  # 취향 상위 둘(다양성 안은 넷), 후보가 모자랄 때만 다음 순위가 더해짐
    for i, p in enumerate(places):
        category = p.matched_preferences[0] if p.matched_preferences else None
        if i in (1, 4):
            continue
        if i == 5:
            if category != E_Preference.ACTIVITY:
                problems.append("6번째가 ACTIVITY가 아님")
        elif category not in allowed:
            problems.append(f"{i + 1}번째가 허용 카테고리가 아님")
    return case, problems, route


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="시험용: 앞의 n개 조합만")
    ap.add_argument("--diversity", action="store_true", help="다양성 안([2])으로 검사한다(주간 후보에 3위, 4위 카테고리 포함)")
    args = ap.parse_args(argv)
    cases = all_cases()[: args.limit] if args.limit else all_cases()
    started = time.time()
    failures: list[tuple] = []
    routes: collections.Counter = collections.Counter()
    kinds: collections.Counter = collections.Counter()
    with multiprocessing.Pool(args.workers, init_worker, (str(args.model_dir), args.diversity)) as pool:
        for n, (case, problems, route) in enumerate(pool.imap_unordered(check, cases, chunksize=8), start=1):
            routes[route] += 1
            if problems:
                failures.append((case, problems))
                kinds.update(problems)
            if n % 500 == 0:
                print(f"  {n}/{len(cases)} · 위반 {len(failures)} · {time.time() - started:.0f}s", flush=True)
    print(f"조합 {len(cases)}건 · 위반 {len(failures)}건 · 경로 {dict(routes)} · {time.time() - started:.0f}s")
    for kind, count in kinds.most_common(10):
        print(f"  {count}건: {kind}")
    for case, problems in failures[:5]:
        print("  예:", case, problems)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
