"""모델 경로(CPU)와 규칙 경로의 응답 시간·메모리를 로컬 DB로 잰다 (M5 T5-2, T5-5, ADR-0004).

  uv run --with onnxruntime --with numpy python -m scripts.bench_model_path --model-dir "$V/m3/bundle_a" --runs 20

지역 5곳 × 취향 3가지 입력으로 규칙 경로와 모델 CPU 경로를 번갈아 부르고, 첫 호출과 이후 호출의 지연(평균, p50, p95)을 적는다.
DB 조회 시간과 추론 시간을 나눠서 적는다. 메모리는 프로세스 최대 상주 메모리(RSS)다.
"""

import argparse
import random
import resource
import statistics
import sys
import time
from pathlib import Path

from app.trips.schemas.schemas import E_Preference, E_Region, Member_Survey, User
from app.trips.services.model.inference import InferenceRouter, LocalEngine
from app.trips.services.model.selection import model_slot_plan_seed, select_places_by_model
from app.trips.services.place_selection import select_places_by_slot_plan
from app.trips.services.slot_plan import build_slot_plan

# 설문 15문항: 역사·문화(0~2), 자연(3~5), 음식(6~8), 액티비티(9~11), 편의·쇼핑(12~14)
PROFILES = {
    "역사문화": [5, 5, 5, 1, 1, 1, 3, 3, 3, 1, 1, 1, 1, 1, 1],
    "자연액티비티": [1, 1, 1, 5, 5, 5, 3, 3, 3, 5, 5, 5, 1, 1, 1],
    "쇼핑음식": [1, 1, 1, 1, 1, 1, 5, 5, 5, 3, 3, 3, 5, 5, 5],
}


def member(profile: list[int]) -> Member_Survey:
    return Member_Survey(user=User(user_id="bench"), survey_result=profile)


def check_invariants(places) -> list[str]:
    problems = []
    if not 5 <= len(places) <= 6:
        problems.append(f"장소 수 {len(places)}")
    if len({p.id for p in places}) != len(places):
        problems.append("중복 장소")
    food = [i for i, p in enumerate(places) if E_Preference.FOOD in p.matched_preferences]
    if len(places) >= 5 and (1 not in food or 4 not in food):
        problems.append("2·5번째가 FOOD가 아님")
    return problems


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * q))]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--runs", type=int, default=20)
    args = ap.parse_args(argv)
    load_start = time.perf_counter()
    cpu = LocalEngine(args.model_dir, max_concurrency=1)
    print(f"모델 적재 {time.perf_counter() - load_start:.2f}s · 코드 사전 {len(cpu.codes)}개")
    router = InferenceRouter(None, cpu)
    rng = random.Random(0)
    cases = [(r, name) for r in E_Region for name in PROFILES]
    times: dict[str, list[float]] = {"rule": [], "model": []}
    first: dict[str, float] = {}
    violations = 0
    for i in range(args.runs):
        region, name = cases[rng.randrange(len(cases))] if i else cases[0]
        members = [member(PROFILES[name])]
        t = time.perf_counter()
        rule_places = select_places_by_slot_plan(build_slot_plan(members), [], region, members)
        times["rule"].append(time.perf_counter() - t)
        t = time.perf_counter()
        plan = build_slot_plan(members, model_slot_plan_seed())
        model_places, route = select_places_by_model(plan, [], region, members, router)
        times["model"].append(time.perf_counter() - t)
        first.setdefault("rule", times["rule"][0])
        first.setdefault("model", times["model"][0])
        problems = check_invariants(model_places)
        violations += bool(problems)
        if problems:
            print(f"  불변식 위반({region.value}, {name}): {problems}")
        if i < 3:
            print(f"예시 {region.value}/{name}: 규칙 {[p.displayName.text for p in rule_places]}")
            print(f"                     모델({route}) {[p.displayName.text for p in model_places]}")
    for key, vals in times.items():
        rest = vals[1:] or vals
        print(f"{key}: 첫 호출 {first[key] * 1000:.0f}ms · 이후 평균 {statistics.mean(rest) * 1000:.0f}ms "
              f"p50 {percentile(rest, 0.5) * 1000:.0f}ms p95 {percentile(rest, 0.95) * 1000:.0f}ms (n={len(vals)})")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024)
    print(f"최대 상주 메모리 {rss:.0f}MB · 모델 경로 불변식 위반 {violations}/{args.runs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
