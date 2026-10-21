"""여러 날 일정 3개 안의 지연, 안 사이 겹침, 다양성 안의 카테고리 비율을 잰다 (ADR-0008 PR 4).

실제 서비스 흐름(trip_orchestrator.build_trip_plans)을 모델 모드로 그대로 부르고, 모델 호출은 감싸서 시간과 경로를 기록한다.
DB는 .env의 로컬 DB를 조회만 한다. 모델 호출은 둘 중 하나다.
  --model-dir  로컬 CPU(ONNX). 비용이 없다. 지연은 CPU 값이라 GPU 서버 지연(ADR-0008의 p95 15초 기준)으로 쓰지 않는다
  --runpod     RunPod GPU. 유료다. --max-calls(GPU 호출 상한, 요청 한 건이 2번 부른다)를 반드시 주고 사용자가 승인한 값만 쓴다.
               주소와 키는 .env의 RUNPOD_MODEL_URL, RUNPOD_MODEL_API_KEY에서 읽고 출력하지 않는다. v3 이미지(days 요청)가 배포된 뒤에만 의미가 있다

  uv run --with onnxruntime --with numpy python -m scripts.measure_trip_plans --model-dir "$V/m3/bundle_a" \\
      --regions 서울,경주 --surveys 3 --lengths 1,5,9 --signal-days 2,3 --out "$V/m3/eval/trip_plans_cpu.json"
  uv run python -m scripts.measure_trip_plans --runpod --max-calls 40 --timeout 60 --surveys 2 --lengths 5,9 --out "$V/m3/eval/trip_plans_gpu.json"

측정하는 것과 기준(ADR-0008)
  모델 호출 지연     안 [1], [2]의 모델 호출 시간. p95가 15초를 넘으면 일자를 나눠 부르는 방식으로 재설계한다
  전체 응답 시간     build_trip_plans 전체. BE 응답 대기 30초 안에 들어와야 한다
  안 사이 겹침       [1]과 [0]이 일자당 저녁 이후를 뺀 슬롯에서 2곳 이상 달라야 한다(2곳보다 슬롯이 적은 날은 전부 달라야 함). 못 채우면 안 사이 중복 제거를 도입한다
  3위·4위 비율       안별로 FOOD가 아닌 주간 장소 중 3위·4위 카테고리인 장소의 비율. 다양성 안 [2]가 기본 안보다 높은지 본다
  모델 사용 여부     [1], [2]가 모델 결과로 응답했는지(아니면 규칙 변형으로 대체됨)
  날씨 신호          신호가 있는 일자(--signal-days)에 야외 type 장소가 0곳인지
"""

import argparse
import copy
import json
import random
import statistics
import time
from collections.abc import Callable, Sequence
from datetime import timedelta
from pathlib import Path

from app.core.config import settings
from app.core.time import get_service_today
from app.trips.schemas.api import Place_Selection_Request
from app.trips.schemas.schemas import E_Preference, E_Region, Member_Survey, Place
from app.trips.services import trip_orchestrator
from app.trips.services.external_factors import (
    PRECHECK_REGION_TO_REGION,
    outdoor_excluded_days,
)
from app.trips.services.model import runtime
from app.trips.services.place_selection import NIGHT_ACTIVITY_TYPE_VALUES
from app.trips.services.slot_plan import build_trip_slot_plan
from app.trips.services.trip_rule import OUTDOOR_TYPE_VALUES

SURVEYS_PATH = Path(__file__).resolve().parents[1] / "tests/trips/data/representative_surveys.json"
GPU_CALLS_PER_REQUEST = 2  # [1], [2]
CALL_LIMIT_SECONDS = 15.0  # ADR-0008: 모델 호출 p95 기준
RESPONSE_LIMIT_SECONDS = 30.0  # BE 응답 대기
MIN_DIFFERENT_PER_DAY = 2
PAIRS = {"plan1_vs_plan0": (1, 0), "plan2_vs_plan0": (2, 0), "plan2_vs_plan1": (2, 1)}
REGION_TO_PRECHECK = {region: name for name, region in PRECHECK_REGION_TO_REGION.items()}


# ---- 순수 계산 (RunPod, DB 없이 테스트한다) ----

def is_evening_place(place: Place) -> bool:
    """저녁 이후 슬롯의 장소(야간형 ACTIVITY type을 가짐). 주간 ACTIVITY 후보는 야간 type을 제외하고 조회한다."""
    return bool(NIGHT_ACTIVITY_TYPE_VALUES & {place_type.value for place_type in place.types})


def day_difference(day: Sequence[Place], other_day: Sequence[Place]) -> tuple[int, int]:
    """(day의 저녁 이후를 뺀 장소 중 other_day에 없는 장소 수, day의 저녁 이후를 뺀 장소 수)."""
    own = [place.id for place in day if not is_evening_place(place)]
    other = {place.id for place in other_day if not is_evening_place(place)}
    return sum(1 for place_id in own if place_id not in other), len(own)


def meets_difference(different: int, slots: int, minimum: int = MIN_DIFFERENT_PER_DAY) -> bool:
    return different >= min(minimum, slots)


def extra_category_share(trip: Sequence[Sequence[Place]], ranking: Sequence[E_Preference]) -> tuple[int, int]:
    """(FOOD가 아닌 주간 장소 중 3위·4위 카테고리인 수, FOOD가 아닌 주간 장소 수)."""
    extra, total = set(ranking[2:]), 0
    count = 0
    for day in trip:
        for place in day:
            if is_evening_place(place) or not place.matched_preferences or place.matched_preferences[0] == E_Preference.FOOD:
                continue
            total += 1
            count += place.matched_preferences[0] in extra
    return count, total


def outdoor_places_on_days(trip: Sequence[Sequence[Place]], days: Sequence[int]) -> int:
    """날씨 신호가 있는 일자(days)에 들어간 야외 type 장소 수. 0이어야 한다."""
    return sum(
        1 for index in days if index < len(trip) for place in trip[index]
        if OUTDOOR_TYPE_VALUES & {place_type.value for place_type in place.types}
    )


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * q))]


def summarize(values: Sequence[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values), "mean": round(statistics.mean(values), 3), "p50": round(percentile(values, 0.5), 3),
        "p95": round(percentile(values, 0.95), 3), "max": round(max(values), 3),
    }


def external_factors_for(region: E_Region, start_date, signal_days: Sequence[int], num_days: int) -> dict | None:
    days = [index for index in signal_days if index < num_days]
    if not days:
        return None
    forecasts = [
        {"date": (start_date + timedelta(days=index)).isoformat(),
         "regions": [{"region": REGION_TO_PRECHECK[region], "has_issue": True,
                      "reasons": [{"category": "weather_risk", "detail": "heavy_rain"}]}]}
        for index in days
    ]
    return {"generated_at": "2026-10-10T10:00:03Z", "forecasts": forecasts}


# ---- 실행 ----

def run_case(case: dict, surveys: list[dict]) -> dict:
    """요청 한 건을 실제 흐름으로 처리하고 지표를 돌려준다. 모델 호출은 감싸서 시간과 경로를 기록한다."""
    members_raw = copy.deepcopy(surveys[case["survey"]]["members"])
    members = [Member_Survey.model_validate(member) for member in members_raw]
    region, num_days, today = E_Region(case["region"]), case["days"], get_service_today()
    request = Place_Selection_Request(
        region=region, start_date=today, end_date=today + timedelta(days=num_days - 1), members=members,
        external_factors=external_factors_for(region, today, case["signal_days"], num_days),
    )
    deal_breakers = [breaker for member in members for breaker in member.deal_breakers]
    signal_days = outdoor_excluded_days(request.external_factors, region, today, num_days)

    recorded: dict[str, dict] = {}
    real = trip_orchestrator.select_trip_by_model

    def timed(trip_plan, *args, diversity=False, **kwargs):
        label = "plan2" if diversity else "plan1"
        started = time.perf_counter()
        try:
            trip, route = real(trip_plan, *args, diversity=diversity, **kwargs)
        except Exception as error:
            recorded[label] = {"sec": time.perf_counter() - started, "error": type(error).__name__}
            raise
        recorded[label] = {"sec": time.perf_counter() - started, "route": route, "trip": trip}
        return trip, route

    trip_orchestrator.select_trip_by_model = timed
    try:
        started = time.perf_counter()
        plans = trip_orchestrator.build_trip_plans(request, deal_breakers, num_days, signal_days)
        total_sec = time.perf_counter() - started
    finally:
        trip_orchestrator.select_trip_by_model = real

    base = trip_orchestrator.trip_seed(request, signal_days)
    ranking = build_trip_slot_plan(members, num_days, random.Random(trip_orchestrator.derive_seed(base, "ties"))).days[0].daytime_category_ranking
    model_calls = {}
    for label, index in (("plan1", 1), ("plan2", 2)):
        info = recorded.get(label, {})
        model_calls[label] = {
            "sec": round(info.get("sec", 0.0), 3), "route": info.get("route"), "error": info.get("error"),
            # 오케스트레이터는 유효한 모델 결과를 그대로 돌려주므로, 같은 객체면 모델 결과로 응답한 것이다
            "used_model": info.get("trip") is plans[index],
        }
    overlap = {}
    for name, (a, b) in PAIRS.items():
        pairs = [day_difference(plans[a][day], plans[b][day]) for day in range(num_days)]
        overlap[name] = {"different": [d for d, _ in pairs], "slots": [s for _, s in pairs],
                         "meets": [meets_difference(d, s) for d, s in pairs]}
    return {
        **case, "total_sec": round(total_sec, 3), "model_calls": model_calls, "overlap": overlap,
        "plan_sizes": [[len(day) for day in plan] for plan in plans],
        "extra_category_share": {f"plan{i}": extra_category_share(plan, ranking) for i, plan in enumerate(plans)},
        "outdoor_on_signal_days": {f"plan{i}": outdoor_places_on_days(plan, sorted(signal_days)) for i, plan in enumerate(plans)},
    }


def should_stop(results: list[dict], limit: int) -> bool:
    """마지막 limit건이 모두 모델 안 둘 다 모델 결과로 응답하지 못했으면 True."""
    if limit <= 0 or len(results) < limit:
        return False
    return all(not any(c["used_model"] for c in r["model_calls"].values()) for r in results[-limit:])


def aggregate(results: list[dict], skip_first: int = 0) -> dict:
    """skip_first건은 지연 요약에서 뺀다(첫 요청은 워커가 내려가 있어 콜드 스타트를 포함할 수 있음). 나머지 지표는 모두 쓴다."""
    timed = results[skip_first:]
    summary: dict = {"requests": len(results), "latency_requests": len(timed), "by_length": {}}
    for length in sorted({r["days"] for r in results}):
        rows = [r for r in timed if r["days"] == length]
        summary["by_length"][length] = {
            "total_sec": summarize([r["total_sec"] for r in rows]),
            "model_call_sec": summarize([r["model_calls"][label]["sec"] for r in rows for label in ("plan1", "plan2")]),
        }
    all_calls = [r["model_calls"][label]["sec"] for r in timed for label in ("plan1", "plan2")]
    summary["model_call_p95_sec"] = round(percentile(all_calls, 0.95), 3) if all_calls else None
    summary["total_max_sec"] = round(max((r["total_sec"] for r in timed), default=0.0), 3)
    summary["model_used"] = {
        label: sum(r["model_calls"][label]["used_model"] for r in results) for label in ("plan1", "plan2")
    }
    summary["overlap"] = {}
    for name in PAIRS:
        meets = [m for r in results for m in r["overlap"][name]["meets"]]
        different = [d for r in results for d in r["overlap"][name]["different"]]
        summary["overlap"][name] = {
            "days": len(meets), "meets_rate": round(sum(meets) / len(meets), 3) if meets else None,
            "different_min": min(different, default=None), "different_mean": round(statistics.mean(different), 2) if different else None,
        }
    summary["extra_category_share"] = {}
    for label in ("plan0", "plan1", "plan2"):
        extra = sum(r["extra_category_share"][label][0] for r in results)
        total = sum(r["extra_category_share"][label][1] for r in results)
        summary["extra_category_share"][label] = {"extra": extra, "total": total, "rate": round(extra / total, 3) if total else None}
    summary["outdoor_on_signal_days"] = {
        label: sum(r["outdoor_on_signal_days"][label] for r in results) for label in ("plan0", "plan1", "plan2")
    }
    summary["criteria"] = {
        "model_call_p95_within_15s": None if summary["model_call_p95_sec"] is None else summary["model_call_p95_sec"] <= CALL_LIMIT_SECONDS,
        "total_within_30s": summary["total_max_sec"] <= RESPONSE_LIMIT_SECONDS,
        "no_outdoor_on_signal_days": all(count == 0 for count in summary["outdoor_on_signal_days"].values()),
        "plan1_vs_plan0_meets_every_day": all(m for r in results for m in r["overlap"]["plan1_vs_plan0"]["meets"]),
    }
    return summary


def build_cases(
    regions: list[str], survey_count: int, lengths: list[int], signal_days: list[int], limit: int | None, survey_start: int = 0,
) -> list[dict]:
    cases = [
        {"region": region, "survey": survey, "days": days, "signal_days": signal_days}
        for region in regions for survey in range(survey_start, survey_start + survey_count) for days in lengths
    ]
    random.Random(1).shuffle(cases)  # 상한이 있어도 지역·길이가 골고루 들어가게 순서를 섞는다(같은 인자는 같은 순서)
    return cases[:limit] if limit else cases


def preflight_runpod(url: str, key: str, timeout: float = 60.0, post: Callable | None = None) -> tuple[bool, str]:
    """엔드포인트가 작업을 받는지 작은 요청 1개로 확인한다. 일시 정지(max_workers=0)면 409로 거절된다.
    /health는 일시 정지 여부를 알려 주지 않는다(워커 0개는 정상 상태와 같다). 접수되면 워커를 깨우는 호출 1번이 든다."""
    import httpx

    post = post or httpx.post
    try:
        response = post(
            f"{url.rstrip('/')}/runsync", headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"input": {"header": [1, 3, 5, 7, 9, 11], "slots": [[]], "width": 10}}, timeout=timeout,
        )
    except Exception as error:  # noqa: BLE001  연결 실패, 시간 초과도 시작하지 않는 사유다
        return False, f"{type(error).__name__}"
    try:
        body = response.json()
    except ValueError:
        body = {}
    if response.status_code != 200:
        return False, f"HTTP {response.status_code} {body.get('code') or body.get('title') or ''} {body.get('detail') or ''}".strip()
    if body.get("status") != "COMPLETED":
        return False, f"작업 상태 {body.get('status')}"
    return True, "접수·완료"


def make_router(args: argparse.Namespace):
    from app.trips.services.model.inference import (
        InferenceRouter,
        LocalEngine,
        RunPodEngine,
    )

    if args.runpod:
        if not settings.runpod_model_url:
            raise SystemExit("RUNPOD_MODEL_URL이 필요함 (.env)")
        # CPU 폴백을 두지 않는다. GPU가 실패하면 규칙 변형으로 대체되고 used_model=false로 드러난다
        timeout = args.timeout if args.timeout is not None else settings.runpod_model_timeout_seconds
        return InferenceRouter(RunPodEngine(settings.runpod_model_url, settings.runpod_model_api_key, timeout), None)
    return InferenceRouter(None, LocalEngine(args.model_dir, max_concurrency=2))  # [1], [2]가 동시에 부르므로 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-dir", type=Path, help="로컬 CPU 모델 묶음(model.onnx, codes.json)")
    source.add_argument("--runpod", action="store_true", help="RunPod GPU 호출(유료). --max-calls 필요")
    parser.add_argument("--max-calls", type=int, help="--runpod의 GPU 호출 상한. 요청 한 건이 2번 부른다")
    parser.add_argument("--regions", default=",".join(region.value for region in E_Region))
    parser.add_argument("--surveys", type=int, default=3, help="대표 설문을 몇 개 쓸지")
    parser.add_argument("--survey-start", type=int, default=0, help="대표 설문을 몇 번째부터 쓸지(같은 요청은 결과가 같으므로 이미 잰 설문을 건너뛸 때 쓴다)")
    parser.add_argument("--lengths", default="1,5,9", help="여행 일수 목록")
    parser.add_argument("--signal-days", default="", help="날씨 신호를 줄 일자 번호(0부터, 예: 2,3). 여행보다 긴 번호는 무시")
    parser.add_argument("--limit", type=int, help="요청 수 상한")
    parser.add_argument("--timeout", type=float, help="--runpod의 호출 대기 시간(초). 기본은 서비스 설정(20초)이라 20초가 넘는 호출은 잘린다. 실제 지연을 보려면 60 정도로 늘린다")
    parser.add_argument("--skip-first", type=int, default=None, help="지연 요약에서 뺄 앞쪽 요청 수(기본: runpod 1, 로컬 0)")
    parser.add_argument("--stop-after-fallbacks", type=int, default=3, help="--runpod에서 두 모델 안이 모두 모델 결과로 응답하지 못한 요청이 이만큼 연속되면 중단(0이면 끄기). 엔드포인트가 꺼져 있을 때 호출을 낭비하지 않는다")
    parser.add_argument("--skip-preflight", action="store_true", help="--runpod 시작 전 사전 점검(작은 요청 1개)을 건너뛴다")
    parser.add_argument("--out", type=Path, help="결과 JSON 저장 경로(레포 밖 권장)")
    args = parser.parse_args(argv)

    if args.runpod and not args.max_calls:
        raise SystemExit("--runpod는 --max-calls(GPU 호출 상한)가 필요함. 유료 호출이므로 승인된 값을 준다")
    surveys = json.loads(SURVEYS_PATH.read_text(encoding="utf-8"))
    regions = [name.strip() for name in args.regions.split(",") if name.strip()]
    lengths = [int(value) for value in args.lengths.split(",") if value.strip()]
    signal_days = [int(value) for value in args.signal_days.split(",") if value.strip()]
    limit = args.limit
    if args.runpod:
        limit = min(limit or 10**9, args.max_calls // GPU_CALLS_PER_REQUEST)
    cases = build_cases(regions, min(args.surveys, len(surveys) - args.survey_start), lengths, signal_days, limit, args.survey_start)
    if args.runpod:
        print(f"RunPod 호출: 요청 {len(cases)}건 → GPU 호출 최대 {len(cases) * GPU_CALLS_PER_REQUEST}회 (상한 {args.max_calls})", flush=True)

    if args.runpod and not args.skip_preflight:
        ok, message = preflight_runpod(settings.runpod_model_url, settings.runpod_model_api_key)
        print(f"사전 점검: {message}", flush=True)
        if not ok:
            raise SystemExit("엔드포인트가 작업을 받지 않음. 콘솔에서 Max Workers(일시 정지 해제)와 이미지를 확인한다")
    router = make_router(args)
    runtime.get_router = lambda: router
    results = []
    for number, case in enumerate(cases, start=1):
        result = run_case(case, surveys)
        results.append(result)
        print(f"  {number}/{len(cases)} {case['region']} {case['days']}일 · 전체 {result['total_sec']}초 · "
              f"모델 {result['model_calls']['plan1']['sec']}/{result['model_calls']['plan2']['sec']}초", flush=True)
        if args.runpod and should_stop(results, args.stop_after_fallbacks):
            print(f"중단: 모델 결과가 없는 요청이 {args.stop_after_fallbacks}건 연속됨. 엔드포인트 상태(워커, 이미지, 키)를 확인한다", flush=True)
            break
    skip_first = args.skip_first if args.skip_first is not None else (1 if args.runpod else 0)
    summary = aggregate(results, skip_first)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
