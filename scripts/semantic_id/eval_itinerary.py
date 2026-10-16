"""일정 단위 품질 지표를 잰다 (M4 T4-1, ADR-0005 보고 지표).

Recall은 정답 방문을 맞히는지만 본다. 여기서는 개발용 하루마다 서비스처럼 5곳을 만들어(모델은 food-slots 조건, 규칙은 규칙 슬롯)
다음을 비교한다. 실제 로그의 같은 날 방문(앞 5곳)을 참고값으로 함께 적는다.

  카테고리 일치도   FOOD가 아닌 추천 중 그 날 취향 상위 둘 카테고리에 속하는 비율
  연속 거리         이어지는 두 장소 사이 거리(km)의 중앙값과 평균
  고유 장소 수      전체 일정에 나온 서로 다른 장소 수
  상위 집중도       가장 많이 나온 상위 1% 장소가 전체 추천에서 차지하는 비율, 가장 자주 나온 장소 10곳

  uv run --with torch --with numpy python -m scripts.semantic_id.eval_itinerary --v1 "$V" --model-dir "$V/m3/model_a" --part dev
"""

import argparse
import collections
import itertools
import json
import math
import pathlib
import statistics
import sys

from app.trips.services.model.generation import generate_itinerary
from scripts.semantic_id.decode import torch_next_logprobs
from scripts.semantic_id.eval_model import load_inputs
from scripts.semantic_id.split import iter_days


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    (la1, lo1), (la2, lo2) = a, b
    p1, p2 = math.radians(la1), math.radians(la2)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lo2 - lo1) / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(h))


def load_coordinates() -> dict[str, tuple[float, float]]:
    import pymysql

    from app.core.config import settings

    conn = pymysql.connect(host=settings.db_host, port=settings.db_port, user=settings.db_user,
                           password=settings.db_password, database=settings.db_name, charset="utf8mb4")
    try:
        cur = conn.cursor()
        cur.execute("SELECT google_place_id, latitude, longitude FROM ai_places")
        return {g: (float(a), float(o)) for g, a, o in cur.fetchall()}
    finally:
        conn.close()


def summarize(name: str, itineraries: list[list[str]], top_two: list[set[str]], places: dict[str, dict], coords: dict) -> dict:
    counts = collections.Counter(g for it in itineraries for g in it)
    non_food = [(g, t) for it, t in zip(itineraries, top_two, strict=True) for g in it if "FOOD" not in places[g]["categories"] or len(places[g]["categories"]) > 1]
    match = sum(bool(set(places[g]["categories"]) & t) for g, t in non_food) / max(len(non_food), 1)
    dists = [haversine_km(coords[a], coords[b]) for it in itineraries for a, b in itertools.pairwise(it) if a in coords and b in coords]
    total = sum(counts.values())
    top_n = max(1, len(counts) // 100)
    return {
        "method": name, "days": len(itineraries), "places_per_day": round(total / max(len(itineraries), 1), 2),
        "category_match": round(match, 4), "consecutive_km_median": round(statistics.median(dists), 2) if dists else None,
        "consecutive_km_mean": round(statistics.mean(dists), 2) if dists else None, "unique_places": len(counts),
        "top1pct_share": round(sum(c for _, c in counts.most_common(top_n)) / max(total, 1), 4),
        "most_common": [(places[g]["name"], c) for g, c in counts.most_common(10)],
    }


def build_itineraries(examples, by_region, next_logprobs):
    from scripts.semantic_id.eval_model import service_header_and_plan

    model_its, rule_its, popular_its, tops = [], [], [], []
    for ex in examples:
        pool = by_region.get(ex["region"], [])
        header, plan = service_header_and_plan(ex["preference"], ex["region"])
        codes = {p["google_place_id"]: p["code"] for p in pool}
        food = [p["google_place_id"] for p in pool if "FOOD" in p["categories"]]
        day = [p["google_place_id"] for p in pool if any(c != "FOOD" for c in p["categories"])]
        slots = [food if cat == "FOOD" else day for cat in plan[:5]]
        chosen = generate_itinerary(header, slots, codes, next_logprobs)
        model_its.append([g for g in chosen if g])
        used: set[str] = set()
        rule = []
        for cat in plan[:5]:
            pick = next((p["google_place_id"] for p in pool if cat in p["categories"] and p["google_place_id"] not in used), None)
            if pick:
                used.add(pick)
                rule.append(pick)
        rule_its.append(rule)
        popular_its.append([p["google_place_id"] for p in pool[:5]])  # 인기순은 모든 일정이 같은 5곳이 된다
        tops.append({plan[0], plan[3]})
    return model_its, rule_its, popular_its, tops


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v1", type=pathlib.Path, required=True)
    ap.add_argument("--model-dir", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)
    import torch

    from scripts.semantic_id.model import PlaceLM
    from scripts.semantic_id.train_data import build_examples

    places_list, by_region = load_inputs(args.v1)
    places = {p["google_place_id"]: p for p in places_list}
    codes = {p["google_place_id"]: p["code"] for p in places_list if p["code"] is not None}
    visit_map = json.loads((args.v1 / "m3/visit_map.json").read_text(encoding="utf-8"))
    examples = build_examples(iter_days(args.v1 / "m1/days.jsonl", args.v1 / "m1/split.json", "dev"), visit_map, codes)
    examples = [e for e in examples if len(e["gids"]) >= 5 and all(e["targets"][:5])]  # 앞 5곳이 모두 정답으로 쓸 수 있는 하루만
    if args.limit:
        examples = examples[:args.limit]
    ckpt = torch.load(args.model_dir / "model.pt", map_location="cpu")
    model = PlaceLM(**ckpt["config"])
    model.load_state_dict(ckpt["state"])
    model_its, rule_its, popular_its, tops = build_itineraries(examples, by_region, torch_next_logprobs(model, torch.device("cpu")))
    real_its = [e["gids"][:5] for e in examples]
    coords = load_coordinates()
    all_places = {g: places[g] for g in places}
    result = [summarize(name, its, tops, all_places, coords) for name, its in
              (("모델", model_its), ("규칙", rule_its), ("인기순", popular_its), ("실제 로그(참고)", real_its))]
    text = json.dumps(result, ensure_ascii=False, indent=1)
    print(text)
    if args.out:
        if args.out.exists():
            print(f"이미 있어서 덮어쓰지 않음: {args.out}", file=sys.stderr)
            return 1
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
