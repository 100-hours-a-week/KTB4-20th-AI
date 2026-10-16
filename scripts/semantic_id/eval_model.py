"""생성 모델과 기준선의 장소 단위 Recall@k를 잰다 (M4 T4-1~T4-3, ADR-0005).

각 정답 방문(DB 범위 안으로 판정된 방문)에 대해, 그 앞 방문까지를 정답으로 준 채(teacher forcing) 다음 장소 순위를 만든다.
후보는 같은 지역의 품질 필터 통과 장소 중 오늘 이미 방문한 장소를 뺀 것이다.

세 가지 조건으로 모델을 잰다(--mode). 취향 입력만 다르고 문맥은 모두 앞 방문의 정답이다.
  free        지역 전체 후보에서 순위를 매긴다(슬롯 제약 없음). 취향은 로그의 0/1 그대로(학습과 같은 입력). 모델 자체의 예측력을 본다.
  food-slots  ADR-0003이 정한 후보 제한이다. 2·5번째는 FOOD, 6번째는 ACTIVITY, 나머지는 FOOD가 아닌 카테고리 전체에서 모델이 고른다.
              취향 입력은 상위 둘만 1이다. 서비스가 ADR대로 동작할 때의 조건이다.
  top2-slots  food-slots에 더해 FOOD가 아닌 슬롯은 취향 상위 둘 카테고리의 후보 안에서만 모델이 고른다(어느 쪽 카테고리, 어느 장소인지는 모델이 정함).
  rule-slots  규칙이 슬롯마다 카테고리까지 정하고(점심·저녁 FOOD, 나머지는 취향 상위 둘, 동점은 고정 순서) 모델은 그 안에서만 고른다.
              selection.py가 PR #78 시점에 구현한 조건이다. 분모는 필터 통과 여부와 무관한 모든 정답 방문이다
(필터에 막힌 정답은 어떤 방법도 맞힐 수 없다).

  모델     후보 제한 빔 서치(폭 1, 5, 10). Recall@k는 폭 안에서 k위 안에 정답이 있는 비율
  인기순   후보를 규칙 점수(평점·평가 수) 내림차순으로 놓은 순위
  규칙     ADR-0000 규칙과 같은 슬롯 구조. 위치마다 규칙이 고르는 카테고리(취향 상위 둘, 동점은 무작위, 시드 10개 평균)의
           후보를 규칙 점수 내림차순으로 놓은 순위. 점심·저녁(2, 5번째)은 FOOD, 6번째는 ACTIVITY

신뢰구간은 여행자 단위 묶음 부트스트랩(95%)이고 차이는 같은 정답 방문에 대한 대응 비교다.
최종용은 --open-final을 명시해야 열린다(한 번만).

  uv run --with torch --with numpy python -m scripts.semantic_id.eval_model --v1 "$V" --model-dir "$V/m3/model_a" --part dev
"""

import argparse
import collections
import json
import pathlib
import random
import sys

from scripts.semantic_id.decode import rank_candidates, torch_next_logprobs
from scripts.semantic_id.split import iter_days
from scripts.semantic_id.tokens import CATEGORIES, header_tokens
from scripts.semantic_id.train_data import build_examples

KS = (1, 5, 10)
WIDTHS = (1, 5, 10)
RULE_SEEDS = 10


def rule_slot_categories(preference: dict[str, int], rng: random.Random) -> list[str]:
    """슬롯별 카테고리. 취향 점수 상위 둘(동점은 무작위), 오전·오후1은 1위, 오후2는 2위 (slot_plan.py와 같은 구조)."""
    ranked = sorted(CATEGORIES, key=lambda c: (-preference[c], rng.random()))
    first, second = ranked[0], ranked[1]
    return [first, "FOOD", first, second, "FOOD", "ACTIVITY"]


def rank_of(target: str, ranking: list[str], limit: int) -> int | None:
    # 1부터 시작하는 순위. limit 안에 없으면 None
    for i, gid in enumerate(ranking[:limit], start=1):
        if gid == target:
            return i
    return None


def load_inputs(v1: pathlib.Path):
    places = json.loads((v1 / "m3/places.json").read_text(encoding="utf-8"))
    by_region: dict[str, list[dict]] = collections.defaultdict(list)
    for p in places:
        if p["region"] and p["passes_filter"] and p["code"] is not None:
            by_region[p["region"]].append(p)
    for lst in by_region.values():
        lst.sort(key=lambda p: (-p["rule_score"], p["google_place_id"]))  # 규칙 점수 내림차순, 같으면 id 순
    return places, by_region


FIXED_SLOTS = (1, 4, 5)  # 점심, 저녁, 저녁 이후(ACTIVITY)는 카테고리가 고정이다
SERVICE_TIE_SEED = 0  # app/trips/services/model/selection.py의 TIE_SEED와 같은 값


def service_header_and_plan(preference: dict[str, int], region: str) -> tuple[list[int], list[str]]:
    plan = rule_slot_categories(preference, random.Random(SERVICE_TIE_SEED))
    top_two = {plan[0], plan[3]}
    return header_tokens({c: int(c in top_two) for c in CATEGORIES}, region), plan


def evaluate(examples: list[dict], by_region: dict[str, list[dict]], next_logprobs, widths=WIDTHS, seeds: int = RULE_SEEDS,
             mode: str = "free") -> list[dict]:
    """정답 방문마다 기록을 만든다: 여행자, 지역, 필터 통과 여부, 모델 순위(폭별), 인기순 순위, 규칙 순위(시드별)."""
    records = []
    for ex in examples:
        region = ex["region"]
        rng_slots = [rule_slot_categories(ex["preference"], random.Random(1000 + s)) for s in range(seeds)]
        pool = by_region.get(region, [])
        for j, is_target in enumerate(ex["targets"]):
            if not is_target:
                continue
            target = ex["gids"][j]
            used = set(ex["gids"][:j])
            if mode in ("rule-slots", "food-slots", "top2-slots"):
                header, plan = service_header_and_plan(ex["preference"], region)
                category = plan[j] if j < len(plan) else None

                def allowed(p, category=category, j=j):
                    if mode == "rule-slots" or category in ("FOOD", "ACTIVITY") and j in FIXED_SLOTS:
                        return category in p["categories"]
                    if mode == "top2-slots":
                        return bool(set(p["categories"]) & {plan[0], plan[3]})  # 취향 상위 둘 카테고리 안에서 모델이 고른다
                    return any(c != "FOOD" for c in p["categories"])  # 모델이 FOOD 아닌 카테고리 전체에서 고른다

                candidates = {p["google_place_id"]: p["code"] for p in pool if allowed(p) and p["google_place_id"] not in used}
                context = header + ex["tokens"][6:6 + 4 * j]
            else:
                candidates = {p["google_place_id"]: p["code"] for p in pool if p["google_place_id"] not in used}
                context = ex["tokens"][:6 + 4 * j]
            rec = {"traveler": ex["traveler_id"], "region": region, "position": j, "in_candidates": target in candidates}
            rec["model"] = {}
            if next_logprobs is not None and candidates:
                for w in widths:
                    ranking = rank_candidates(context, candidates, next_logprobs, w)
                    rec["model"][w] = rank_of(target, ranking, 10)
            popular = [p["google_place_id"] for p in pool if p["google_place_id"] not in used]
            rec["popular"] = rank_of(target, popular, 10)
            rule_ranks = []
            for slots in rng_slots:
                category = slots[j] if j < len(slots) else None
                ranking = [p["google_place_id"] for p in pool if category in p["categories"] and p["google_place_id"] not in used]
                rule_ranks.append(rank_of(target, ranking, 10))
            rec["rule"] = rule_ranks
            records.append(rec)
    return records


def hit(rank: int | None, k: int) -> float:
    return float(rank is not None and rank <= k)


def recall(records: list[dict], source: str, k: int, width: int | None = None) -> float:
    if not records:
        return 0.0
    total = 0.0
    for r in records:
        if source == "model":
            total += hit(r["model"].get(width), k)
        elif source == "popular":
            total += hit(r["popular"], k)
        else:
            total += sum(hit(x, k) for x in r["rule"]) / len(r["rule"])
    return total / len(records)


def per_traveler(records: list[dict], metric) -> dict[str, tuple[float, int]]:
    out: dict[str, list[float]] = collections.defaultdict(lambda: [0.0, 0])
    for r in records:
        cell = out[r["traveler"]]
        cell[0] += metric(r)
        cell[1] += 1
    return {t: (v[0], v[1]) for t, v in out.items()}


def bootstrap_diff(records: list[dict], metric_a, metric_b, draws: int = 2000, seed: int = 20261016) -> dict:
    """a - b의 Recall 차이와 여행자 단위 묶음 부트스트랩 95% 신뢰구간."""
    a, b = per_traveler(records, metric_a), per_traveler(records, metric_b)
    travelers = sorted(a)
    if not travelers:
        return {"diff": 0.0, "lo": 0.0, "hi": 0.0, "travelers": 0, "visits": 0}
    n_visits = sum(a[t][1] for t in travelers)
    point = (sum(a[t][0] for t in travelers) - sum(b[t][0] for t in travelers)) / n_visits
    rng = random.Random(seed)
    diffs = []
    for _ in range(draws):
        pick = [travelers[rng.randrange(len(travelers))] for _ in travelers]
        n = sum(a[t][1] for t in pick)
        diffs.append((sum(a[t][0] for t in pick) - sum(b[t][0] for t in pick)) / n)
    diffs.sort()
    return {"diff": point, "lo": diffs[int(0.025 * draws)], "hi": diffs[int(0.975 * draws) - 1], "travelers": len(travelers), "visits": n_visits}


def metric(source: str, k: int, width: int | None = None):
    def fn(r):
        if source == "model":
            return hit(r["model"].get(width), k)
        if source == "popular":
            return hit(r["popular"], k)
        return sum(hit(x, k) for x in r["rule"]) / len(r["rule"])
    return fn


def report(records: list[dict], widths=WIDTHS) -> dict:
    out: dict = {"targets": len(records), "in_candidates": sum(r["in_candidates"] for r in records) / max(len(records), 1)}
    for source in ("popular", "rule"):
        out[source] = {f"recall@{k}": recall(records, source, k) for k in KS}
    out["model"] = {}
    for w in widths:
        out["model"][f"width{w}"] = {f"recall@{k}": recall(records, "model", k, w) for k in KS if k <= w or k == 10}
    regions = sorted({r["region"] for r in records})
    out["by_region"] = {}
    for region in regions:
        sub = [r for r in records if r["region"] == region]
        row = {"targets": len(sub), "travelers": len({r["traveler"] for r in sub}), "in_candidates": sum(r["in_candidates"] for r in sub) / len(sub),
               "rule@10": recall(sub, "rule", 10), "popular@10": recall(sub, "popular", 10)}
        if widths:
            row["model@10"] = recall(sub, "model", 10, max(widths))
            row["diff_vs_rule@10"] = bootstrap_diff(sub, metric("model", 10, max(widths)), metric("rule", 10))
        out["by_region"][region] = row
    if widths:
        out["diff_vs_rule@10_all"] = bootstrap_diff(records, metric("model", 10, max(widths)), metric("rule", 10))
        out["diff_vs_popular@10_all"] = bootstrap_diff(records, metric("model", 10, max(widths)), metric("popular", 10))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v1", type=pathlib.Path, required=True)
    ap.add_argument("--model-dir", type=pathlib.Path, help="없으면 기준선만 잰다")
    ap.add_argument("--part", choices=("dev", "final"), default="dev")
    ap.add_argument("--open-final", action="store_true", help="최종용을 연다. 한 번만 쓸 것")
    ap.add_argument("--mode", choices=("free", "food-slots", "top2-slots", "rule-slots"), default="free", help="모델 후보 조건(위 설명 참고)")
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--limit", type=int, help="시험용: 하루 수 제한")
    args = ap.parse_args(argv)
    if args.part == "final" and not args.open_final:
        print("최종용은 --open-final을 명시해야 연다", file=sys.stderr)
        return 1
    places, by_region = load_inputs(args.v1)
    visit_map = json.loads((args.v1 / "m3/visit_map.json").read_text(encoding="utf-8"))
    codes = {p["google_place_id"]: p["code"] for p in places if p["code"] is not None}
    days = iter_days(args.v1 / "m1/days.jsonl", args.v1 / "m1/split.json", args.part, open_final=args.open_final)
    examples = build_examples(days, visit_map, codes)
    if args.limit:
        examples = examples[:args.limit]
    next_logprobs, widths = None, ()
    if args.model_dir:
        import torch

        from scripts.semantic_id.model import PlaceLM

        ckpt = torch.load(args.model_dir / "model.pt", map_location="cpu")
        model = PlaceLM(**ckpt["config"])
        model.load_state_dict(ckpt["state"])
        next_logprobs, widths = torch_next_logprobs(model, torch.device("cpu")), WIDTHS
    records = evaluate(examples, by_region, next_logprobs, widths, mode=args.mode)
    result = report(records, widths)
    text = json.dumps(result, ensure_ascii=False, indent=1)
    print(text)
    if args.out:
        if args.out.exists():
            print(f"이미 있어서 덮어쓰지 않음: {args.out}", file=sys.stderr)
            return 1
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
