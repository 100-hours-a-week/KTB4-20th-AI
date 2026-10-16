"""여행자 단위로 학습 70% / 개발 15% / 최종 15%로 나눈다 (M1 T1-8, ADR-0005).

같은 여행자의 하루가 두 몫에 걸치면 취향 프로필이 새므로 여행자 단위로 묶는다(이 데이터에서는 여행과 여행자가 1:1).
서비스 지역 비율이 세 몫에서 비슷해지도록, 여행자를 섞은 뒤 하나씩 넣되 넣을 때마다 목표 비율에서 가장 덜 벗어나는 몫을 고른다.

나누기는 한 번만 한다. 이미 결과 파일이 있으면 덮어쓰지 않고 멈춘다.
최종용(final)은 모델을 고치는 데 쓰지 않고, 규칙 대비 비교를 한 번 할 때만 연다. 읽으려면 open_final을 명시해야 한다.

  # 레포 루트에서 실행. days.jsonl은 build_days의 결과
  uv run python -m scripts.semantic_id.split --days "$OUT_DIR/days.jsonl" --out "$OUT_DIR/split.json"
"""

import argparse
import collections
import json
import pathlib
import random
import sys
from collections.abc import Iterator

PARTS = ("train", "dev", "final")
RATIOS = {"train": 0.70, "dev": 0.15, "final": 0.15}
SEED = 20261015
MIN_DAYS_FOR_REGION_VERDICT = 100  # ADR-0005: 최종용 일수가 이보다 적은 지역은 판정 보류


class FinalLockedError(Exception):
    pass


def split_travelers(
    traveler_regions: dict[str, collections.Counter], ratios: dict[str, float] = RATIOS, seed: int = SEED
) -> dict[str, str]:
    # traveler_regions: 여행자 → {지역: 하루 수}. 넣을 때마다 세 몫의 목표 비율 이탈(제곱합)이 가장 작아지는 몫을 고른다
    parts = list(ratios)
    travelers = sorted(traveler_regions)
    random.Random(seed).shuffle(travelers)
    in_part = {p: collections.Counter() for p in parts}
    seen: collections.Counter = collections.Counter()
    result: dict[str, str] = {}
    for t in travelers:
        counts = traveler_regions[t]
        after_seen = seen + counts
        best, best_cost = parts[0], float("inf")
        for p in parts:
            cost = 0.0
            for q in parts:
                got = in_part[q] + counts if q == p else in_part[q]
                cost += sum((got[s] - ratios[q] * after_seen[s]) ** 2 for s in after_seen)
            if cost < best_cost - 1e-9:  # 같으면 앞의 몫(학습)을 고른다
                best, best_cost = p, cost
        in_part[best] += counts
        seen = after_seen
        result[t] = best
    return result


def traveler_regions_from_days(days: list[dict]) -> dict[str, collections.Counter]:
    out: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for d in days:
        out[d["traveler_id"]][d["region"]] += 1
    return out


def summarize(days: list[dict], parts_by_traveler: dict[str, str]) -> dict:
    days_by = {p: collections.Counter() for p in PARTS}
    trips_by: dict[str, set[str]] = {p: set() for p in PARTS}
    travelers_by: dict[str, set[str]] = {p: set() for p in PARTS}
    for d in days:
        part = parts_by_traveler[d["traveler_id"]]
        days_by[part][d["region"]] += 1
        trips_by[part].add(d["travel_id"])
        travelers_by[part].add(d["traveler_id"])
    regions = sorted({r for c in days_by.values() for r in c})
    pending = [r for r in regions if days_by["final"][r] < MIN_DAYS_FOR_REGION_VERDICT]
    return {
        "days": {p: {r: days_by[p][r] for r in regions} | {"합계": sum(days_by[p].values())} for p in PARTS},
        "travelers": {p: len(travelers_by[p]) for p in PARTS},
        "trips": {p: len(trips_by[p]) for p in PARTS},
        "regions_pending_verdict": pending,
    }


def write_split(path: pathlib.Path, parts_by_traveler: dict[str, str], summary: dict, seed: int, ratios: dict) -> None:
    if path.exists():
        raise FileExistsError(f"이미 있어서 덮어쓰지 않음: {path} (다시 나누려면 사람이 직접 지운다)")
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"seed": seed, "ratios": ratios, "summary": summary, "traveler_part": dict(sorted(parts_by_traveler.items()))}
    path.write_text(json.dumps(body, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def load_part(split_path: pathlib.Path, part: str, open_final: bool = False) -> set[str]:
    # part에 속한 여행자 ID 집합. final은 open_final을 명시해야 읽힌다
    if part not in PARTS:
        raise ValueError(f"알 수 없는 몫: {part} (train, dev, final 중 하나)")
    if part == "final" and not open_final:
        raise FinalLockedError("final은 규칙 대비 비교를 한 번 할 때만 연다. 정말 열 때만 open_final=True를 줄 것")
    mapping = json.loads(split_path.read_text(encoding="utf-8"))["traveler_part"]
    return {t for t, p in mapping.items() if p == part}


def iter_days(days_path: pathlib.Path, split_path: pathlib.Path, part: str, open_final: bool = False) -> Iterator[dict]:
    travelers = load_part(split_path, part, open_final)
    with days_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                if d["traveler_id"] in travelers:
                    yield d


def format_summary(summary: dict) -> list[str]:
    lines = []
    for p in PARTS:
        lines.append(f"{p}: 여행자 {summary['travelers'][p]}명, 여행 {summary['trips'][p]}건, 하루 {summary['days'][p]}")
    pending = summary["regions_pending_verdict"]
    lines.append(f"최종용 하루가 {MIN_DAYS_FOR_REGION_VERDICT}일 미만이라 판정 보류인 지역: {pending or '없음'}")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=pathlib.Path, required=True, help="build_days가 만든 days.jsonl")
    ap.add_argument("--out", type=pathlib.Path, required=True, help="나누기 결과 split.json (레포 밖에 둘 것)")
    args = ap.parse_args(argv)
    if args.out.exists():
        print(f"이미 있어서 덮어쓰지 않음: {args.out} (다시 나누려면 사람이 직접 지운다)", file=sys.stderr)
        return 1
    days = [json.loads(line) for line in args.days.read_text(encoding="utf-8").splitlines() if line.strip()]
    parts_by_traveler = split_travelers(traveler_regions_from_days(days))
    summary = summarize(days, parts_by_traveler)
    write_split(args.out, parts_by_traveler, summary, SEED, RATIOS)
    for line in format_summary(summary):
        print(line)
    print(f"저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
