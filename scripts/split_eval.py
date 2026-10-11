"""평가셋(A·B·B 보강)을 여행 단위로 개선용(dev) 60%, 최종용(final) 40%로 나눈다.

개선용으로만 오류를 분석하고 임계값·프롬프트를 고친다. 최종용은 출시 판단 때(10/16) 한 번만 본다.
같은 여행의 사진은 서로 비슷해서, 한 여행이 양쪽에 나뉘면 최종용 점수가 실제보다 좋게 나온다.
A 여행 대부분이 B에도 있으므로 A·B를 합쳐 한 번에 나눈다. B 보강은 B 사진을 쓰므로 B를 따라간다.

여행을 섞은 뒤 하나씩 넣되, 넣을 때마다 A와 B 카테고리별로 60:40에 가장 가까워지는 쪽을 고른다.

  # 레포 루트에서 실행. 결과 파일은 레포 밖에 둘 것
  uv run python -m scripts.split_eval --aihub /Volumes/T7/aihub --out /Volumes/T7/aihub/eval_split.json

  나누기는 한 번만 한다. 이미 결과 파일이 있으면 덮어쓰지 않고 멈춘다.
"""

import argparse
import collections
import datetime
import json
import pathlib
import random
from typing import Literal

from pydantic import BaseModel

from scripts.build_eval_category import EVAL_A_LABELS, EVAL_A_MAPPING

Side = Literal["dev", "final"]
DEV_RATIO = 0.6
SEED = 20261010


class EvalPhoto(BaseModel):
    photo: str
    travel_id: str
    stratum: str  # 비율을 맞출 묶음. "A" 또는 "B:카테고리"


def load_split(path: pathlib.Path, side: str) -> set[str]:
    # split_eval.py가 만든 eval_split.json에서 한쪽(dev 또는 final)의 사진 이름을 읽는다
    return set(json.loads(path.read_text(encoding="utf-8"))["photos"][side])


def split_error(split: str | None, split_file: pathlib.Path | None, open_final: bool, needs_split: bool) -> str | None:
    # final을 실수로 보지 않게 막는다. 측정(baseline_verify·measure_eval_b)과 집계(eval_cases)가 같은 검사를 쓴다. 문제가 없으면 None
    if bool(split) != bool(split_file):
        return "--split과 --split-file은 같이 써야 함"
    if split == "final" and not open_final:
        return "final은 출시 판단 때 한 번만 연다. 정말 열 때만 --open-final을 같이 줄 것"
    if needs_split and not split and not open_final:
        return "평가셋은 --split dev로 dev만 볼 것. 나누기 없이 전체를 보면 final이 섞여 --open-final이 필요함"
    return None


def _gap(dev: collections.Counter, seen: collections.Counter, strata: collections.Counter, ratio: float) -> float:
    # 이 여행이 닿는 묶음마다 "지금까지 넣은 사진 중 개선용 수"가 목표에서 얼마나 벗어났는지
    return sum((dev[s] - ratio * seen[s]) ** 2 for s in strata)


def split_travels(photos: list[EvalPhoto], dev_ratio: float = DEV_RATIO, seed: int = SEED) -> dict[str, Side]:
    by_travel: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for p in photos:
        by_travel[p.travel_id][p.stratum] += 1
    travels = sorted(by_travel)
    random.Random(seed).shuffle(travels)

    dev: collections.Counter = collections.Counter()
    seen: collections.Counter = collections.Counter()
    sides: dict[str, Side] = {}
    for t in travels:
        counts = by_travel[t]
        to_dev = _gap(dev + counts, seen + counts, counts, dev_ratio)
        side: Side = "dev" if to_dev <= _gap(dev, seen + counts, counts, dev_ratio) else "final"
        sides[t] = side
        if side == "dev":
            dev += counts
        seen += counts
    return sides


def summarize(photos: list[EvalPhoto], sides: dict[str, Side]) -> dict[str, tuple[int, int]]:
    table: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for p in photos:
        table[p.stratum][0 if sides[p.travel_id] == "dev" else 1] += 1
    return {s: (d, f) for s, (d, f) in sorted(table.items())}


def load_photos(aihub: pathlib.Path) -> list[EvalPhoto]:
    # 평가셋 A: 사람 정답이 있는 사진 200장. 여행 ID는 사진 목록 TSV에 있다
    labeled = {x["photo"] for x in json.loads((aihub / EVAL_A_LABELS).read_text(encoding="utf-8"))["labels"]}
    photos = []
    for line in (aihub / EVAL_A_MAPPING).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        photo, travel_id = line.split("\t")[:2]
        if photo in labeled:
            photos.append(EvalPhoto(photo=photo, travel_id=travel_id, stratum="A"))
    # 평가셋 B: 400장 (B 보강 172장도 이 사진들이다)
    for line in (aihub / "eval_b/manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            m = json.loads(line)
            photos.append(EvalPhoto(photo=m["photo"], travel_id=m["travel_id"], stratum=f"B:{m['category']}"))
    return photos


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aihub", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out}가 이미 있음. 나누기는 한 번만 한다 (최종용을 본 뒤 다시 나누면 안 됨)")

    photos = load_photos(args.aihub)
    sides = split_travels(photos, seed=args.seed)
    print("묶음 | 개선용 | 최종용 | 개선용 비율")
    for stratum, (d, f) in summarize(photos, sides).items():
        print(f"{stratum} | {d} | {f} | {d / (d + f):.0%}")
    travels = collections.Counter(sides.values())
    print(f"여행 {len(sides)}개 · 개선용 {travels['dev']} · 최종용 {travels['final']}")

    out = {
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "seed": args.seed,
        "dev_ratio": DEV_RATIO,
        "note": "여행 단위로 나눔. 최종용(final)은 출시 판단 때 한 번만 본다",
        "travel_ids": {s: sorted(t for t, v in sides.items() if v == s) for s in ("dev", "final")},
        "photos": {s: sorted({p.photo for p in photos if sides[p.travel_id] == s}) for s in ("dev", "final")},
    }
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
