"""학습 데이터 1차 파일럿: 사진 뽑기.

카테고리 5개(역사·문화 포함)마다 사진을 뽑고, 사진마다 미션을 붙이는 방법(role)을 정한다.
  photo_written    사진을 보고 미션을 쓴다 (write_train_missions.py). 맞는 미션을 못 쓰는 사진이 있어 예비를 더 뽑는다
  service_generate 서비스 generate로 장소만 보고 미션을 만든다
조건 하나만 바꾼 미션(condition_swap)은 photo_written 사진에서 파생하므로 따로 뽑지 않는다.

평가셋과 겹치지 않게 평가셋 여행(excluded_travel_ids.txt)을 빼고, 평가셋 사진과 파일 내용이 같은 사진도 뺀다.
같은 여행 사진은 서로 비슷해서 여행당 1장만 쓰고, 지역 5곳을 돌아가며 뽑는다.

  # 레포 루트에서 실행. 결과는 레포 밖에 둘 것
  uv run python -m scripts.build_train_pilot --aihub /Volumes/T7/aihub --out-dir /Volumes/T7/aihub/train/pilot --dry-run
  uv run python -m scripts.build_train_pilot --aihub /Volumes/T7/aihub --out-dir /Volumes/T7/aihub/train/pilot

  이미 manifest.jsonl이 있으면 덮어쓰지 않고 멈춘다.
"""

import argparse
import hashlib
import json
import pathlib
import random
from collections import Counter
from collections.abc import Callable

from scripts.baseline_verify import index_photos, read_photo
from scripts.build_eval_category import (
    CATEGORIES,
    DATASETS,
    EVAL_A_LABELS,
    REGIONS,
    TYPE_TO_CATEGORY,
    collect_candidates,
)

# 역사·문화(유적지·문화시설) 방문지 유형 코드를 더한다. 평가셋 B는 A가 역사·문화를 다뤄서 뺐었다
TRAIN_TYPE_TO_CATEGORY = {**TYPE_TO_CATEGORY, "2": "HISTORY_CULTURE", "3": "HISTORY_CULTURE"}
TRAIN_CATEGORIES = ("HISTORY_CULTURE", *CATEGORIES)
# 카테고리마다 사진 11장: 사진 보고 쓰기 7장(쓸 4장 + 예비 3장), 서비스 미션 4장.
# 두 방법이 지역을 고르게 나눠 갖도록 번갈아 배정한다
ROLE_ORDER = ("photo_written", "service_generate") * 4 + ("photo_written",) * 3
SEED = 20261010


def pick(
    candidates: list[dict],
    categories: tuple[str, ...],
    excluded_travels: set[str],
    seed: int,
    is_excluded_photo: Callable[[dict], bool] = lambda _: False,
) -> list[dict]:
    picked: list[dict] = []
    used_travels = set(excluded_travels)
    for category in categories:
        by_region = {
            region: sorted((c for c in candidates if c["category"] == category and c["region"] == region),
                           key=lambda c: c["photo"])
            for region in REGIONS
        }
        for region, pool in by_region.items():
            random.Random(f"{seed}-{category}-{region}").shuffle(pool)
        chosen: list[dict] = []
        while len(chosen) < len(ROLE_ORDER) and any(by_region.values()):
            for region in REGIONS:  # 지역을 돌아가며 1장씩
                pool = by_region[region]
                while pool:
                    c = pool.pop(0)
                    if c["travel_id"] in used_travels or is_excluded_photo(c):
                        continue
                    chosen.append(c)
                    used_travels.add(c["travel_id"])
                    break
                if len(chosen) == len(ROLE_ORDER):
                    break
        picked += [{**c, "role": role} for c, role in zip(chosen, ROLE_ORDER, strict=False)]
    return picked


def eval_photo_hashes(aihub: pathlib.Path, index: dict) -> set[str]:
    # 평가셋 A(사람 정답이 있는 200장)와 B(400장, B 보강도 같은 사진) 사진의 파일 내용 해시
    names = {x["photo"] for x in json.loads((aihub / EVAL_A_LABELS).read_text(encoding="utf-8"))["labels"]}
    for line in (aihub / "eval_b/manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            names.add(json.loads(line)["photo"])
    missing = [n for n in names if n not in index]
    if missing:
        raise SystemExit(f"평가셋 사진 {len(missing)}장을 zip에서 찾을 수 없음 (예: {missing[0]})")
    return {hashlib.md5(read_photo(index[n])).hexdigest() for n in names}  # 보안용이 아니라 같은 파일 찾기용


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aihub", type=pathlib.Path, required=True)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True, help="결과 폴더 (레포 밖에 둘 것)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--dry-run", action="store_true", help="파일을 쓰지 않고 뽑은 수만 출력")
    args = ap.parse_args()

    excluded_travels = set((args.aihub / "eval_b/excluded_travel_ids.txt").read_text(encoding="utf-8").split())
    index = index_photos([args.aihub / rel for paths in DATASETS.values() for rel in paths["photos"]])
    eval_hashes = eval_photo_hashes(args.aihub, index)

    def same_as_eval_photo(c: dict) -> bool:
        return hashlib.md5(read_photo(index[c["photo"]])).hexdigest() in eval_hashes

    candidates = collect_candidates(args.aihub, TRAIN_TYPE_TO_CATEGORY)
    picked = pick(candidates, TRAIN_CATEGORIES, excluded_travels, args.seed, same_as_eval_photo)

    print(f"후보 {len(candidates)}장 · 평가셋 여행 {len(excluded_travels)}개 제외 · 평가셋 사진 해시 {len(eval_hashes)}개")
    print("카테고리 | 사진 보고 쓰기 | 서비스 미션 | 지역")
    for category in TRAIN_CATEGORIES:
        rows = [p for p in picked if p["category"] == category]
        roles = Counter(p["role"] for p in rows)
        regions = Counter(p["region"] for p in rows)
        print(f"{category} | {roles['photo_written']} | {roles['service_generate']} | "
              + " ".join(f"{r}{regions[r]}" for r in REGIONS))
    short = [c for c in TRAIN_CATEGORIES if sum(p["category"] == c for p in picked) < len(ROLE_ORDER)]
    if short:
        print(f"  부족한 카테고리: {short}")
    if args.dry_run:
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "manifest.jsonl").open("x", encoding="utf-8") as f:
        for p in picked:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"저장: {args.out_dir / 'manifest.jsonl'} ({len(picked)}장)")


if __name__ == "__main__":
    main()
