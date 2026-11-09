"""학습 데이터 550개(1차 학습)의 후보 사진 고르기.

계획서 §4 구성을 따른다. 이 스크립트는 "후보"만 고른다. 정답을 사람이 표시한 뒤 build_train_dataset에서
보정(Q2 다시 보기)·검증·제외를 다시 적용하고 550개를 확정한다.

순서 (앞에서 쓴 여행·방문지는 뒤에서도 상한에 들어간다)
  1. 파일럿: 표시가 끝난 미션을 재사용한다. 미션 애매, 자동 다른 장소(other_place), 제외 여행·진단 장소는 뺀다
  2. 닮은 장소: 기존 짝(lookalike_split의 train)과 새 짝(천지연폭포↔정방폭포). 사진마다 own·wrong 미션 2개
  3. 보강용 여행: 남은 일반 후보 여행 중 20%를 고정 시드로 떼어 둔다 (11/2~3 보강 학습용)
  4. 일반: 카테고리마다 사진 보고 쓰기(쓸 것 + 예비)와 서비스 미션. 조건 바꾸기는 사진 보고 쓰기에서 파생한다

제외 (여행 ID 단위)
  - 평가셋 여행(eval_b/excluded_travel_ids.txt), 진단 여행(lookalike_split의 diag_seen·diag_unseen)
  - 종묘·김녕·월정리를 다녀온 여행 (사진 후보가 아니라 전체 방문 기록에서 찾는다). 처음 보는 장소 진단용이다
  - 평가셋·진단 사진과 파일 내용이 같은 사진, 이미 고른 사진과 파일 내용이 같은 사진

  # 레포 루트에서 실행. 결과는 레포 밖에 둘 것
  uv run python -m scripts.build_train_550 --aihub /Volumes/T7/aihub --out-dir /Volumes/T7/aihub/train/run550 --dry-run
  uv run python -m scripts.build_train_550 --aihub /Volumes/T7/aihub --out-dir /Volumes/T7/aihub/train/run550

  이미 결과 파일이 있으면 덮어쓰지 않고 멈춘다.
"""

import argparse
import hashlib
import json
import pathlib
import random
from collections import Counter
from collections.abc import Callable, Iterable

from scripts.aihub_common import DATASETS, read_csv_in_zip
from scripts.baseline_verify import index_photos, read_photo
from scripts.build_eval_category import REGIONS, collect_candidates
from scripts.build_train_dataset import load_missions, merge_labels
from scripts.build_train_pilot import (
    TRAIN_CATEGORIES,
    TRAIN_TYPE_TO_CATEGORY,
    eval_photo_hashes,
)

SEED = 20261029
CRITERIA = "label_criteria.md 2026-10-26 (Q2 경계) · 지시문 v3"
MAX_PER_TRAVEL = 3  # 여행당 최대 3장, 같은 (여행, 방문지)는 1장. 상한일 뿐 채울 의무는 없다
RESERVE_RATIO = 0.2

# 진단용 장소. 이름에 들어가면 사진·목표 모두 막고, 그 장소를 다녀온 여행은 통째로 뺀다
# "월정"으로 찾으면 기존 짝인 월정교·월정사까지 걸리므로 "월정리"로 찾는다
BLOCKED_KEYS = ("종묘", "김녕", "월정리")

# 카테고리마다 사진 66장: 사진 보고 쓰기 24개 + 예비 18장, 서비스 미션 24장. 파일럿(4 + 예비 3, 4)의 6배.
# 앞 48장은 두 방법이 지역을 고르게 나눠 갖도록 번갈아 배정한다. 미션 작성기는 순서대로 쓰다가 개수를 채우면 멈춘다
PW_PER_CATEGORY, GEN_PER_CATEGORY, PW_SPARE = 24, 24, 18
ROLE_SEQ = ("photo_written", "service_generate") * GEN_PER_CATEGORY + ("photo_written",) * PW_SPARE
# 일반 예비 목록: 카테고리마다 사진 보고 쓰기 6 + 서비스 8. 지금은 미션을 만들지 않는다.
# 1차 표시 뒤 부족한 등급·문항 유형·카테고리부터 이 목록에서 꺼내 쓴다 (탈락률은 출처마다 달라 미리 정할 수 없다)
SPARE_SEQ = ("photo_written", "service_generate") * 6 + ("service_generate",) * 2

# 닮은 장소: 사진 1장에 기록 장소(own)와 닮은 장소(wrong) 미션. 서울 궁궐은 나머지 3곳 중 사진마다 고정 시드로 1곳
SEOUL_PALACES = ("경복궁", "창덕궁", "창경궁", "덕수궁")
PAIR_OF = {"월정교": ("동궁과월지",), "동궁과월지": ("월정교",), "천지연폭포": ("정방폭포",), "정방폭포": ("천지연폭포",),
           **{p: tuple(q for q in SEOUL_PALACES if q != p) for p in SEOUL_PALACES}}
# 기존 짝 약 50장. 학습 사진이 여행 34개뿐이라, 닮은 장소만 같은 방문지 두 번째 사진을 허용해야 채울 수 있다
# (같은 방문지 1장이면 최대 36장, 2장이면 64장. 2026-10-29 집계)
OLD_PAIR_QUOTA = {"경복궁": 12, "창덕궁": 9, "창경궁": 6, "덕수궁": 10, "월정교": 7, "동궁과월지": 6}
LOOKALIKE_VISIT_CAP = 2
NEW_PAIR_QUOTA = {"천지연폭포": 10, "정방폭포": 10}
TEMPLATE = "{place}{josa} 잘 보이게 사진 찍기"  # 평가셋 A·닮은 장소 다시 표시와 같은 문구
USABLE = ("clear", "partial")


def blocked_place(name: str | None) -> bool:
    return bool(name) and any(key in name for key in BLOCKED_KEYS)


def blocked_travels(visits: Iterable[tuple[str, str]], eval_travels: set[str], diag_travels: set[str]) -> set[str]:
    # visits: 전체 방문 기록의 (여행 ID, 방문지 이름)
    return {t for t, name in visits if blocked_place(name)} | eval_travels | diag_travels


class Usage:
    """모든 출처에서 고른 사진. 여행당 장수와 (여행, 방문지)를 함께 센다."""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.photos: set[str] = set()
        self.per_travel: Counter = Counter()
        self.visits: Counter = Counter()

    def fits(self, c: dict, cap: int = MAX_PER_TRAVEL, visit_cap: int = 1) -> bool:
        return (c["photo"] not in self.photos and self.per_travel[c["travel_id"]] < cap
                and self.visits[(c["travel_id"], c["visit_area_id"])] < visit_cap)

    def add(self, c: dict) -> None:
        self.rows.append(c)
        self.photos.add(c["photo"])
        self.per_travel[c["travel_id"]] += 1
        self.visits[(c["travel_id"], c["visit_area_id"])] += 1


def _take(pools: list[list[dict]], n: int, usage: Usage, excluded: set[str],
          is_excluded_photo: Callable[[dict], bool], visit_cap: int = 1) -> list[dict]:
    # 묶음(지역·장소)을 돌아가며 1장씩. 먼저 새 여행만 쓰고, 모자라면 여행당 2장, 3장까지 허용한다
    # visit_cap이 2면 그래도 모자랄 때만 같은 방문지의 두 번째 사진을 쓴다 (second_in_visit으로 표시)
    # is_excluded_photo는 다른 조건을 다 통과한 사진에만 부른다 (통과하면 바로 고르므로 해시를 기록해도 된다)
    chosen: list[dict] = []
    steps = [(cap, 1) for cap in range(1, MAX_PER_TRAVEL + 1)] + [(MAX_PER_TRAVEL, v) for v in range(2, visit_cap + 1)]
    for cap, vcap in steps:
        rest = [p[:] for p in pools]
        while len(chosen) < n and any(rest):
            for pool in rest:
                while pool:
                    c = pool.pop(0)
                    if c["travel_id"] in excluded or not usage.fits(c, cap, vcap) or is_excluded_photo(c):
                        continue
                    second = usage.visits[(c["travel_id"], c["visit_area_id"])] > 0
                    chosen.append({**c, "second_in_visit": True} if second else c)
                    usage.add(c)
                    break
                if len(chosen) == n:
                    break
        if len(chosen) == n:
            break
    return chosen


def _shuffled(rows: Iterable[dict], key: str) -> list[dict]:
    pool = sorted(rows, key=lambda c: c["photo"])  # 입력 순서와 무관하게
    random.Random(key).shuffle(pool)
    return pool


def pick_general(candidates: list[dict], categories: tuple[str, ...], usage: Usage, excluded: set[str],
                 reserved: set[str], seed: int,
                 is_excluded_photo: Callable[[dict], bool] = lambda _: False,
                 role_seq: tuple[str, ...] = ROLE_SEQ, group: str = "general") -> tuple[list[dict], dict[str, int]]:
    # 카테고리마다 role_seq 길이만큼. 모자라면 다른 카테고리로 채우지 않고 부족으로 돌려준다
    picked, short = [], {}
    for category in categories:
        pools = [_shuffled((c for c in candidates if c["category"] == category and c["region"] == region),
                           f"{seed}-{group}-{category}-{region}") for region in REGIONS]
        chosen = _take(pools, len(role_seq), usage, excluded | reserved, is_excluded_photo)
        picked += [{**c, "group": group, "role": role,
                    "spare": group != "general" or i >= 2 * GEN_PER_CATEGORY}
                   for i, (c, role) in enumerate(zip(chosen, role_seq, strict=False))]
        if len(chosen) < len(role_seq):
            short[category] = len(chosen)
    return picked, short


def pick_lookalike(items: list[dict], quota: dict[str, int], usage: Usage, excluded: set[str], seed: int,
                   is_excluded_photo: Callable[[dict], bool] = lambda _: False, visit_cap: int = 1) -> list[dict]:
    # 장소마다 quota장. 사람이 "쓸 수 있음"(분명히·일부만)으로 표시한 사진만
    # 닮은 장소는 사례가 적어 같은 방문지 두 번째 사진을 허용할 수 있다(visit_cap 2). 독립 사례가 늘어나는 것은 아니고,
    # 거의 같은 구도인지는 메타데이터로 알 수 없어 표시 화면에서 사람이 거른다
    picked = []
    for place, n in quota.items():
        pool = _shuffled((x for x in items if x["place"] == place and x["label"] in USABLE), f"{seed}-{place}")
        picked += _take([pool], n, usage, excluded, is_excluded_photo, visit_cap)
    return picked


def _josa(word: str) -> str:
    code = ord(word[-1]) - 0xAC00
    return "이" if 0 <= code < 11172 and code % 28 else "가"


def lookalike_missions(row: dict) -> list[dict]:
    place = row["place"]
    wrong = random.Random(f"{SEED}-{row['photo']}").choice(PAIR_OF[place])
    return [{"mission_id": f"{row['photo']}#{kind}", "photo": row["photo"], "kind": kind, "place": place,
             "target_place": target, "mission": TEMPLATE.format(place=target, josa=_josa(target))}
            for kind, target in (("own", place), ("wrong", wrong))]


def pilot_keep(missions: dict[str, dict], labels: list[dict], excluded: set[str]) -> list[dict]:
    # 정답이 있는 파일럿 미션 중 재사용할 것. 정답은 아직 고치지 않는다 (Q2 보정은 build_train_dataset에서)
    kept = []
    for label in labels:
        m = missions.get(label["mission_id"])
        if m is None or label.get("final_grade") in (None, "unclear") or m["source"] == "other_place":
            continue
        if m["travel_id"] in excluded or blocked_place(m["place"]) or blocked_place(m.get("target_place")):
            continue
        kept.append(m)
    return kept


def reserve_travels(travels: set[str], keep: set[str], ratio: float, seed: int) -> set[str]:
    pool = sorted(travels - keep)
    random.Random(f"{seed}-reserve").shuffle(pool)
    return set(pool[:round(len(pool) * ratio)])


def _jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aihub", type=pathlib.Path, required=True)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True, help="결과 폴더 (레포 밖에 둘 것)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--dry-run", action="store_true", help="파일을 쓰지 않고 수만 출력")
    args = ap.parse_args()
    aihub = args.aihub

    # 제외 여행
    split = json.loads((aihub / "lookalike/lookalike_split.json").read_text(encoding="utf-8"))
    diag_items = [x for x in split["items"] if x["split"] in ("diag_seen", "diag_unseen")]
    diag_travels = {t for t, v in split["trips"].items() if v["split"] in ("diag_seen", "diag_unseen")}
    diag_travels |= {x["travel_id"] for x in diag_items}  # 여행 표와 사진 목록의 합집합
    eval_travels = set((aihub / "eval_b/excluded_travel_ids.txt").read_text(encoding="utf-8").split())
    visits, photo_visit = [], {}
    for paths in DATASETS.values():
        visits += [(r["TRAVEL_ID"], r["VISIT_AREA_NM"]) for r in read_csv_in_zip(aihub / paths["labels"], "tn_visit_area_info")]
        photo_visit |= {r["PHOTO_FILE_NM"]: r["VISIT_AREA_ID"] for r in read_csv_in_zip(aihub / paths["labels"], "tn_tour_photo")}
    excluded = blocked_travels(visits, eval_travels, diag_travels)

    # 파일 내용이 같은 사진: 평가셋·진단 사진, 그리고 이미 고른 사진
    index = index_photos([aihub / rel for paths in DATASETS.values() for rel in paths["photos"]])
    blocked_hashes = eval_photo_hashes(aihub, index) | {hashlib.md5(read_photo(index[x["photo"]])).hexdigest()
                                                        for x in diag_items}
    seen_hashes: set[str] = set()

    def same_file(c: dict) -> bool:
        h = hashlib.md5(read_photo(index[c["photo"]])).hexdigest()  # 보안용이 아니라 같은 파일 찾기용
        if h in blocked_hashes or h in seen_hashes:
            return True
        seen_hashes.add(h)
        return False

    usage = Usage()

    # 1. 파일럿
    pilot_dir = aihub / "train/pilot"
    pilot_rows = {r["photo"]: r for f in ("manifest.jsonl", "manifest_pilot2.jsonl") for r in _jsonl(pilot_dir / f)}
    missions = load_missions((pilot_dir / "missions.jsonl").read_text(encoding="utf-8").splitlines())
    labels = merge_labels([json.loads((pilot_dir / f).read_text(encoding="utf-8"))["items"]
                           for f in ("train_pilot_labels.json", "train_pilot2_labels.json")])
    pilot = pilot_keep(missions, labels, excluded)
    for photo in sorted({m["photo"] for m in pilot}):
        row = pilot_rows[photo]
        if usage.fits(row) and not same_file(row):
            usage.add(row)
    pilot = [m for m in pilot if m["photo"] in usage.photos]
    for row in pilot_rows.values():  # 재사용하지 않는 파일럿 사진도 다시 뽑지 않는다
        usage.photos.add(row["photo"])

    # 2. 닮은 장소
    def with_visit(x: dict) -> dict | None:
        visit = photo_visit.get(x["photo"])
        return None if visit is None or x["photo"] not in index else {**x, "visit_area_id": visit}

    old_items = [y for x in split["items"] if x["split"] == "train" and (y := with_visit(x))]
    new_items = [y for x in json.loads((aihub / "lookalike_new/new_pair_labels.json").read_text(encoding="utf-8"))["items"]
                 if (y := with_visit(x))]
    old = pick_lookalike(old_items, OLD_PAIR_QUOTA, usage, excluded, args.seed, same_file, LOOKALIKE_VISIT_CAP)
    new = pick_lookalike(new_items, NEW_PAIR_QUOTA, usage, excluded, args.seed, same_file)
    look_rows = [{**x, "group": group} for group, rows in (("lookalike_old", old), ("lookalike_new", new)) for x in rows]
    look_missions = [m for x in look_rows for m in lookalike_missions(x)]

    # 3. 보강용 여행, 4. 일반
    candidates = collect_candidates(aihub, TRAIN_TYPE_TO_CATEGORY)
    candidates = [c for c in candidates if not blocked_place(c["place_name"])]
    eligible = {c["travel_id"] for c in candidates} - excluded
    lookalike_travels = {x["travel_id"] for x in old_items + new_items}
    reserved = reserve_travels(eligible, keep=set(usage.per_travel) | lookalike_travels, ratio=RESERVE_RATIO, seed=args.seed)
    general, short = pick_general(candidates, TRAIN_CATEGORIES, usage, excluded, reserved, args.seed, same_file)
    spare, spare_short = pick_general(candidates, TRAIN_CATEGORIES, usage, excluded, reserved, args.seed, same_file,
                                      role_seq=SPARE_SEQ, group="general_spare")

    # 마지막 검사: 어느 출처든 제외 여행·진단 장소가 섞이면 멈춘다
    bad = [x["photo"] for x in [*pilot, *look_rows, *general, *spare] if x["travel_id"] in excluded
           or blocked_place(x.get("place") or x.get("place_name")) or blocked_place(x.get("target_place"))]
    bad += [x["photo"] for x in [*look_rows, *general, *spare] if x["travel_id"] in reserved]
    bad += [m["mission_id"] for m in look_missions if blocked_place(m["target_place"])]
    if bad:
        raise SystemExit(f"제외해야 할 사진·미션이 섞임 {len(bad)}개 (예: {bad[:3]})")

    pool = {c["travel_id"] for c in candidates}
    print(f"제외 여행 {len(excluded)}개 (평가 {len(eval_travels)} · 진단 {len(diag_travels)} · 종묘·김녕·월정리 방문 포함, "
          f"사진이 없는 여행 포함) · 사진 후보 여행 {len(pool)}개 중 {len(pool & excluded)}개 제외")
    print(f"진단 여행 {len(diag_travels)}개 모두 제외: {sorted(diag_travels) if diag_travels <= excluded else '아님!'}")
    print(f"보강용 여행 {len(reserved)}개 (남은 일반 후보 여행의 {RESERVE_RATIO:.0%})")
    print(f"1. 파일럿 재사용: 미션 {len(pilot)}개 · 사진 {len({m['photo'] for m in pilot})}장 "
          f"· 출처 {dict(Counter(m['source'] for m in pilot))}")
    for group, rows, quota in (("2. 기존 짝", old, OLD_PAIR_QUOTA), ("2. 새 짝", new, NEW_PAIR_QUOTA)):
        got = Counter(x["place"] for x in rows)
        print(f"{group}: 사진 {len(rows)}장 (같은 방문지 두 번째 {sum(bool(x.get('second_in_visit')) for x in rows)}장) "
              f"→ 미션 {2 * len(rows)}개 · " + " ".join(f"{p} {got[p]}/{n}" for p, n in quota.items()))
    # 기존 짝 후보가 왜 빠졌는지 (여행·방문지 상한에 걸린 것은 나머지)
    usable_old = [x for x in split["items"] if x["split"] == "train" and x["label"] in USABLE]
    picked_old = {x["photo"] for x in old}
    lost = Counter("파일럿이 쓴 사진" if x["photo"] in pilot_rows else "제외 여행" if x["travel_id"] in excluded
                   else "방문지 정보 없음" if x["photo"] not in photo_visit else "상한·장소별 개수"
                   for x in usable_old if x["photo"] not in picked_old)
    print(f"   기존 짝 쓸 수 있음 {len(usable_old)}장 중 안 고른 것: {dict(lost)}")
    print(f"4. 일반: 사진 {len(general)}장 (카테고리마다 쓰기 {PW_PER_CATEGORY} + 예비 {PW_SPARE}, 서비스 {GEN_PER_CATEGORY})")
    print("   카테고리 | 사진 보고 쓰기(예비 포함) | 서비스 | 지역")
    for category in TRAIN_CATEGORIES:
        rows = [p for p in general if p["category"] == category]
        roles = Counter(p["role"] for p in rows)
        regions = Counter(p["region"] for p in rows)
        print(f"   {category} | {roles['photo_written']} | {roles['service_generate']} | "
              + " ".join(f"{r}{regions[r]}" for r in REGIONS))
    if short:
        print(f"   부족한 카테고리: {short}")
    if spare_short:
        print(f"   예비 목록 부족: {spare_short}")
    spare_roles = Counter(p["role"] for p in spare)
    print(f"   예비 목록(미션 안 만듦): 사진 {len(spare)}장 · 사진 보고 쓰기 {spare_roles['photo_written']} · "
          f"서비스 {spare_roles['service_generate']}")
    n_cat = len(TRAIN_CATEGORIES)
    base = len(pilot) + 2 * len(look_rows) + n_cat * (PW_PER_CATEGORY + GEN_PER_CATEGORY)
    print(f"예상 예제 (표시 전, 빠지는 것 없다고 칠 때): 파일럿 {len(pilot)} + 닮은 장소 {len(look_rows)}장×2 "
          f"+ 일반 {n_cat}×(쓰기 {PW_PER_CATEGORY} + 서비스 {GEN_PER_CATEGORY}) = {base}")
    print(f"   + 조건 바꾸기: 쓰기 미션 중 사람이 성공으로 표시한 것에서만 만든다. 최대 {n_cat * PW_PER_CATEGORY}개, 지금 확정 0개")
    print(f"전체 사진 {len(usage.rows)}장 · 여행 {len(usage.per_travel)}개 · (여행, 방문지) {len(usage.visits)}쌍 "
          f"· 여행당 최대 {max(usage.per_travel.values())}장")
    if args.dry_run:
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"seed": args.seed, "criteria": CRITERIA}
    outputs = {
        "pilot_reuse.jsonl": [{**m, **meta} for m in pilot],
        "manifest.jsonl": [{**x, **meta} for x in general],  # write_train_missions 입력
        "general_spare.jsonl": [{**x, **meta} for x in spare],  # 예비. 1차 표시 뒤 부족한 유형부터 꺼내 쓴다
        "lookalike_photos.jsonl": [{**x, **meta} for x in look_rows],
        "lookalike_missions.jsonl": [{**m, **meta} for m in look_missions],
    }
    for name, rows in outputs.items():
        with (args.out_dir / name).open("x", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    with (args.out_dir / "reserve_travel_ids.txt").open("x", encoding="utf-8") as f:
        f.write("\n".join(sorted(reserved)) + "\n")
    print(f"저장: {args.out_dir} ({', '.join([*outputs, 'reserve_travel_ids.txt'])})")


if __name__ == "__main__":
    main()
