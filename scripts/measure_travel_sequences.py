"""여행로그의 방문지를 하루 단위 시퀀스로 묶어 길이 분포를 잰다.

시퀀스 생성 모델은 '앞 장소들을 보고 다음 장소를 맞히는' 문제로 학습한다. 길이 n 시퀀스는 학습 문제를 n-1개 만든다.
그래서 하루 5슬롯을 생성하려면 길이 3 이상 시퀀스가 서비스 지역마다 얼마나 있는지 먼저 봐야 한다.

하루 = 같은 TRAVEL_ID, 같은 VISIT_START_YMD. 순서는 VISIT_ORDER를 따른다. 집·친구집·사무실·숙소는 뺀다.
세 가지 기준으로 센다: 만족도가 있는 방문 전체, 만족도 4 이상인 방문만,
만족도·재방문 의향·추천 의향이 모두 4 이상인 방문만.

  # 레포 루트에서 실행. 권역별 zip 전체
  uv run python -m scripts.measure_travel_sequences --aihub /Volumes/T7/aihub

  # 받은 zip을 직접 지정 (.part0 그대로 읽는다)
  uv run python -m scripts.measure_travel_sequences --label-zip "<경로>/TL_csv.zip.part0" --label-zip "<경로>/VL_csv.zip.part0"s

  # 방문지정보 CSV 한 개만 (샘플 재현용)
  uv run python -m scripts.measure_travel_sequences --visit-csv "../Sample/02.라벨링데이터/csv/tn_visit_area_info_방문지정보_A.csv"
"""

import argparse
import collections
import csv
import pathlib

from scripts.build_eval_category import DATASETS, read_csv_in_zip, region_of

# 집, 친구/친지집, 사무실, 숙소: 관광 장소가 아니어서 시퀀스에서 뺀다
NON_PLACE_TYPES = frozenset({"21", "22", "23", "24"})
SATISFIED_FROM = 4
SCORE_FIELDS = ("DGSTFN", "REVISIT_INTENTION", "RCMDTN_INTENTION")  # 만족도, 재방문 의향, 추천 의향
MAX_BUCKET = 5  # 길이 5 이상은 한 묶음으로 센다


def to_day_sequences(visits: list[dict]) -> dict[tuple[str, str], list[dict]]:
    # (TRAVEL_ID, 방문일) → VISIT_ORDER 순서의 방문 목록. 집·숙소는 뺀다
    days: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for v in visits:
        if v["VISIT_AREA_TYPE_CD"] in NON_PLACE_TYPES:
            continue
        days[(v["TRAVEL_ID"], v["VISIT_START_YMD"])].append(v)
    for rows in days.values():
        rows.sort(key=lambda r: int(r["VISIT_ORDER"]))
    return days


def sequence_region(rows: list[dict]) -> str:
    # 하루의 첫 번째로 서비스 지역 주소가 잡히는 방문의 지역. 없으면 "기타"
    for r in rows:
        region = region_of(r["ROAD_NM_ADDR"] or r["LOTNO_ADDR"] or "")
        if region:
            return region
    return "기타"


def _scores_at_least(v: dict, fields: tuple[str, ...], minimum: int) -> bool:
    # 지정한 점수 칼럼이 모두 채워져 있고 minimum 이상인가
    return all(v[f].strip() and int(float(v[f])) >= minimum for f in fields)


def summarize(
    visits: list[dict], min_satisfaction: int | None, fields: tuple[str, ...] = ("DGSTFN",)
) -> dict[str, dict]:
    # 지역별 {days, visits, lengths{1..5+}, examples(다음 장소 맞히기 문제 수), len3_plus}
    # min_satisfaction이 None이면 만족도 칼럼이 채워진 방문만, 숫자면 fields가 모두 그 이상인 방문만 남긴다
    if min_satisfaction is None:
        kept = [v for v in visits if v["DGSTFN"].strip()]
    else:
        kept = [v for v in visits if _scores_at_least(v, fields, min_satisfaction)]
    out: dict[str, dict] = collections.defaultdict(
        lambda: {"days": 0, "visits": 0, "lengths": collections.Counter(), "examples": 0, "len3_plus": 0}
    )
    for rows in to_day_sequences(kept).values():
        n = len(rows)
        for key in (sequence_region(rows), "전체"):
            s = out[key]
            s["days"] += 1
            s["visits"] += n
            s["lengths"][min(n, MAX_BUCKET)] += 1
            s["examples"] += n - 1
            s["len3_plus"] += n >= 3
    return out


def print_summary(title: str, result: dict[str, dict]) -> None:
    print(f"\n[{title}]")
    for region in ("전체", "서울", "부산", "경주", "전주", "제주", "기타"):
        s = result.get(region)
        if not s:
            continue
        lengths = " ".join(f"{k}:{s['lengths'][k]}" for k in range(1, MAX_BUCKET + 1))
        print(
            f"  {region}: 하루 {s['days']}개, 방문 {s['visits']}건, 평균 {s['visits'] / s['days']:.2f}, "
            f"길이 3+ {s['len3_plus']}개, 학습 문제 {s['examples']}개 | 길이 {lengths}"
        )


def load_visits(args: argparse.Namespace) -> list[dict]:
    if args.label_zip:
        visits = []
        for path in args.label_zip:
            rows = read_csv_in_zip(path, "tn_visit_area_info")
            print(f"{path.name} ({path.parent.parent.parent.parent.name}): 방문 {len(rows)}건")
            visits.extend(rows)
        return visits
    if args.visit_csv:
        with open(args.visit_csv, encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    visits = []
    for dataset, paths in DATASETS.items():
        rows = read_csv_in_zip(args.aihub / paths["labels"], "tn_visit_area_info")
        print(f"{dataset}: 방문 {len(rows)}건")
        visits.extend(rows)
    return visits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--aihub", type=pathlib.Path, help="AI Hub 데이터 폴더 (예: /Volumes/T7/aihub)")
    src.add_argument("--label-zip", type=pathlib.Path, action="append", help="TL_csv.zip 또는 VL_csv.zip (여러 번 지정)")
    src.add_argument("--visit-csv", type=pathlib.Path, help="tn_visit_area_info CSV 한 개")
    args = ap.parse_args()

    visits = load_visits(args)
    print_summary("만족도가 있는 방문 전체", summarize(visits, None))
    print_summary(f"만족도 {SATISFIED_FROM} 이상 방문만", summarize(visits, SATISFIED_FROM))
    print_summary(
        f"만족도·재방문 의향·추천 의향 모두 {SATISFIED_FROM} 이상",
        summarize(visits, SATISFIED_FROM, SCORE_FIELDS),
    )


if __name__ == "__main__":
    main()
