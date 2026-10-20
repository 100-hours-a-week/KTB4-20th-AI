"""여행로그를 하루 단위 학습 예제로 만든다 (M1 T1-2~T1-5, ADR-0005).

단계마다 남는 하루 수를 깔때기로 보고한다.
  1. 만족도·재방문 의향·추천 의향이 모두 4 이상인 방문만 남기고(집·숙소 제외) 하루로 묶는다
  2. 여행 메타(`tn_travel`)가 있는 여행만 남긴다
  3. 취향 벡터(미션 → 4개 카테고리)에 1이 하나 이상 있는 여행만 남긴다
  4. 서비스 5개 지역으로 판정되는 하루만 남긴다

결과: <out-dir>/days.jsonl (하루 한 줄, 여행 ID·방문일 순)과 <out-dir>/funnel.json

  # 레포 루트에서 실행. 결과는 레포 밖 경로에 둘 것
  uv run python -m scripts.semantic_id.build_days --data-root "$AIHUB_ROOT" --out-dir "$OUT_DIR"
"""

import argparse
import collections
import json
import pathlib
import sys

from scripts.semantic_id import loader
from scripts.semantic_id.preference import (
    has_preference,
    preference_vector,
    unknown_codes,
)
from scripts.semantic_id.sequences import (
    SERVICE_REGIONS,
    day_region,
    group_days,
    visit_record,
)

MAX_LEN_BUCKET = 5  # 길이 5 이상은 한 묶음으로 센다


def build_days(visits: list[dict], meta: list[dict]) -> tuple[list[dict], dict]:
    meta_by_trip = {r["TRAVEL_ID"]: r for r in meta}
    days = group_days(visits)
    funnel = {"days_satisfied": len(days)}

    with_meta = {key: rows for key, rows in days.items() if key[0] in meta_by_trip}
    funnel["days_with_meta"] = len(with_meta)

    unknown: collections.Counter = collections.Counter()
    examples = []
    with_preference = 0
    for (travel_id, date), rows in sorted(with_meta.items()):
        row = meta_by_trip[travel_id]
        unknown.update(unknown_codes(row["TRAVEL_MISSION"]))
        vector = preference_vector(row["TRAVEL_MISSION"])
        if not has_preference(vector):
            continue
        with_preference += 1
        region = day_region(rows)
        if region not in SERVICE_REGIONS:
            continue
        examples.append({
            "travel_id": travel_id,
            "traveler_id": row["TRAVELER_ID"],
            "date": date,
            "region": region,
            "preference": vector,
            "visits": [visit_record(v) for v in rows],
        })
    funnel["days_with_preference"] = with_preference
    funnel["days_service_regions"] = len(examples)
    funnel["visits"] = sum(len(e["visits"]) for e in examples)  # 시작 표시를 포함한 학습 문제 수와 같다
    funnel["trips"] = len({e["travel_id"] for e in examples})
    funnel["days_by_region"] = dict(collections.Counter(e["region"] for e in examples))
    lengths = collections.Counter(min(len(e["visits"]), MAX_LEN_BUCKET) for e in examples)
    funnel["length_distribution"] = {str(k): lengths[k] for k in sorted(lengths)}
    funnel["unknown_mission_codes"] = dict(unknown)
    return examples, funnel


def write_outputs(out_dir: pathlib.Path, examples: list[dict], funnel: dict, overwrite: bool) -> None:
    days_path, funnel_path = out_dir / "days.jsonl", out_dir / "funnel.json"
    existing = [p.name for p in (days_path, funnel_path) if p.exists()]
    if existing and not overwrite:
        raise loader.DataRootError(f"이미 있어서 덮어쓰지 않음: {', '.join(existing)} (--overwrite로 다시 만든다)")
    out_dir.mkdir(parents=True, exist_ok=True)
    with days_path.open("w", encoding="utf-8") as f:
        for e in examples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    funnel_path.write_text(json.dumps(funnel, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def format_funnel(funnel: dict) -> list[str]:
    return [
        f"하루(세 값 모두 4 이상, 집·숙소 제외): {funnel['days_satisfied']}",
        f"  → 여행 메타 있음: {funnel['days_with_meta']}",
        f"  → 취향 카테고리 1개 이상: {funnel['days_with_preference']}",
        f"  → 서비스 5개 지역: {funnel['days_service_regions']} (여행 {funnel['trips']}건)",
        f"학습 문제(시작 표시 포함, 방문 수): {funnel['visits']}",
        f"지역별 하루: {funnel['days_by_region']}",
        f"길이 분포(5는 5 이상): {funnel['length_distribution']}",
        f"매핑표에 없는 미션 코드: {funnel['unknown_mission_codes'] or '없음'}",
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=pathlib.Path, help=f"AI-Hub 폴더 (없으면 환경 변수 {loader.ENV_NAME})")
    ap.add_argument("--label-zip", type=pathlib.Path, action="append", help="라벨링 zip 직접 지정 (여러 번). 8개 확인을 건너뜀")
    ap.add_argument("--out-dir", type=pathlib.Path, required=True, help="결과 폴더 (레포 밖에 둘 것)")
    ap.add_argument("--overwrite", action="store_true", help="이미 있는 결과 파일을 덮어쓴다")
    args = ap.parse_args(argv)
    try:
        if args.label_zip:
            zips = [loader.zip_from_path(p) for p in args.label_zip]
        else:
            root = loader.resolve_data_root(args.data_root)
            if root is None:
                ap.error(f"--data-root 또는 환경 변수 {loader.ENV_NAME}가 필요함")
            zips = loader.find_label_zips(root)
        examples, funnel = build_days(
            loader.load_rows(zips, "tn_visit_area_info"), loader.load_rows(zips, "tn_travel_")
        )
        write_outputs(args.out_dir, examples, funnel, args.overwrite)
    except loader.DataRootError as e:
        print(e, file=sys.stderr)
        return 1
    for line in format_funnel(funnel):
        print(line)
    print(f"저장: {args.out_dir / 'days.jsonl'}, {args.out_dir / 'funnel.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
