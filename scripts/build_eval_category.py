"""verify 카테고리별 평가셋(평가셋 B) 사진 추출.

v1 평가셋(A)은 경주 랜드마크(역사·문화)만 다뤄서, 나머지 4개 카테고리(자연·힐링, 음식, 액티비티, 쇼핑·편의)를
서비스 지역 5곳에서 고르게 뽑는다. 사진은 AI Hub zip 안에서 바로 읽고, 결과는 목록 파일(JSONL)로만 남긴다.
정답은 docs/photomissions/label_criteria.md 기준으로 사람이 표시한다.

  # 레포 루트에서 실행. --aihub는 AI Hub 데이터를 둔 폴더, --out-dir은 레포 밖에 둘 것
  uv run python -m scripts.build_eval_category --aihub /Volumes/T7/aihub --out-dir /Volumes/T7/aihub/eval_b --dry-run
  uv run python -m scripts.build_eval_category --aihub /Volumes/T7/aihub --out-dir /Volumes/T7/aihub/eval_b

  결과:
    manifest.jsonl          뽑은 사진 1장당 한 줄 (사진 위치, 방문지, 카테고리, 지역, 좌표)
    excluded_travel_ids.txt 평가셋 A·B에 쓴 여행 ID. 학습 데이터를 뽑을 때 이 여행은 빼야 한다

  평가셋은 한 번 만든 manifest.jsonl이 기준이다. 쓰는 사진 zip이 바뀌면 후보가 달라져 다른 사진이 뽑히므로
  다시 만들지 않는다. 이미 manifest.jsonl이 있으면 덮어쓰지 않고 멈춘다.
"""

import argparse
import csv
import io
import json
import os
import pathlib
import random
import zipfile
from collections import Counter, defaultdict

# AI Hub 방문지 유형 코드(VISIT_AREA_TYPE_CD) → 서비스 카테고리. 역사·문화(2, 3)는 평가셋 A가 다룬다
TYPE_TO_CATEGORY = {
    "1": "NATURE_HEALING", "7": "NATURE_HEALING",  # 자연관광지, 산책로·둘레길
    "11": "FOOD",  # 식당/카페
    "5": "ACTIVITY", "6": "ACTIVITY", "8": "ACTIVITY", "13": "ACTIVITY",  # 레저, 테마시설, 축제, 체험
    "4": "CONVENIENCE_SHOPPING", "10": "CONVENIENCE_SHOPPING",  # 상업지구, 상점
}
CATEGORIES = ("NATURE_HEALING", "FOOD", "ACTIVITY", "CONVENIENCE_SHOPPING")
REGIONS = ("경주", "부산", "서울", "전주", "제주")

# AI Hub 데이터셋 폴더(번호_권역)마다 라벨 CSV zip과 사진 zip 위치. 받은 사진 zip만 쓴다
DATASETS = {
    "277_수도권": {
        "labels": "277_수도권/277.국내 여행로그 데이터(수도권)/01-1.정식개방데이터/Training/02.라벨링데이터/TL_csv.zip",
        "photos": ["277_수도권/277.국내 여행로그 데이터(수도권)-사진/01-1.정식개방데이터/Training/01.원천데이터/TS_photo_4.zip"],
    },
    "278_동부권": {
        "labels": "278_동부권_추가/278.국내 여행로그 데이터(동부권)/01-1.정식개방데이터/Training/02.라벨링데이터/TL_csv.zip",
        "photos": [
            "278_동부권/Training/278.국내 여행로그 데이터(동부권)/01-1.정식개방데이터/Training/01.원천데이터/TS_photo_2.zip",
            "278_동부권_추가/278.국내 여행로그 데이터(동부권)/01-1.정식개방데이터/Training/01.원천데이터/TS_photo_5.zip",
        ],
    },
    "279_서부권": {
        "labels": "279_서부권/279.국내 여행로그 데이터(서부권)/01-1.정식개방데이터/Training/02.라벨링데이터/TL_csv.zip",
        "photos": [
            "279_서부권/279.국내 여행로그 데이터(서부권)/01-1.정식개방데이터/Training/01.원천데이터/TS_photo_4.zip",
            "279_서부권/279.국내 여행로그 데이터(서부권)/01-1.정식개방데이터/Training/01.원천데이터/TS_photo_5.zip",
        ],
    },
    "280_제주": {
        "labels": "280_제주/280.국내 여행로그 데이터(제주도 및 도서지역)/01-1.정식개방데이터/Training/02.라벨링데이터/TL_csv.zip",
        # 다운로드 중 연결이 끊겨 원본 zip이 손상돼, 받아 둔 앞부분(약 88%)에서 CRC가 맞는 사진만 모아 다시 만든 zip
        "photos": ["280_제주/280.국내 여행로그 데이터(제주도 및 도서지역)/01-1.정식개방데이터/Training/01.원천데이터/TS_photo_7_recovered.zip"],
    },
}
# 평가셋 A(경주 랜드마크) 사진 → 여행 ID 매핑. A에 쓴 여행은 학습 데이터에서 뺀다 (B에서는 먼저 골라 다시 쓴다)
EVAL_A_MAPPING = "278_동부권/target_photos_training.tsv"
EVAL_A_LABELS = "labeling/gyeongju_labels.json"


def region_of(address: str) -> str | None:
    # 서비스 지역 5곳만 남긴다. 주소 앞부분으로 지역 판정 (제주는 서귀포 포함)
    if address.startswith("서울"):
        return "서울"
    if address.startswith("부산"):
        return "부산"
    if address.startswith("제주"):
        return "제주"
    head = address[:12]
    if "경주" in head:
        return "경주"
    if "전주" in head:
        return "전주"
    return None

# zip 안의 CSV 읽기
def read_csv_in_zip(zip_path: pathlib.Path, prefix: str) -> list[dict]:
    # AI Hub zip 안의 파일명은 한글이 깨져 있어 앞부분 영문 이름으로 찾는다
    with zipfile.ZipFile(zip_path) as zf:
        name = next((n for n in zf.namelist() if n.strip("/").startswith(prefix)), None)
        if name is None:
            raise SystemExit(f"{zip_path} 안에서 {prefix} CSV를 찾을 수 없음")
        text = zf.read(name).decode("utf-8-sig")
    # 일부 CSV는 헤더 줄이 데이터 중간에 한 번 더 들어 있어 걸러낸다
    return [r for r in csv.DictReader(io.StringIO(text)) if r.get("TRAVEL_ID") != "TRAVEL_ID"]

# 평가셋 A의 여행 알아내기 ( 이 여행들은 나중에 학습 데이터에서 제외한다 )
def eval_a_travel_ids(aihub: pathlib.Path) -> set[str]:
    labeled = {lab["photo"] for lab in json.loads((aihub / EVAL_A_LABELS).read_text(encoding="utf-8"))["labels"]}
    travel_ids = set()
    for line in (aihub / EVAL_A_MAPPING).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        photo, travel_id = line.split("\t")[:2]
        if photo in labeled:
            travel_ids.add(travel_id)
    return travel_ids

# 후보 사진 수집 
# ① 사진 zip에 실제로 들어 있는 사진 파일 목록을 만든다 (located)
# ② 방문지 CSV를 읽어 "여행 + 방문지 → 방문지 정보" 표를 만든다 (areas)
# ③ 사진 CSV를 한 줄씩 보면서
#    - zip에 없는 사진이면 건너뜀
#    - 방문지 유형 → 카테고리, 주소 → 지역, 방문지 좌표를 구함
#    - 카테고리·지역·좌표 중 하나라도 없으면 건너뜀
#    - 통과하면 후보에 추가

def collect_candidates(aihub: pathlib.Path) -> list[dict]:
    # 서비스 지역 5곳 × 카테고리 4개에 해당하고, 받은 사진 zip에 실제로 있는 사진만 후보로 만든다
    candidates = []
    for dataset, paths in DATASETS.items():
        located: dict[str, tuple[str, str]] = {}  # 파일명 → (사진 zip 경로, zip 안 경로)
        for rel in paths["photos"]:
            if not zipfile.is_zipfile(aihub / rel):
                raise SystemExit(f"사진 zip을 열 수 없음 (없거나 손상됨): {aihub / rel}")
            with zipfile.ZipFile(aihub / rel) as zf:
                for member in zf.namelist():
                    if member.lower().endswith(".jpg"):
                        located[os.path.basename(member)] = (rel, member)
        labels = aihub / paths["labels"]
        areas = {(r["TRAVEL_ID"], r["VISIT_AREA_ID"]): r for r in read_csv_in_zip(labels, "tn_visit_area_info")}
        for p in read_csv_in_zip(labels, "tn_tour_photo"):
            if p["PHOTO_FILE_NM"] not in located:
                continue
            area = areas.get((p["TRAVEL_ID"], p["VISIT_AREA_ID"]))
            if area is None:
                continue
            category = TYPE_TO_CATEGORY.get(area["VISIT_AREA_TYPE_CD"])
            region = region_of(area["ROAD_NM_ADDR"] or area["LOTNO_ADDR"] or "")
            place_coords = _coords(area["Y_COORD"], area["X_COORD"])
            # verify는 장소 좌표가 필수라, 좌표가 없는 방문지는 평가에 쓸 수 없어 뺀다
            if category is None or region is None or place_coords is None:
                continue
            zip_rel, member = located[p["PHOTO_FILE_NM"]]
            candidates.append({
                "photo": p["PHOTO_FILE_NM"], "dataset": dataset, "zip": zip_rel, "member": member,
                "travel_id": p["TRAVEL_ID"], "visit_area_id": p["VISIT_AREA_ID"],
                "place_name": area["VISIT_AREA_NM"], "category": category, "region": region,
                "place_coords": place_coords,
                "photo_coords": _coords(p["PHOTO_FILE_Y_COORD"], p["PHOTO_FILE_X_COORD"]),
            })
    return candidates

# 칸마다 20장 고르기
def _coords(lat: str, lon: str) -> dict | None:
    try:
        return {"latitude": float(lat), "longitude": float(lon)}
    except (TypeError, ValueError):
        return None


def sample(candidates: list[dict], per_cell: int, seed: int, eval_travels: set[str]) -> tuple[list[dict], dict]:
    # 카테고리 × 지역 칸마다 per_cell장. 같은 여행 사진은 비슷해서 한 칸에 여행당 1장만 쓴다
    # 평가에 쓴 여행은 학습에서 빠지므로, 이미 평가에 쓴 여행(A와 앞 칸에서 뽑힌 B)을 먼저 골라
    # B 때문에 학습에서 새로 빠지는 여행을 줄인다.
    # 대신 B가 여러 곳을 다닌 여행자 쪽으로 조금 치우친다 (예: 경주 B 사진 상당수가 A의 랜드마크 여행자 사진)
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for c in candidates:
        cells[(c["category"], c["region"])].append(c)
    reused = set(eval_travels)
    picked, shortage = [], {}
    # 후보가 0장인 칸도 "부족"으로 드러나도록, 후보에서 칸을 만들지 않고 정해진 칸을 모두 돈다
    # 순서에 따라 다시 쓰는 여행 수가 달라진다. 2026-10-02 데이터 기준 이름순(ACTIVITY 먼저)이 새로 빠지는 여행이 더 적었다
    # (이름순 126개, CATEGORIES 순 143개)
    for key in sorted((category, region) for category in CATEGORIES for region in REGIONS):
        pool = sorted(cells[key], key=lambda c: c["photo"])         # ① 이름순 정렬
        random.Random(f"{seed}-{key[0]}-{key[1]}").shuffle(pool)    # ② 무작위로 섞기
        pool.sort(key=lambda c: c["travel_id"] not in reused)       # ③ 평가에 쓴 여행을 앞으로
        chosen, seen = [], set()                                    # ④ 여행당 1장씩 20장
        for c in pool:
            if len(chosen) == per_cell:
                break
            if c["travel_id"] in seen:
                continue
            chosen.append(c)
            seen.add(c["travel_id"])
        reused |= seen                                              # ⑤ 고른 여행을 "이미 쓴 여행"에 추가
        if len(chosen) < per_cell:
            shortage[key] = len(chosen)
        picked.extend(chosen)
    return picked, shortage


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aihub", type=pathlib.Path, required=True, help="AI Hub 데이터 폴더 (예: /Volumes/T7/aihub)")
    ap.add_argument("--out-dir", type=pathlib.Path, required=True, help="결과 폴더 (레포 밖에 둘 것)")
    ap.add_argument("--per-cell", type=int, default=20, help="카테고리 × 지역 칸마다 뽑을 사진 수 (기본 20 → 400장)")
    ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--dry-run", action="store_true", help="파일을 쓰지 않고 칸별 수만 출력")
    args = ap.parse_args()

    eval_a_travels = eval_a_travel_ids(args.aihub)
    candidates = collect_candidates(args.aihub)
    picked, shortage = sample(candidates, args.per_cell, args.seed, eval_a_travels)
    new_travels = {p["travel_id"] for p in picked} - eval_a_travels
    excluded = eval_a_travels | new_travels

    print(f"후보 사진 {len(candidates)}장 → 뽑은 사진 {len(picked)}장")
    print(f"평가셋 A 여행 {len(eval_a_travels)}개 + B에서 새로 쓴 여행 {len(new_travels)}개 = 학습에서 뺄 여행 {len(excluded)}개")
    counts = Counter((p["category"], p["region"]) for p in picked)
    print("카테고리 | " + " | ".join(REGIONS))
    for category in CATEGORIES:
        print(f"{category} | " + " | ".join(str(counts[(category, r)]) for r in REGIONS))
    for (category, region), n in shortage.items():
        print(f"  부족: {category} × {region} {n}/{args.per_cell}장")
    if args.dry_run:
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "manifest.jsonl").open("x", encoding="utf-8") as f:
        for p in picked:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    (args.out_dir / "excluded_travel_ids.txt").write_text("\n".join(sorted(excluded)) + "\n", encoding="utf-8")
    print(f"저장: {args.out_dir / 'manifest.jsonl'}, 제외 여행 {len(excluded)}개")


if __name__ == "__main__":
    main()
