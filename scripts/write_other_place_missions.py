"""학습 데이터 2차 검증: 다른 장소 미션 붙이기.

1차 파일럿의 실패 10개가 모두 "대상 없음"이었다. "다른 장소" 실패를 배우도록,
역사·유적 사진에 같은 지역의 5km 안 다른 역사·유적 장소의 미션을 붙인다.
서비스는 사진 좌표가 장소에서 5km 넘게 떨어지면 AI 전에 거르므로(pipeline_verify.CLEAR_MISMATCH_KM),
5km 안의 다른 장소가 실제로 모델까지 들어오는 "다른 장소" 사례다.
미션은 서비스 generate로 그 다른 장소 이름을 넣어 만든다. 정답은 사람이 표시한다(무조건 실패로 두지 않는다).

  # 레포 루트에서 실행. 호출 없이 계획만 확인
  uv run python -m scripts.write_other_place_missions --aihub /Volumes/T7/aihub --pilot-dir /Volumes/T7/aihub/train/pilot --dry-run
  # 실제 호출 (서비스 generate, 비용 발생)
  uv run python -m scripts.write_other_place_missions --aihub /Volumes/T7/aihub --pilot-dir /Volumes/T7/aihub/train/pilot

  뽑은 사진은 pilot-dir/manifest_pilot2.jsonl, 미션은 pilot-dir/missions.jsonl에 batch "pilot2"로 쌓인다.
"""

import argparse
import asyncio
import hashlib
import json
import pathlib
import random

from fastapi import HTTPException

from app.photomissions import gemini_client
from app.photomissions.pipeline_generate import generate_missions
from app.photomissions.pipeline_verify import CLEAR_MISMATCH_KM, _haversine_km
from app.photomissions.schemas import Coordinates
from scripts.baseline_verify import index_photos, read_photo
from scripts.build_eval_category import DATASETS, REGIONS, collect_candidates
from scripts.build_train_pilot import eval_photo_hashes, pick
from scripts.generate_eval_missions import make_batches, to_request

PER_REGION = 2
SEED = 20261011
# 방문지 유형 2번(역사·유적·종교 시설)만 쓴다. 3번(문화시설)에는 영화관·공연장·팝업 매장이 섞여 있어
# 미션이 "대상 없음" 실패만 만들고, 사진으로 장소를 구별하는 "다른 장소" 사례가 되지 않는다 (2026-10-11 dry run에서 확인)
SITE_TYPE = {"2": "HISTORY_CULTURE"}


def pick_other_place(photo: dict, places: list[dict], seed: int) -> dict | None:
    # 같은 지역, 이름이 다르고 5km 안에 있는 장소 중 하나. 없으면 None
    here = Coordinates(**photo["place_coords"])
    near = sorted(
        (p for p in places if p["region"] == photo["region"] and p["place_name"] != photo["place_name"]
         and _haversine_km(here, Coordinates(**p["place_coords"])) <= CLEAR_MISMATCH_KM),
        key=lambda p: p["place_name"],
    )
    return random.Random(f"{seed}-{photo['photo']}").choice(near) if near else None


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aihub", type=pathlib.Path, required=True)
    ap.add_argument("--pilot-dir", type=pathlib.Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    manifest = args.pilot_dir / "manifest_pilot2.jsonl"
    out = args.pilot_dir / "missions.jsonl"
    done = {json.loads(x)["mission_id"] for x in out.read_text(encoding="utf-8").splitlines() if x and "error" not in x}

    if manifest.exists():
        rows = [json.loads(x) for x in manifest.read_text(encoding="utf-8").splitlines() if x]
    else:
        # 평가셋 여행·사진과 1차 파일럿 여행을 빼고 역사·문화 사진을 뽑는다
        excluded = set((args.aihub / "eval_b/excluded_travel_ids.txt").read_text(encoding="utf-8").split())
        excluded |= {json.loads(x)["travel_id"] for x in (args.pilot_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if x}
        index = index_photos([args.aihub / rel for paths in DATASETS.values() for rel in paths["photos"]])
        eval_hashes = eval_photo_hashes(args.aihub, index)
        candidates = collect_candidates(args.aihub, SITE_TYPE)
        history = [c for c in candidates if c["category"] == "HISTORY_CULTURE"]
        places = list({(c["region"], c["place_name"]): c for c in history}.values())
        picked = pick(history, ("HISTORY_CULTURE",), excluded, SEED,
                      lambda c: hashlib.md5(read_photo(index[c["photo"]])).hexdigest() in eval_hashes)
        rows = []
        for region in REGIONS:
            for c in (p for p in picked if p["region"] == region):
                other = pick_other_place(c, places, SEED)
                if other is None or sum(r["region"] == region for r in rows) >= PER_REGION:
                    continue
                distance = _haversine_km(Coordinates(**c["place_coords"]), Coordinates(**other["place_coords"]))
                rows.append({**c, "role": "other_place", "target_place": other["place_name"],
                             "distance_km": round(distance, 2)})
        if not args.dry_run:
            manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    todo = [r for r in rows if f"{r['photo']}#other" not in done]
    print(f"사진 {len(rows)}장 (지역마다 최대 {PER_REGION}장) · 남은 것 {len(todo)}장")
    for r in rows:
        print(f"  {r['region']} 사진: {r['place_name']} → 미션 장소: {r['target_place']} ({r['distance_km']}km)")
    batches = make_batches([{**r, "place_name": r["target_place"]} for r in todo], batch_size=5, seed=SEED)
    print(f"서비스 generate {len(batches)}회")
    if args.dry_run:
        return

    with out.open("a", encoding="utf-8") as f:
        for i, batch in enumerate(batches):
            try:
                response = await generate_missions(to_request(batch, f"train_pilot2_other_{i:03d}"))
                missions = {m.place_id: m for m in response.missions}
                results = [{"mission": missions[r["photo"]].description, "scope": missions[r["photo"]].scope,
                            "usable": True} for r in batch]
            except HTTPException as e:
                results = [{"error": f"{e.status_code}: {e.detail}"} for _ in batch]
            for row, result in zip(batch, results, strict=True):
                orig = next(r for r in rows if r["photo"] == row["photo"])
                record = {"photo": row["photo"], "travel_id": row["travel_id"], "place": orig["place_name"],
                          "region": row["region"], "category": row["category"], "mission_id": f"{row['photo']}#other",
                          "source": "other_place", "parent_mission_id": None, "with_condition": False,
                          "target_place": row["target_place"], "distance_km": row["distance_km"],
                          "model": gemini_client.MODEL_NAME, "batch": "pilot2", **result}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
            print(f"[다른 장소] {batch[0]['region']}: "
                  + " / ".join(f"{r['target_place']}: {x.get('mission') or x.get('error')}" for r, x in zip(batch, results, strict=True)))


if __name__ == "__main__":
    asyncio.run(main())
