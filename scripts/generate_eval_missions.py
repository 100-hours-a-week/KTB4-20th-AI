"""평가셋 B 사진마다 미션 문구 생성.

build_eval_category.py가 만든 manifest.jsonl의 방문지로 서비스의 generate를 그대로 불러,
실제 서비스와 같은 프롬프트·모델·재시도로 미션 문구를 만든다. 실제 동선처럼 같은 지역 장소를 묶어 한 번에 호출한다.

  # 레포 루트에서 실행. 호출 없이 묶음 계획만 확인
  uv run python -m scripts.generate_eval_missions --manifest /Volumes/T7/aihub/eval_b/manifest.jsonl \\
      --out /Volumes/T7/aihub/eval_b/missions.jsonl --dry-run
  # 실제 호출 (Gemini 비용 발생)
  uv run python -m scripts.generate_eval_missions --manifest /Volumes/T7/aihub/eval_b/manifest.jsonl \\
      --out /Volumes/T7/aihub/eval_b/missions.jsonl

  같은 --out으로 다시 실행하면 이미 미션이 만들어진 묶음은 건너뛰고, 에러 난 묶음만 다시 호출한다.
"""

import argparse
import asyncio
import json
import pathlib
import random
from collections import defaultdict

from fastapi import HTTPException

from app.photomissions import gemini_client
from app.photomissions.pipeline_generate import generate_missions
from app.photomissions.schemas import (
    DisplayName,
    MissionPlace,
    PhotoMissionGenerateRequest,
)

# generate 1회 비용 추정 (원). verify Pro 실측 1회 약 27~30원을 기준으로 잡은 대략값이라 실제와 다를 수 있다
WON_PER_CALL = 30


def make_batches(rows: list[dict], batch_size: int, seed: int) -> list[list[dict]]:
    # 실제 동선처럼 같은 지역 장소끼리 묶는다. 지역 안에서는 섞어서 카테고리가 고르게 섞이게 한다
    by_region: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_region[row["region"]].append(row)
    batches = []
    for region in sorted(by_region):
        pool = sorted(by_region[region], key=lambda r: r["photo"])
        random.Random(f"{seed}-{region}").shuffle(pool)
        batches += [pool[i : i + batch_size] for i in range(0, len(pool), batch_size)]
    return batches


def to_request(batch: list[dict], batch_id: str) -> PhotoMissionGenerateRequest:
    # 장소 ID는 사진 파일명을 쓴다. 평가 단위가 사진 1장이고, 같은 방문지 사진이 여러 장일 수 있어서다
    return PhotoMissionGenerateRequest(
        itinerary_id=batch_id,
        places=[
            MissionPlace(
                id=row["photo"],
                displayName=DisplayName(text=row["place_name"], languageCode="ko"),
                selected_for=[],
                matched_preferences=[row["category"]],
            )
            for row in batch
        ],
    )


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True, help="결과 JSONL 경로 (레포 밖에 둘 것)")
    ap.add_argument("--batch-size", type=int, default=5, help="한 번에 묶는 장소 수 (실제 동선 하루 장소 수에 맞춤)")
    ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--max-calls", type=int, default=0, help="이번 실행의 최대 호출 수 (0이면 제한 없음)")
    ap.add_argument("--dry-run", action="store_true", help="호출하지 않고 묶음 계획만 출력")
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    batches = make_batches(rows, args.batch_size, args.seed)
    done = set()
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if "error" not in record:
                done.add(record["photo"])
    todo = [(i, b) for i, b in enumerate(batches) if not all(row["photo"] in done for row in b)]
    if args.max_calls:
        todo = todo[: args.max_calls]

    print(f"모델: {gemini_client.MODEL_NAME}")
    print(f"사진 {len(rows)}장 → 묶음 {len(batches)}개 (이미 완료 {len(batches) - len(todo)}개) → 이번에 호출할 것 {len(todo)}개")
    print(f"예상 비용: 약 {len(todo) * WON_PER_CALL:,}원 (1회 약 {WON_PER_CALL}원 추정)")
    if args.dry_run:
        for i, batch in todo[:3]:
            print(f"  묶음 {i} ({batch[0]['region']}): " + ", ".join(f"{r['place_name']}[{r['category']}]" for r in batch))
        return

    with args.out.open("a", encoding="utf-8") as f:
        for n, (i, batch) in enumerate(todo, 1):
            batch_id = f"eval_b_{i:03d}"
            try:
                response = await generate_missions(to_request(batch, batch_id))
                missions = {m.place_id: m for m in response.missions}
                lines = [
                    {"photo": row["photo"], "batch": batch_id, "mission_description": missions[row["photo"]].description,
                     "scope": missions[row["photo"]].scope, "primary_category": missions[row["photo"]].primary_category}
                    for row in batch
                ]
                result = " / ".join(line["mission_description"] for line in lines)
            except HTTPException as e:  # 측정 스크립트라 실패도 기록하고 다음 묶음으로 넘어간다
                lines = [{"photo": row["photo"], "batch": batch_id, "error": f"{e.status_code}: {e.detail}"} for row in batch]
                result = f"에러 {e.status_code}"
            for line in lines:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{n}/{len(todo)}] {batch_id} ({batch[0]['region']}): {result}")


if __name__ == "__main__":
    asyncio.run(main())
