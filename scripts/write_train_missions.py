"""학습 데이터 1차 파일럿: 사진마다 미션 붙이기.

build_train_pilot.py가 뽑은 사진(manifest.jsonl)에 세 가지 방법으로 미션을 붙인다. 정답은 나중에 사람이 표시한다.
  photo_written    사진을 보고 미션을 쓴다. 카테고리마다 4개, 그중 2개는 사진에서 확인되는 조건을 꼭 넣는다
  condition_swap   조건을 넣은 photo_written 미션에서 조건 하나만 바꿔, 이 사진이 지키지 않은 조건으로 만든다 (같은 사진)
  service_generate 서비스 generate로 장소만 보고 미션을 만든다 (실제 서비스와 같은 프롬프트·모델)
미션을 쓰는 모델은 비교 대상이 아닌 모델이다 (write_photo_missions.WRITER_MODEL, 서비스 generate 모델).

  # 레포 루트에서 실행. 호출 없이 계획만 확인
  uv run python -m scripts.write_train_missions --pilot-dir /Volumes/T7/aihub/train/pilot --aihub /Volumes/T7/aihub --dry-run
  # 실제 호출 (Gemini 비용 발생)
  uv run python -m scripts.write_train_missions --pilot-dir /Volumes/T7/aihub/train/pilot --aihub /Volumes/T7/aihub

  결과는 pilot-dir/missions.jsonl에 미션 1개당 한 줄로 쌓인다. 다시 실행하면 이미 만든 미션은 건너뛴다.
"""

import argparse
import asyncio
import json
import pathlib
import zipfile
from collections import defaultdict

from fastapi import HTTPException
from google.genai import types
from pydantic import BaseModel, Field

from app.photomissions import gemini_client
from app.photomissions.pipeline_generate import generate_missions
from app.photomissions.pipeline_verify import normalize_image
from scripts.build_train_pilot import TRAIN_CATEGORIES
from scripts.generate_eval_missions import make_batches, to_request
from scripts.write_photo_missions import WRITER_MODEL, write_one

CATEGORY_KO = {
    "HISTORY_CULTURE": "역사·문화", "NATURE_HEALING": "자연·힐링", "FOOD": "음식",
    "ACTIVITY": "액티비티", "CONVENIENCE_SHOPPING": "쇼핑·편의",
}
PW_COND, PW_PLAIN = 2, 2  # 카테고리마다 조건을 넣은 미션 2개 + 그냥 쓴 미션 2개
# "앞에서"는 사람 없이 대상만 보여도 성공이라(label_criteria.md) 조건으로 치지 않는다 (2026-10-10 파일럿에서 절반이 "앞에서"였음)
WITH_CONDITION = (
    "이번 미션에는 사진에서 실제로 지켜진 조건(들고, 다 같이, 가까이, 각자) 하나를 반드시 붙인다. "
    "\"앞에서\"는 조건으로 치지 않는다. 사진에서 확인되는 조건이 없으면 usable을 false로 한다. "
    "\"다 같이\"는 \"○○ 앞에서 다 같이 찍기\"처럼 사람들이 사진에 나와야 함을 알 수 있게 쓴다 "
    "(\"다 같이 바다 찍기\"는 모두가 바다를 찍는다는 뜻으로도 읽힌다)."
)

# 조건 바꾸기 2차 (2026-10-11, 파일럿 결과 반영)
# - 1차는 10개 중 8개가 "다 같이"로 쏠렸고, 사진에 이미 여러 명이 있는데 "다 같이"를 붙여 성공이 된 경우가 2개였다
# - 그래서 바꿀 조건을 미리 돌아가며 정해 주고, 사람 수를 먼저 세게 하고, "다 같이 + 가까이"는 코드에서 막는다
# - 2차 검증에서 "다 같이 + 대상 + 찍기"는 "모두가 대상을 찍는다"로도 읽혀 3개 모두 미션 애매였다 → "다 같이"는 쓰지 않는다
SWAP_ORDER = ("가까이", "들고")
SWAP_PROMPT = """\
너는 여행 사진 미션의 조건을 하나만 바꾸는 사람이다. 사진 한 장, 이 사진에 맞게 쓴 미션, 이번에 쓸 조건을 받는다.
이번에 쓸 조건으로 미션을 바꾸거나 더해서, 이 사진으로는 그 조건을 지켰다고 볼 수 없는 미션을 만든다.

순서
1. 사진 속 사람 수를 먼저 센다(people). 얼굴이 안 보여도 사람이면 센다.
2. 이번에 쓸 조건을 이 사진이 분명히 지키지 않는지 확인한다. 지키고 있거나 애매하면 usable을 false로 한다.
   - "가까이": 대상이 멀리 작게 보이거나 사진의 주인공이 아닐 때만 쓴다
   - "들고": 손에 들 수 있는 대상(음식, 음료, 기념품, 소품, 티켓)이고, 사진에서 아무도 그것을 들고 있지 않을 때만 쓴다
3. 대상과 나머지 문구는 그대로 두고 조건만 바꾸거나 더한다. 끝은 "~찍기"로 맺는다.

규칙
- 결과는 실제 여행 앱이 낼 법한 자연스러운 한국어 미션이어야 한다. 억지로 만들지 않는다.
- before에는 원래 조건(없었으면 "없음"), after에는 이번에 쓴 조건을 쓴다.

예시
- 원래 "월정교 앞에서 찍기", 조건 "가까이", 다리가 멀리 작게 보임 → "월정교 가까이 찍기"
- 원래 "주문한 차 찍기", 조건 "들고", 찻잔이 테이블에 놓여 있음 → "주문한 차 들고 찍기"
- 원래 "석갈비 찍기", 조건 "들고" → usable false (불판 위 음식은 들 수 없음)
"""


class SwappedMission(BaseModel):
    people: int = Field(default=0)  # 사진 속 사람 수
    usable: bool
    mission: str = Field(default="")
    before: str = Field(default="")
    after: str = Field(default="")


def conflicts(mission: str) -> bool:
    # 1차 파일럿에서 "다 같이 ~ 가까이" 미션은 판정이 어렵다고 나왔다 (미션 애매)
    return "다 같이" in mission and "가까이" in mission


def pw_mode(records: list[dict]) -> str | None:
    # 한 카테고리의 photo_written 기록을 보고 다음 사진에 쓸 방식을 정한다. 다 채웠으면 None
    usable = [r for r in records if r["source"] == "photo_written" and r.get("usable")]
    if sum(r["with_condition"] for r in usable) < PW_COND:
        return "with_condition"
    if sum(not r["with_condition"] for r in usable) < PW_PLAIN:
        return "plain"
    return None


def swap_targets(records: list[dict], success_ids: set[str] | None = None) -> list[dict]:
    # 아직 조건을 바꾸지 않은 photo_written 미션
    # success_ids(사람이 성공으로 표시한 미션)를 주면 그중에서만 고른다. 원래 미션이 틀린 사진에서 바꾸면 의미가 없다
    # 주지 않으면 1차 방식대로 조건을 넣어 쓴 미션에서 고른다
    swapped = {r["parent_mission_id"] for r in records if r["source"] == "condition_swap"}
    return [r for r in records if r["source"] == "photo_written" and r.get("usable") and r["mission_id"] not in swapped
            and (r["mission_id"] in success_ids if success_ids is not None else r["with_condition"])]


async def swap_one(mission: str, image: bytes, condition: str) -> SwappedMission:
    config = types.GenerateContentConfig(
        system_instruction=SWAP_PROMPT, response_mime_type="application/json", response_schema=SwappedMission,
    )
    # 2차에서 "들고"가 건물·길 같은 원본에 배정돼 모두 못 바꿈이 됐다. 우선 조건이 안 맞으면 다른 조건을 쓰게 한다
    other = next(c for c in SWAP_ORDER if c != condition)
    contents = [f"원래 미션: {mission}\n이번에 쓸 조건: {condition} (이 사진·대상에 쓸 수 없으면 {other})", types.Part.from_bytes(data=normalize_image(image), mime_type="image/jpeg")]
    response = await gemini_client._get_client().aio.models.generate_content(
        model=WRITER_MODEL, contents=contents, config=config,
    )
    if not isinstance(response.parsed, SwappedMission):
        raise TypeError("응답이 스키마와 맞지 않음")
    return response.parsed


def _base(row: dict) -> dict:
    return {"photo": row["photo"], "travel_id": row["travel_id"], "place": row["place_name"], "region": row["region"],
            "category": row["category"]}


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot-dir", type=pathlib.Path, required=True)
    ap.add_argument("--aihub", type=pathlib.Path, required=True)
    ap.add_argument("--labels", type=pathlib.Path,
                    help="파일럿 표시 결과. 주면 사람이 성공으로 표시한 미션에서만 조건을 바꾼다 (2차)")
    ap.add_argument("--max-calls", type=int, default=0, help="이번 실행의 최대 호출 수 (0이면 제한 없음)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(line) for line in (args.pilot_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line]
    out = args.pilot_dir / "missions.jsonl"
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line] if out.exists() else []
    records = [r for r in records if "error" not in r]  # 에러 난 것은 다시 호출한다
    done_ids = {r["mission_id"] for r in records}
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r)

    gen_rows = [r for r in rows if r["role"] == "service_generate" and f"{r['photo']}#gen" not in done_ids]
    gen_batches = make_batches(gen_rows, batch_size=5, seed=20261010)
    pw_left = {c: pw_mode(by_cat[c]) for c in TRAIN_CATEGORIES}
    success_ids = None
    if args.labels:
        items = json.loads(args.labels.read_text(encoding="utf-8"))["items"]
        success_ids = {x["mission_id"] for x in items if x.get("final_grade") == "success"}
    print(f"사진 {len(rows)}장 · 지금까지 미션 {len(records)}개")
    print(f"사진 보고 쓰기: 아직 덜 채운 카테고리 {[c for c, m in pw_left.items() if m]}")
    print(f"조건 바꾸기: 지금 바꿀 수 있는 미션 {len(swap_targets(records, success_ids))}개"
          + (" (사람이 성공으로 표시한 미션에서만)" if success_ids is not None else " (사진 보고 쓰기가 끝나면 늘어남)"))
    print(f"서비스 미션: 남은 사진 {len(gen_rows)}장 → generate {len(gen_batches)}회")
    if args.dry_run:
        return

    calls = 0
    zips: dict[str, zipfile.ZipFile] = {}

    def image_of(row: dict) -> bytes:
        return zips.setdefault(row["zip"], zipfile.ZipFile(args.aihub / row["zip"])).read(row["member"])

    def budget_left() -> bool:
        return not args.max_calls or calls < args.max_calls

    with out.open("a", encoding="utf-8") as f:
        def save(record: dict) -> None:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            if "error" not in record:
                records.append(record)
                by_cat[record["category"]].append(record)

        # 1) 사진 보고 쓰기: 카테고리마다 예비 사진까지 순서대로, 4개를 채우면 멈춘다
        for category in TRAIN_CATEGORIES:
            for row in (r for r in rows if r["category"] == category and r["role"] == "photo_written"):
                mode = pw_mode(by_cat[category])
                if mode is None or not budget_left():
                    break
                if f"{row['photo']}#pw" in done_ids:
                    continue
                record = {**_base(row), "mission_id": f"{row['photo']}#pw", "source": "photo_written",
                          "parent_mission_id": None, "with_condition": mode == "with_condition", "model": WRITER_MODEL}
                try:
                    written = await write_one({**row, "category_ko": CATEGORY_KO[category]}, image_of(row),
                                              WITH_CONDITION if mode == "with_condition" else "")
                    record.update(usable=written.usable, mission=written.mission.strip())
                except Exception as e:  # noqa: BLE001 — 데이터 만드는 스크립트라 실패도 기록하고 넘어간다
                    record["error"] = f"{type(e).__name__}: {e}"
                calls += 1
                save(record)
                print(f"[쓰기] {CATEGORY_KO[category]} {row['place_name']} ({mode}): "
                      f"{record.get('error') or (record['mission'] if record['usable'] else '(못 씀)')}")

        # 2) 조건 바꾸기: 원래 미션마다 1개. 바꿀 조건은 SWAP_ORDER를 돌아가며 정한다
        row_of = {r["photo"]: r for r in rows}
        for i, parent in enumerate(swap_targets(records, success_ids)):
            if not budget_left():
                break
            row = row_of[parent["photo"]]
            condition = SWAP_ORDER[i % len(SWAP_ORDER)]
            record = {**_base(row), "mission_id": f"{row['photo']}#swap", "source": "condition_swap",
                      "parent_mission_id": parent["mission_id"], "with_condition": True, "model": WRITER_MODEL,
                      "batch": "pilot2" if success_ids is not None else "pilot", "target_condition": condition}
            try:
                swapped = await swap_one(parent["mission"], image_of(row), condition)
                mission = swapped.mission.strip()
                usable = swapped.usable and not conflicts(mission)
                record.update(usable=usable, mission=mission, before=swapped.before, after=swapped.after,
                              people=swapped.people)
                if swapped.usable and not usable:
                    record["rejected"] = "다 같이 + 가까이"
            except Exception as e:  # noqa: BLE001
                record["error"] = f"{type(e).__name__}: {e}"
            calls += 1
            save(record)
            shown = record.get("error") or (record["mission"] if record.get("usable") else f"(못 바꿈) {record.get('rejected', '')}")
            print(f"[바꾸기:{condition}] {parent['mission']} → {shown} · 사람 {record.get('people', '?')}명")

        # 3) 서비스 미션: 같은 지역 장소를 묶어 실제 동선처럼 generate를 부른다
        for i, batch in enumerate(gen_batches):
            if not budget_left():
                break
            try:
                response = await generate_missions(to_request(batch, f"train_pilot_{i:03d}"))
                missions = {m.place_id: m for m in response.missions}
                results = [{"mission": missions[r["photo"]].description, "scope": missions[r["photo"]].scope,
                            "usable": True} for r in batch]
            except HTTPException as e:
                results = [{"error": f"{e.status_code}: {e.detail}"} for _ in batch]
            calls += 1
            for row, result in zip(batch, results, strict=True):
                save({**_base(row), "mission_id": f"{row['photo']}#gen", "source": "service_generate",
                      "parent_mission_id": None, "with_condition": False, "model": gemini_client.MODEL_NAME, **result})
            print(f"[서비스] {batch[0]['region']}: " + " / ".join(r.get("mission") or r.get("error", "") for r in results))

    print(f"이번 호출 {calls}회 · 미션 {len(records)}개")


if __name__ == "__main__":
    asyncio.run(main())
