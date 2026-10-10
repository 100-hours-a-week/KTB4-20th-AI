"""사진을 보고 그 사진에 맞는 미션 문구 쓰기 (평가셋 B 보강, 학습 데이터 성공 쪽 보강용).

AI Hub 사진은 미션을 받고 찍은 사진이 아니라 generate 미션으로 판정하면 정답이 대부분 실패가 된다(평가셋 B 74%).
맞게 찍은 사진을 막지 않는지(FN)를 재려면 성공 정답이 필요해, 사진에 맞는 미션을 모델이 쓰고 사람은 등급만 표시한다.
미션을 쓰는 모델은 비교 대상(Pro · Flash Lite · 자체 모델)이 아닌 모델을 쓴다. 비교 대상이 쓴 문제로 그 모델을 재면 점수가 부풀려진다.

  # 레포 루트에서 실행. 평가셋 B에서 "실패"로 표시한 사진 중 카테고리마다 50장을 골라 미션을 쓴다
  uv run python -m scripts.write_photo_missions --aihub /Volumes/T7/aihub \\
      --out /Volumes/T7/aihub/eval_b/augment/missions.jsonl --dry-run
  uv run python -m scripts.write_photo_missions --aihub /Volumes/T7/aihub \\
      --out /Volumes/T7/aihub/eval_b/augment/missions.jsonl --max-calls 3

  같은 --out으로 다시 실행하면 이미 쓴 사진은 건너뛴다.
"""

import argparse
import asyncio
import json
import pathlib
import random
import zipfile

from google.genai import types
from pydantic import BaseModel, Field

from app.photomissions import gemini_client
from app.photomissions.pipeline_verify import normalize_image

# 비교 대상이 아닌 모델. 사용할 수 있는 이름은 client.models.list()로 확인함 (2026-10-06)
WRITER_MODEL = "gemini-3.5-flash"
CATEGORIES = ("자연·힐링", "음식", "액티비티", "쇼핑·편의")
REGIONS = ("경주", "부산", "서울", "전주", "제주")

SYSTEM_PROMPT = """\
너는 여행 사진 미션을 만드는 사람이다. 사진 한 장과 그 사진을 찍은 장소 정보를 보고,
"이 사진을 찍은 사람이 받았을 법한 미션"을 한 문장으로 쓴다.

규칙
- 20자 안팎, 친구에게 말하듯 짧게 쓴다. 끝은 "~찍기"로 맺는다.
- 대상은 "그 장소에 가면 누구나 볼 수 있는 것" 중에서 사진에 보이는 것 하나로 한다.
  (예: 고분, 소나무 숲길, 바다 위 바위, 석상, 놀이기구, 주문한 음식, 가게 진열대)
  미션을 만드는 사람은 사진을 보지 못하고 장소만 안다고 생각하고, 그 사람이 낼 법한 대상을 고른다.
- 사람의 소지품(유모차, 가방, 휴대폰), 우연히 찍힌 물건, 쓰레기처럼 장소와 상관없는 것은 대상으로 쓰지 않는다.
- 안내판, 표지판, 간판, 현판처럼 글씨가 주인 것은 대상으로 쓰지 않는다.
  사진에 그 장소의 대상이 글씨 말고는 보이지 않으면 usable을 false로 한다.
- 조건은 사진에 실제로 맞을 때만 붙인다. 사람이 보이지 않는 사진이면 "○○ 찍기"나 "○○ 가까이 찍기"로 쓰고,
  "앞에서"를 습관적으로 붙이지 않는다.
- "신기한", "예쁜", "멋진" 같은 꾸미는 말을 쓰지 않는다.
- 조건은 다음 중에서만 붙인다: 들고, 앞에서, 다 같이, 가까이, 각자. 그 밖의 동작(손 대고, 점프하며, 먹으며, 자르며)은 쓰지 않는다.
- 장소 이름보다 대상 이름을 쓴다. (X) 경주월드 찍기 (O) 놀이기구 앞에서 찍기

예시
- 대릉원 숲길 사진 → (O) 소나무 숲길 찍기 / (X) 소나무 숲길에서 유모차와 함께 찍기
- 문무대왕릉 바닷가 사진 → (O) 바다 위 바위 찍기 / (X) 신기한 생선 뼈 옆에 손 대고 찍기
- 교촌마을 조형물 사진 → (O) 말뚝박기 석상 앞에서 찍기
- 사진이 너무 어둡거나 흔들려서, 또는 셀카·영수증처럼 장소와 무관해서 맞는 미션을 쓸 수 없으면 usable을 false로 한다.
"""


class WrittenMission(BaseModel):
    usable: bool  # 맞는 미션을 쓸 수 있는 사진인가
    mission: str = Field(default="")  # usable이 false면 빈 문자열


def select_photos(aihub: pathlib.Path, per_category: int, per_region: int, seed: int) -> list[dict]:
    # 평가셋 B에서 원래 미션으로 "실패"였던 사진 중 카테고리마다 per_category장, 지역마다 per_region장을 목표로 고른다
    eval_b = aihub / "eval_b"
    labels = json.loads((eval_b / "eval_b_labels.json").read_text(encoding="utf-8"))["labels"]
    manifest = {
        row["photo"]: row
        for row in (json.loads(line) for line in (eval_b / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line)
    }
    chosen = []
    for category in CATEGORIES:
        fails = sorted((x for x in labels if x["category"] == category and x["label"] == "fail"), key=lambda x: x["photo"])
        random.Random(f"{seed}-{category}").shuffle(fails)
        picked = []
        for region in REGIONS:
            picked += [x for x in fails if x["region"] == region][:per_region]
        picked += [x for x in fails if x not in picked][: per_category - len(picked)]
        chosen += [{**manifest[x["photo"]], "category_ko": category} for x in picked]
    return chosen


async def write_one(row: dict, image: bytes) -> WrittenMission:
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT, response_mime_type="application/json", response_schema=WrittenMission,
    )
    contents = [
        f"장소: {row['place_name']} ({row['region']}, {row['category_ko']})",
        types.Part.from_bytes(data=normalize_image(image), mime_type="image/jpeg"),
    ]
    response = await gemini_client._get_client().aio.models.generate_content(
        model=WRITER_MODEL, contents=contents, config=config,
    )
    if not isinstance(response.parsed, WrittenMission):
        raise TypeError("응답이 스키마와 맞지 않음")
    return response.parsed


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aihub", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True, help="결과 JSONL (레포 밖에 둘 것)")
    ap.add_argument("--per-category", type=int, default=50)
    ap.add_argument("--per-region", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--max-calls", type=int, default=0, help="이번 실행의 최대 호출 수 (0이면 제한 없음)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = select_photos(args.aihub, args.per_category, args.per_region, args.seed)
    done = set()
    if args.out.exists():
        done = {json.loads(line)["photo"] for line in args.out.read_text(encoding="utf-8").splitlines()
                if line and "error" not in json.loads(line)}
    todo = [r for r in rows if r["photo"] not in done]
    if args.max_calls:
        todo = todo[: args.max_calls]
    print(f"모델: {WRITER_MODEL} · 사진 {len(rows)}장 (이미 작성 {len(rows) - len([r for r in rows if r['photo'] not in done])}장) → 이번에 호출할 것 {len(todo)}장")
    if args.dry_run:
        for r in todo[:5]:
            print(f"  - {r['photo']} {r['region']} {r['category_ko']} {r['place_name']}")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    zips: dict[str, zipfile.ZipFile] = {}
    with args.out.open("a", encoding="utf-8") as f:
        for i, row in enumerate(todo, 1):
            zf = zips.setdefault(row["zip"], zipfile.ZipFile(args.aihub / row["zip"]))
            record = {"photo": row["photo"], "place": row["place_name"], "region": row["region"],
                      "category": row["category_ko"], "model": WRITER_MODEL}
            try:
                written = await write_one(row, zf.read(row["member"]))
                record.update(usable=written.usable, mission=written.mission.strip())
                shown = written.mission if written.usable else "(맞는 미션을 쓸 수 없음)"
            except Exception as e:  # noqa: BLE001 — 측정용 스크립트라 실패도 기록하고 다음으로 넘어간다
                record["error"] = f"{type(e).__name__}: {e}"
                shown = record["error"]
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(todo)}] {row['place_name']}: {shown}")


if __name__ == "__main__":
    asyncio.run(main())
