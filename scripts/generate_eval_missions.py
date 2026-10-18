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

프롬프트 비교 실행(--prompt를 주면): 같은 dev 장소×카테고리·묶음으로 이전 프롬프트(v1)와 지금 프롬프트(current)를 따로 생성한다.
--split-file이 꼭 있어야 하고, 실행마다 --out을 다르게 준다. 결과 줄마다 프롬프트·묶음 계획 지문을 남기고,
기존 파일과 지문이 다르면(다른 프롬프트, 다른 표본·묶음, 지문 없는 예전 결과) 이어 쓰지 않고 멈춘다.
실행 조건(모델, 시각, git 커밋, 인자)은 <out>.meta.json에 남는다.
  uv run python -m scripts.generate_eval_missions --manifest /Volumes/T7/aihub/eval_b/manifest.jsonl \\
      --split-file /Volumes/T7/aihub/eval_split.json --split dev --sample-places 80 \\
      --prompt v1 --out /Volumes/T7/aihub/eval_b/prompt_compare/v1_run1.jsonl --dry-run
"""

import argparse
import asyncio
import datetime
import hashlib
import json
import pathlib
import random
import subprocess
import sys
from collections import defaultdict

from fastapi import HTTPException
from google.genai import errors

from app.photomissions import gemini_client, pipeline_generate
from app.photomissions.pipeline_generate import generate_missions
from app.photomissions.prompts import MISSION_SYSTEM_PROMPT
from app.photomissions.schemas import (
    DisplayName,
    MissionPlace,
    PhotoMissionGenerateRequest,
)
from scripts.mission_prompt_v1 import MISSION_SYSTEM_PROMPT_V1

PROMPTS = {"current": MISSION_SYSTEM_PROMPT, "v1": MISSION_SYSTEM_PROMPT_V1}

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


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def prompt_sha(text: str) -> str:
    # 결과 파일에 어떤 프롬프트로 만들었는지 남긴다. 같은 이름이어도 내용이 바뀌면 값이 달라진다
    return _sha(text)


def plan_sha(batches: list[list[dict]], batch_size: int, seed: int) -> str:
    # 어떤 입력을 어떤 묶음·순서로 불렀는지의 지문. 표본·seed·묶음 크기, Gemini에 가는 장소명·카테고리가 바뀌면 달라진다
    batches_in = [[[r["photo"], r["place_name"], r["category"]] for r in b] for b in batches]
    plan = {"batch_size": batch_size, "seed": seed, "batches": batches_in}
    return _sha(json.dumps(plan, ensure_ascii=False, sort_keys=True))


def read_done(out: pathlib.Path, expect: dict | None = None) -> set[str]:
    # 이미 미션이 만들어진 사진. expect(비교 실행)가 있으면 기존 줄이 모두 같은 프롬프트·묶음 계획이어야 이어 쓴다
    done: set[str] = set()
    if not out.exists():
        return done
    for line in out.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if expect is not None:
            got = {key: record.get(key) for key in expect}
            if got != expect:
                raise SystemExit(f"{out}의 기존 결과가 이번 실행과 다름 ({got} != {expect}). --out을 바꿀 것")
        if "error" not in record:
            done.add(record["photo"])
    return done


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False).stdout.strip()


def write_meta(path: pathlib.Path, expect: dict) -> None:
    # 실행할 때마다 조건을 한 줄씩 쌓는다. 이어 쓰기에서 모델이나 코드가 바뀌었는지 나중에 볼 수 있다
    meta = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {**expect, "runs": []}
    if {key: meta.get(key) for key in expect} != expect:
        raise SystemExit(f"{path}가 이번 실행과 다름. --out을 바꿀 것")
    meta["runs"].append({
        "started_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": gemini_client.MODEL_NAME,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "argv": sys.argv[1:],
    })
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def use_prompt(name: str) -> str:
    # 서비스 코드는 고치지 않고, 이 실행에서만 generate가 쓰는 프롬프트를 바꾼다
    # (write_descriptions가 호출될 때 pipeline_generate.MISSION_SYSTEM_PROMPT를 읽는다)
    pipeline_generate.MISSION_SYSTEM_PROMPT = PROMPTS[name]
    return prompt_sha(PROMPTS[name])


def sample_places(rows: list[dict], n: int, seed: int) -> list[dict]:
    # 같은 방문지 사진이 여러 장이면 한 장만 남기고, 지역×카테고리 칸을 돌아가며 n곳을 고른다
    unique: dict[tuple, dict] = {}
    for row in sorted(rows, key=lambda r: r["photo"]):
        unique.setdefault((row["region"], row["category"], row["place_name"]), row)
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for (region, category, _), row in unique.items():
        cells[(region, category)].append(row)
    for key, pool in cells.items():
        random.Random(f"{seed}-{key}").shuffle(pool)
    picked: list[dict] = []
    while len(picked) < n and any(cells.values()):
        for key in sorted(cells):
            if cells[key] and len(picked) < n:
                picked.append(cells[key].pop())
    return picked


def describe_cause(e: HTTPException) -> str:
    # 서비스는 원인을 숨기고 "미션 문구 생성 실패"만 돌려주므로, 측정에서는 원래 에러를 같이 남긴다
    cause = e.__cause__
    if cause is None:
        return ""
    if isinstance(cause, errors.APIError):
        return f"{type(cause).__name__} {cause.code} {cause.status}: {str(cause.message)[:200]}"
    return f"{type(cause).__name__}: {str(cause)[:200]}"


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


async def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True, help="결과 JSONL 경로 (레포 밖에 둘 것)")
    ap.add_argument("--batch-size", type=int, default=5, help="한 번에 묶는 장소 수 (실제 동선 하루 장소 수에 맞춤)")
    ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--max-calls", type=int, default=0, help="이번 실행의 최대 호출 수 (0이면 제한 없음)")
    ap.add_argument("--dry-run", action="store_true", help="호출하지 않고 묶음 계획만 출력")
    ap.add_argument("--prompt", choices=sorted(PROMPTS), help="주면 프롬프트 비교 실행. v1은 비교용으로 고정한 이전 프롬프트")
    ap.add_argument("--split-file", type=pathlib.Path, help="eval_split.json. 비교 실행에서는 꼭 줘야 한다")
    ap.add_argument("--split", choices=["dev"], default="dev", help="프롬프트 비교는 dev만 쓴다")
    ap.add_argument("--sample-places", type=int, default=0, help="방문지(장소×카테고리) n개만 골라 쓴다 (0이면 전부)")
    args = ap.parse_args(argv)
    if args.batch_size <= 0 or args.sample_places < 0 or args.max_calls < 0:
        ap.error("--batch-size는 1 이상, --sample-places·--max-calls는 0 이상이어야 함")
    if args.prompt is None and (args.split_file or args.sample_places):
        ap.error("--split-file·--sample-places는 비교 실행(--prompt)에서만 씀")
    if args.prompt is not None and args.split_file is None:
        ap.error("비교 실행(--prompt)에는 --split-file이 꼭 필요함 (final 사진이 섞이지 않게)")

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    expect = None
    if args.prompt is not None:
        allowed = set(json.loads(args.split_file.read_text(encoding="utf-8"))["photos"][args.split])
        rows = [r for r in rows if r["photo"] in allowed]
        if args.sample_places:
            rows = sample_places(rows, args.sample_places, args.seed)
    batches = make_batches(rows, args.batch_size, args.seed)
    if args.prompt is not None:
        expect = {"prompt": args.prompt, "prompt_sha": use_prompt(args.prompt),
                  "plan_sha": plan_sha(batches, args.batch_size, args.seed)}
    done = read_done(args.out, expect)
    todo = [(i, b) for i, b in enumerate(batches) if not all(row["photo"] in done for row in b)]
    remaining = len(todo)  # --max-calls로 이번에 건너뛰는 묶음은 완료가 아니라 남은 묶음이다
    if args.max_calls:
        todo = todo[: args.max_calls]

    print(f"모델: {gemini_client.MODEL_NAME}" + (f" · 비교 실행 {expect}" if expect else ""))
    print(f"사진 {len(rows)}장 → 묶음 {len(batches)}개 (이미 완료 {len(batches) - remaining}개) → 이번에 호출할 것 {len(todo)}개")
    print(f"예상 비용: 약 {len(todo) * WON_PER_CALL:,}원 (1회 약 {WON_PER_CALL}원 추정)")
    if args.dry_run:
        for i, batch in todo[:3]:
            print(f"  묶음 {i} ({batch[0]['region']}): " + ", ".join(f"{r['place_name']}[{r['category']}]" for r in batch))
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if expect:
        write_meta(args.out.with_name(args.out.name + ".meta.json"), expect)
    with args.out.open("a", encoding="utf-8") as f:
        for n, (i, batch) in enumerate(todo, 1):
            batch_id = f"eval_b_{i:03d}"
            try:
                response = await generate_missions(to_request(batch, batch_id))
                missions = {m.place_id: m for m in response.missions}
                lines = [
                    {"photo": row["photo"], "batch": batch_id, "mission_description": missions[row["photo"]].description,
                     "scope": missions[row["photo"]].scope, "primary_category": missions[row["photo"]].primary_category,
                     **(expect or {})}
                    for row in batch
                ]
                result = " / ".join(line["mission_description"] for line in lines)
            except HTTPException as e:  # 측정 스크립트라 실패도 기록하고 다음 묶음으로 넘어간다
                cause = describe_cause(e)
                lines = [{"photo": row["photo"], "batch": batch_id, "error": f"{e.status_code}: {e.detail}",
                          "cause": cause, **(expect or {})}
                         for row in batch]
                result = f"에러 {e.status_code} ({cause})"
            for line in lines:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{n}/{len(todo)}] {batch_id} ({batch[0]['region']}): {result}")


if __name__ == "__main__":
    asyncio.run(main())
