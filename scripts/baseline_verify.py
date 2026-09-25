"""verify 사진 판정 baseline 측정.

AI Hub 평가셋 사진을 score_photo와 같은 조건(프롬프트·리사이즈·응답 스키마)으로 판정하고,
호출마다 점수·토큰·비용·시간을 JSONL로 한 줄씩 남긴다. 결과로 임계값(SUCCESS/RETRY/LANDMARK,
CLEAR_MISMATCH_KM)을 정하고 2단계 baseline 지표(latency·구조화 출력 1차 성공률)를 채운다.

  # 레포 루트에서 실행 (app 패키지를 불러오려면 -m으로 실행해야 한다)
  # 호출 없이 계획만 확인
  uv run python -m scripts.baseline_verify --mapping <tsv> --photos <zip|dir> --per-place 5 --out <jsonl> --dry-run
  # 실제 호출 (최대 호출 수로 비용 상한)
  uv run python -m scripts.baseline_verify --mapping <tsv> --photos <zip|dir> --per-place 5 --max-calls 10 --out <jsonl>

  같은 --out으로 다시 실행하면 이미 성공한 건은 건너뛰고 이어서 호출한다(에러 난 건은 다시 호출).
  모델·설정을 바꿔 비교할 때는 --out을 따로 써야 결과가 섞이지 않는다.

  # 다른 모델로, 기존 결과 파일에서 성공한 건과 같은 사진·미션만 호출 (모델 비교용)
  uv run python -m scripts.baseline_verify ... --model gemini-3.1-flash-lite --same-as <pro 결과 jsonl> --out <새 jsonl>
"""

import argparse
import asyncio
import functools
import json
import os
import pathlib
import random
import statistics
import time
import zipfile

from google.genai import types

from app.photomissions import gemini_client
from app.photomissions.pipeline_verify import _haversine_km, normalize_image
from app.photomissions.prompts import VERIFY_SYSTEM_PROMPT
from app.photomissions.schemas import Coordinates, VlmResult

LANDMARKS = {
    "대릉원": Coordinates(latitude=35.8384, longitude=129.2118),
    "첨성대": Coordinates(latitude=35.8347, longitude=129.2192),
    "동궁과월지": Coordinates(latitude=35.8349, longitude=129.2266),
    "월정교": Coordinates(latitude=35.8293, longitude=129.2178),
}
# AI Hub 사진은 미션 수행용이 아닌 일반 여행 사진이라, 구체적인 문구 대신 "장소가 보이는지"를 묻는 고정 문구를 쓴다
MISSIONS = {
    "대릉원": "대릉원이 잘 보이게 사진 찍기",
    "첨성대": "첨성대가 잘 보이게 사진 찍기",
    "동궁과월지": "동궁과월지가 잘 보이게 사진 찍기",
    "월정교": "월정교가 잘 보이게 사진 찍기",
}

# 유료 단가(USD / 100만 토큰, 입력 / 출력(생각 포함)). ai.google.dev/gemini-api/docs/pricing 기준
# pro-preview는 1회 실측 청구액(27원)과 일치 확인함
PRICES = {
    "gemini-3.1-pro-preview": (2.0, 12.0),
    "gemini-3.1-flash-lite": (0.25, 1.50),
}


def load_mapping(path: pathlib.Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, travel_id, area_id, place, lat, lon = line.split("\t")
        photo_coords = Coordinates(latitude=float(lat), longitude=float(lon)) if lat and lon else None
        rows.append({"photo": name, "travel_id": travel_id, "visit_area_id": area_id,
                     "place": place, "photo_coords": photo_coords})
    return rows


@functools.cache
def _open_zip(path: pathlib.Path) -> zipfile.ZipFile:
    # 수만 개 파일 목록을 사진마다 다시 읽지 않도록, zip은 한 번 열어 계속 쓴다
    return zipfile.ZipFile(path)


def index_photos(sources: list[pathlib.Path]) -> dict[str, tuple[pathlib.Path, str | None]]:
    # 파일명 → (zip 경로, zip 안 경로) 또는 (파일 경로, None). zip은 풀지 않고 안에서 바로 읽는다
    index: dict[str, tuple[pathlib.Path, str | None]] = {}
    for src in sources:
        if src.suffix.lower() == ".zip":
            for member in _open_zip(src).namelist():
                if member.lower().endswith(".jpg"):
                    index[os.path.basename(member)] = (src, member)
        else:
            for root, _, files in os.walk(src):
                for f in files:
                    if f.lower().endswith(".jpg"):
                        index[f] = (pathlib.Path(root) / f, None)
    return index


def read_photo(location: tuple[pathlib.Path, str | None]) -> bytes:
    path, member = location
    if member is None:
        return path.read_bytes()
    return _open_zip(path).read(member)


def build_cases(rows: list[dict], available: set[str], per_place: int, seed: int) -> list[dict]:
    # 장소마다 사진 per_place장을 뽑아, 같은 장소 미션(positive)과 다른 장소 미션(negative)을 한 번씩 만든다
    # 장소별로 순서를 한 번 섞어두고 앞에서부터 가져가서, per_place를 늘려도 앞서 뽑힌 사진은 그대로 유지된다
    cases = []
    for place in LANDMARKS:
        pool = sorted((r for r in rows if r["place"] == place and r["photo"] in available), key=lambda r: r["photo"])
        random.Random(f"{seed}-{place}").shuffle(pool)
        for r in pool[:per_place]:
            other = random.Random(f"{seed}-{r['photo']}").choice([p for p in LANDMARKS if p != place])
            for kind, target in (("positive", place), ("negative", other)):
                cases.append({**r, "kind": kind, "target": target})
    return cases


def case_key(case: dict) -> str:
    return f"{case['photo']}|{case['kind']}|{case['target']}"


async def judge(case: dict, image_bytes: bytes) -> dict:
    # score_photo와 같은 입력으로 1번만 호출한다. 재시도하지 않아야 1차 성공률과 1회 비용을 그대로 잴 수 있다
    t0 = time.perf_counter()
    processed = normalize_image(image_bytes)
    t1 = time.perf_counter()
    contents = [
        f"목표 장소: {case['target']}\n미션 내용: {MISSIONS[case['target']]}",
        types.Part.from_bytes(data=processed, mime_type="image/jpeg"),
    ]
    config = types.GenerateContentConfig(
        system_instruction=VERIFY_SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=VlmResult,
    )
    record = {"normalize_sec": round(t1 - t0, 3), "resized_bytes": len(processed), "original_bytes": len(image_bytes)}
    try:
        response = await gemini_client._get_client().aio.models.generate_content(
            model=gemini_client.MODEL_NAME, contents=contents, config=config,
        )
    except Exception as e:  # noqa: BLE001 — 측정 스크립트라 실패도 결과로 기록한다
        record.update(model_sec=round(time.perf_counter() - t1, 3), error=f"{type(e).__name__}: {e}")
        return record
    record["model_sec"] = round(time.perf_counter() - t1, 3)

    u = response.usage_metadata
    input_per_m, output_per_m = PRICES[gemini_client.MODEL_NAME]
    image_tokens = sum(d.token_count or 0 for d in (u.prompt_tokens_details or []) if "IMAGE" in str(d.modality))
    prompt, output, thoughts = u.prompt_token_count or 0, u.candidates_token_count or 0, u.thoughts_token_count or 0
    record.update(
        prompt_tokens=prompt, image_tokens=image_tokens, output_tokens=output, thought_tokens=thoughts,
        cost_usd=round(prompt * input_per_m / 1e6 + (output + thoughts) * output_per_m / 1e6, 6),
        parsed=isinstance(response.parsed, VlmResult),
    )
    if isinstance(response.parsed, VlmResult):
        vlm = response.parsed
        record.update(
            match_score=vlm.match_score,
            landmark_confidence=vlm.landmark_confidence,
            detected_labels=[label.model_dump() for label in vlm.detected_labels],
            retry_hint=vlm.retry_hint,
        )
    return record


def summarize(out: pathlib.Path) -> None:
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    ok = [r for r in records if "error" not in r]
    print(f"\n[요약] 기록 {len(records)}건 / 에러 {len(records) - len(ok)}건")
    if not ok:
        return
    parsed = [r for r in ok if r.get("parsed")]
    print(f"  구조화 출력 1차 성공률: {len(parsed)}/{len(ok)}")
    secs = sorted(r["model_sec"] for r in ok)
    p95 = secs[min(len(secs) - 1, int(len(secs) * 0.95))]
    print(f"  모델 호출 시간: 평균 {statistics.mean(secs):.2f}초 / p95 {p95:.2f}초")
    print(f"  총 비용: ${sum(r['cost_usd'] for r in ok):.4f}")
    for kind in ("positive", "negative"):
        scores = [r["match_score"] for r in parsed if r["kind"] == kind]
        if scores:
            print(f"  {kind} match_score: 평균 {statistics.mean(scores):.1f} / 최소 {min(scores)} / 최대 {max(scores)} ({len(scores)}건)")


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mapping", type=pathlib.Path, required=True)
    ap.add_argument("--photos", type=pathlib.Path, nargs="+", required=True, help="사진이 든 zip 또는 폴더")
    ap.add_argument("--per-place", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--out", type=pathlib.Path, required=True, help="결과 JSONL 경로 (레포 밖에 둘 것)")
    ap.add_argument("--max-calls", type=int, default=0, help="이번 실행의 최대 호출 수 (0이면 제한 없음)")
    ap.add_argument("--dry-run", action="store_true", help="호출하지 않고 계획만 출력")
    ap.add_argument("--model", default=gemini_client.MODEL_NAME, choices=sorted(PRICES), help="판정 모델")
    ap.add_argument("--same-as", type=pathlib.Path, help="이 결과 파일에서 성공한 건과 같은 사진·미션만 호출")
    args = ap.parse_args()
    if args.same_as and args.same_as.resolve() == args.out.resolve():
        ap.error("--same-as와 --out은 다른 파일이어야 함")
    # judge는 호출할 때마다 gemini_client.MODEL_NAME을 읽으므로 여기서 바꾸면 이 실행 전체가 이 모델로 판정한다
    gemini_client.MODEL_NAME = args.model

    rows = load_mapping(args.mapping)
    index = index_photos(args.photos)
    cases = build_cases(rows, set(index), args.per_place, args.seed)
    if args.same_as:
        base = [json.loads(line) for line in args.same_as.read_text(encoding="utf-8").splitlines() if line.strip()]
        keep = {case_key(r) for r in base if "error" not in r}
        cases = [c for c in cases if case_key(c) in keep]
    done = set()
    if args.out.exists():
        records = [json.loads(line) for line in args.out.read_text(encoding="utf-8").splitlines() if line.strip()]
        done = {case_key(r) for r in records if "error" not in r}
    todo = [c for c in cases if case_key(c) not in done]
    if args.max_calls:
        todo = todo[: args.max_calls]

    print(f"모델: {gemini_client.MODEL_NAME}")
    print(f"매핑 {len(rows)}장 중 찾은 사진 {sum(r['photo'] in index for r in rows)}장")
    for place in LANDMARKS:
        n = sum(1 for c in cases if c["place"] == place) // 2
        print(f"  {place}: 표본 {n}장")
    print(f"케이스 {len(cases)}건 (이미 기록 {len(done & {case_key(c) for c in cases})}건) → 이번에 호출할 것 {len(todo)}건")
    # pro-preview 1회 27원 실측을 단가 비율로 환산한 대략값. 토큰 수가 모델마다 달라 실제와 차이날 수 있다
    per_call_won = 27 * PRICES[gemini_client.MODEL_NAME][0] / PRICES["gemini-3.1-pro-preview"][0]
    print(f"예상 비용: 약 {len(todo) * per_call_won:,.0f}원 (pro-preview 1회 27원 실측을 단가 비율로 환산)")
    if args.dry_run:
        for c in todo[:6]:
            print(f"  - {c['photo']} ({c['place']} 사진) → 목표 {c['target']} [{c['kind']}]")
        return

    with args.out.open("a", encoding="utf-8") as f:
        for i, case in enumerate(todo, 1):
            try:
                record = await judge(case, read_photo(index[case["photo"]]))
            except Exception as e:  # noqa: BLE001 — 사진 하나가 깨져도 전체 실행을 멈추지 않는다
                record = {"model_sec": 0.0, "error": f"사진 처리 실패 {type(e).__name__}: {e}"}
            distance = _haversine_km(case["photo_coords"], LANDMARKS[case["target"]]) if case["photo_coords"] else None
            line = {
                "photo": case["photo"], "place": case["place"], "kind": case["kind"], "target": case["target"],
                "distance_km": round(distance, 3) if distance is not None else None, "model": gemini_client.MODEL_NAME,
                **record,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            f.flush()
            score = line.get("match_score", line.get("error", "-"))
            print(f"[{i}/{len(todo)}] {case['photo']} → {case['target']} [{case['kind']}]: {score} ({line['model_sec']}초)")
    summarize(args.out)


if __name__ == "__main__":
    asyncio.run(main())
