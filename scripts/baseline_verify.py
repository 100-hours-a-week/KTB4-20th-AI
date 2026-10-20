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

  # 로컬 모델(Ollama 등 OpenAI 호환 서버)로, 닮은 장소(월정교↔동궁과월지) 케이스만 호출
  uv run python -m scripts.baseline_verify ... --model qwen2.5vl:7b --lookalike-only --out <새 jsonl>
"""

import argparse
import asyncio
import base64
import collections
import functools
import json
import os
import pathlib
import random
import statistics
import time
import zipfile

import httpx
from google.genai import types
from pydantic import ValidationError

from app.core.config import settings
from app.photomissions import gemini_client
from app.photomissions.pipeline_verify import _haversine_km, normalize_image
from app.photomissions.prompts import VERIFY_SYSTEM_PROMPT
from app.photomissions.schemas import Coordinates, VlmResult
from scripts.place_probe import (
    IDENTIFY_SYSTEM_PROMPT,
    IDENTIFY_USER_TEXT,
    QUESTION_OBS_SYSTEM_PROMPT,
    IdentifyAnswer,
    QuestionAnswerObs,
    identify_fields,
    option_probs,
    prob_field,
    question_obs_fields,
)
from scripts.question_format import (
    QUESTION_SYSTEM_PROMPT,
    QUESTION_SYSTEM_PROMPT_V2,
    QuestionAnswer,
    rule_grade,
)
from scripts.split_eval import load_split, split_error

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
# 자체 서빙 후보. OpenAI 호환 API(/chat/completions)로 부르므로 Ollama(맥 사전 평가)와 vLLM(RunPod) 모두 같은 코드로 잰다
# qwen3-vl 기본 태그(8b)는 생각(thinking) 모델이라 사진 1건에 108초가 걸리고(생각 끄는 옵션도 안 먹힘) 그 1건은 답이 비었다.
# 바로 답하는 instruct 태그를 쓴다
DEFAULT_LOCAL_URL = "http://localhost:11434/v1"
# 겉모습이 비슷해 Flash Lite가 자주 헷갈린 장소 쌍. 로컬 모델 사전 평가는 이 쌍의 오답 미션만 먼저 돌린다
LOOKALIKE = {"월정교", "동궁과월지"}


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


def build_cases(
    rows: list[dict], available: set[str], per_place: int, seed: int, all_negatives: bool = False
) -> list[dict]:
    # 장소마다 사진 per_place장을 뽑아, 같은 장소 미션(positive)과 다른 장소 미션(negative)을 한 번씩 만든다
    # 장소별로 순서를 한 번 섞어두고 앞에서부터 가져가서, per_place를 늘려도 앞서 뽑힌 사진은 그대로 유지된다
    # all_negatives면 나머지 장소 미션을 모두 만든다. 정답이 "실패"로 정해져 있어 사람 표시 없이 평가 건수를 늘릴 수 있다
    cases = []
    for place in LANDMARKS:
        pool = sorted((r for r in rows if r["place"] == place and r["photo"] in available), key=lambda r: r["photo"])
        random.Random(f"{seed}-{place}").shuffle(pool)
        for r in pool[:per_place]:
            others = [p for p in LANDMARKS if p != place]
            if not all_negatives:
                others = [random.Random(f"{seed}-{r['photo']}").choice(others)]
            cases.append({**r, "kind": "positive", "target": place})
            for other in others:
                cases.append({**r, "kind": "negative", "target": other})
    return cases


def is_openai_compat(model: str) -> bool:
    # 가격표에 있는 이름은 Gemini, 나머지(로컬 Ollama, RunPod vLLM 등)는 OpenAI 호환 서버로 부른다
    return model not in PRICES


def auth_headers(api_key: str) -> dict[str, str]:
    # RunPod처럼 키가 필요한 서버만 헤더를 붙인다. 로컬 Ollama는 키 없이 부른다
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def case_key(case: dict) -> str:
    return f"{case['photo']}|{case['kind']}|{case['target']}"


def is_lookalike(case: dict) -> bool:
    # 월정교 사진에 동궁과월지 미션(또는 반대)을 준 오답 케이스. 여기서 "맞다"고 하면 닮은 장소를 못 가린 것
    return case["kind"] == "negative" and {case["place"], case["target"]} == LOOKALIKE


def lookalike_cases(cases: list[dict], with_positives: bool = False) -> list[dict]:
    # 닮은 장소 오답만. with_positives면 그 사진들에 맞는 장소 미션(positive)도 함께 남긴다
    photos = {c["photo"] for c in cases if is_lookalike(c)}
    return [c for c in cases if is_lookalike(c) or (with_positives and c["kind"] == "positive" and c["photo"] in photos)]


def user_text(case: dict, fmt: str = "score") -> str:
    # score_photo와 같은 형식. 평가셋 B처럼 사진마다 미션이 다르면 case["mission"]을 쓴다
    if fmt == "identify":  # 장소 알아보기 실험은 목표 장소를 알려 주지 않는다
        return IDENTIFY_USER_TEXT
    mission = case.get("mission") or MISSIONS[case["target"]]
    return f"목표 장소: {case['target']}\n미션 내용: {mission}"


def vlm_fields(vlm: VlmResult) -> dict:
    return {
        "match_score": vlm.match_score,
        "landmark_confidence": vlm.landmark_confidence,
        "detected_labels": [label.model_dump() for label in vlm.detected_labels],
        "retry_hint": vlm.retry_hint,
    }


def question_fields(answer: QuestionAnswer) -> dict:
    # 객관식 답 원본을 남겨야 나중에 등급 규칙을 바꿔도 다시 호출하지 않고 다시 셀 수 있다. grade는 기록 시점 규칙으로 정한 참고값
    return {"answer": answer.model_dump(), "grade": rule_grade(answer)}


# 판정 형식별 (지시문, 응답 스키마, 기록할 칸). score = 지금 서비스(점수), question = 개선 ② 객관식(학습 데이터와 같은 지시문)
# question_v2 = 장소 질문만 바꾼 객관식 실험. 답 형식·등급 규칙은 question과 같다
# question_obs·identify = 장소 구별 실험 (scripts/place_probe.py)
FORMATS = {
    "score": (VERIFY_SYSTEM_PROMPT, VlmResult, vlm_fields),
    "question": (QUESTION_SYSTEM_PROMPT, QuestionAnswer, question_fields),
    "question_v2": (QUESTION_SYSTEM_PROMPT_V2, QuestionAnswer, question_fields),
    "question_obs": (QUESTION_OBS_SYSTEM_PROMPT, QuestionAnswerObs, question_obs_fields),
    "identify": (IDENTIFY_SYSTEM_PROMPT, IdentifyAnswer, identify_fields),
}


def out_conflict(records: list[dict], model: str, fmt: str) -> str | None:
    # 이미 기록한 사진은 건너뛰므로, 다른 모델·형식 파일에 이어 쓰면 두 결과가 한 파일에 조용히 섞인다
    # format 칸이 없는 예전 기록은 점수 형식이다
    found = {(r["model"], r.get("format", "score")) for r in records}
    if found - {(model, fmt)}:
        return f"결과 파일에 다른 모델·형식 기록이 있음 {sorted(found)}. 이번 실행({model}, {fmt})은 --out을 따로 쓸 것"
    return None


def add_format_args(ap: argparse.ArgumentParser) -> None:
    # baseline_verify(평가셋 A)·measure_eval_b(B·B 보강)가 같은 옵션을 쓴다
    ap.add_argument("--format", choices=tuple(FORMATS), default="score", help="score = 지금 서비스(점수), question = 객관식, question_v2 = 장소 질문만 바꾼 객관식, "
                         "question_obs = 관찰 칸을 먼저 쓰는 객관식, identify = 목표 장소 없이 장소 알아보기")
    ap.add_argument("--split-file", type=pathlib.Path, help="개선용·최종용 나누기 결과 (eval_split.json)")
    ap.add_argument("--split", choices=("dev", "final"), help="이쪽 사진만 모델에 보낸다")
    ap.add_argument("--open-final", action="store_true", help="final 사진도 보낸다 (--split final 또는 나누기 없이). 출시 판단 때만")


async def judge(case: dict, image_bytes: bytes, fmt: str = "score") -> dict:
    # score_photo와 같은 입력으로 1번만 호출한다. 재시도하지 않아야 1차 성공률과 1회 비용을 그대로 잴 수 있다
    system_prompt, schema, fields = FORMATS[fmt]
    t0 = time.perf_counter()
    processed = normalize_image(image_bytes)
    t1 = time.perf_counter()
    contents = [
        user_text(case, fmt),
        types.Part.from_bytes(data=processed, mime_type="image/jpeg"),
    ]
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        response_schema=schema,
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
        parsed=isinstance(response.parsed, schema),
    )
    if isinstance(response.parsed, schema):
        record.update(fields(response.parsed))
    return record


async def judge_local(
    case: dict, image_bytes: bytes, base_url: str, model: str, api_key: str = "", fmt: str = "score",
    logprobs: bool = False,
) -> dict:
    # judge와 같은 프롬프트·리사이즈·응답 형식으로 OpenAI 호환 서버를 1번 부른다. 비용은 GPU 시간으로 따로 계산하므로 0으로 적는다
    # logprobs면 장소 답 자리의 선택지별 확률(place_probs)도 남긴다 (장소 구별 실험 E)
    system_prompt, schema, fields = FORMATS[fmt]
    t0 = time.perf_counter()
    processed = normalize_image(image_bytes)
    t1 = time.perf_counter()
    image_url = "data:image/jpeg;base64," + base64.b64encode(processed).decode()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_text(case, fmt)},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]},
        ],
        # Gemini의 response_schema와 같은 역할. 서버가 이 스키마 모양으로만 답하게 막는다
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
        },
    }
    if logprobs:
        payload.update(logprobs=True, top_logprobs=10)
    record = {"normalize_sec": round(t1 - t0, 3), "resized_bytes": len(processed), "original_bytes": len(image_bytes)}
    try:
        # 첫 호출은 모델을 메모리에 올리느라(서버리스는 콜드스타트) 수십 초 걸릴 수 있어 넉넉히 기다린다
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                f"{base_url.rstrip('/')}/chat/completions", json=payload, headers=auth_headers(api_key),
            )
            response.raise_for_status()
    except Exception as e:  # noqa: BLE001 — 측정 스크립트라 실패도 결과로 기록한다
        record.update(model_sec=round(time.perf_counter() - t1, 3), error=f"{type(e).__name__}: {e}")
        return record
    record["model_sec"] = round(time.perf_counter() - t1, 3)

    body = response.json()
    usage = body.get("usage") or {}
    message = body["choices"][0]["message"]
    content = message.get("content") or ""
    # 추론 모델을 vLLM reasoning parser로 띄우면 생각은 content가 아닌 칸에 따로 온다 (실험 B). 출력 토큰에는 생각도 들어 있다
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    record.update(
        prompt_tokens=usage.get("prompt_tokens", 0), image_tokens=0,
        output_tokens=usage.get("completion_tokens", 0), thought_tokens=0, cost_usd=0.0,
        reasoning_chars=len(reasoning),
    )
    if logprobs:
        tokens = (body["choices"][0].get("logprobs") or {}).get("content") or []
        record["place_probs"] = option_probs(tokens, *prob_field(fmt))
    try:
        record.update(parsed=True, **fields(schema.model_validate_json(content)))
    except ValidationError:
        # 스키마를 벗어난 응답은 1차 실패로 센다. 원인을 보려고 앞부분만 남긴다
        record.update(parsed=False, raw=content[:500])
    return record


def summarize(out: pathlib.Path) -> None:
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    ok = [r for r in records if "error" not in r]
    print(f"\n[요약] 기록 {len(records)}건 / 에러 {len(records) - len(ok)}건")
    if not ok:
        return
    parsed = [r for r in ok if r.get("parsed")]
    print(f"  구조화 출력 1차 성공률: {len(parsed)}/{len(ok)}")
    if any("answer" in r for r in parsed):
        # 객관식 기록은 점수가 없다. 등급 분포만 본다
        for kind in ("positive", "negative"):
            grades = collections.Counter(rule_grade(QuestionAnswer.model_validate(r["answer"])) for r in parsed if r["kind"] == kind)
            if grades:
                print(f"  {kind} 등급: {dict(grades)}")
    secs = sorted(r["model_sec"] for r in ok)
    p95 = secs[min(len(secs) - 1, int(len(secs) * 0.95))]
    print(f"  모델 호출 시간: 평균 {statistics.mean(secs):.2f}초 / p95 {p95:.2f}초")
    print(f"  총 비용: ${sum(r['cost_usd'] for r in ok):.4f}")
    for kind in ("positive", "negative"):
        scores = [r["match_score"] for r in parsed if r["kind"] == kind and "match_score" in r]
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
    ap.add_argument("--model", default=gemini_client.MODEL_NAME,
                    help="Gemini 모델 이름, 또는 OpenAI 호환 서버의 모델 이름 (예: qwen2.5vl:7b, Qwen/Qwen2.5-VL-7B-Instruct)")
    ap.add_argument("--base-url", default=DEFAULT_LOCAL_URL, help="OpenAI 호환 서버 주소 (로컬 Ollama 또는 RunPod)")
    ap.add_argument("--same-as", type=pathlib.Path, help="이 결과 파일에서 성공한 건과 같은 사진·미션만 호출")
    ap.add_argument("--lookalike-only", action="store_true", help="월정교↔동궁과월지 오답 미션만 호출")
    ap.add_argument("--all-negatives", action="store_true", help="사진마다 나머지 장소 미션을 모두 만든다 (기본은 1개)")
    ap.add_argument("--with-positives", action="store_true",
                    help="--lookalike-only와 함께: 같은 사진에 맞는 장소 미션도 넣는다 (오답만 보면 '아니오'만 하는 모델도 좋아 보인다)")
    ap.add_argument("--logprobs", action="store_true", help="장소 답의 선택지별 확률을 남긴다 (OpenAI 호환 서버만)")
    add_format_args(ap)
    args = ap.parse_args()
    # 평가셋 A도 개선용·최종용으로 나뉘어 있다. 끝의 요약에 final이 섞이지 않게 나누기를 요구한다
    if error := split_error(args.split, args.split_file, args.open_final, needs_split=True):
        ap.error(error)
    if args.same_as and args.same_as.resolve() == args.out.resolve():
        ap.error("--same-as와 --out은 다른 파일이어야 함")
    if args.with_positives and not args.lookalike_only:
        ap.error("--with-positives는 --lookalike-only와 함께 쓴다")
    local = is_openai_compat(args.model)
    if args.logprobs and not local:
        ap.error("--logprobs는 OpenAI 호환 서버(vLLM 등)에서만 쓸 수 있음")
    # judge는 호출할 때마다 gemini_client.MODEL_NAME을 읽으므로 여기서 바꾸면 이 실행 전체가 이 모델로 판정한다
    gemini_client.MODEL_NAME = args.model

    rows = load_mapping(args.mapping)
    index = index_photos(args.photos)
    cases = build_cases(rows, set(index), args.per_place, args.seed, args.all_negatives)
    if args.lookalike_only:
        cases = lookalike_cases(cases, args.with_positives)
    if args.format == "identify":
        # 목표 장소를 알려 주지 않으므로 사진마다 한 번만 묻는다 (positive 케이스 = 사진 한 장)
        cases = [c for c in cases if c["kind"] == "positive"]
    if args.same_as:
        base = [json.loads(line) for line in args.same_as.read_text(encoding="utf-8").splitlines() if line.strip()]
        keep = {case_key(r) for r in base if "error" not in r}
        cases = [c for c in cases if case_key(c) in keep]
    if args.split:
        photos = load_split(args.split_file, args.split)
        cases = [c for c in cases if c["photo"] in photos]
    records = []
    if args.out.exists():
        records = [json.loads(line) for line in args.out.read_text(encoding="utf-8").splitlines() if line.strip()]
    if error := out_conflict(records, args.model, args.format):
        ap.error(error)
    done = {case_key(r) for r in records if "error" not in r}
    todo = [c for c in cases if case_key(c) not in done]
    if args.max_calls:
        todo = todo[: args.max_calls]

    key_note = "키 있음" if settings.vlm_api_key else "키 없음"  # 키 값은 출력하지 않는다
    print(f"모델: {gemini_client.MODEL_NAME}" + (f" (OpenAI 호환, {args.base_url}, {key_note})" if local else "")
          + f" · 형식 {args.format}" + (f" · {args.split}만" if args.split else ""))
    print(f"매핑 {len(rows)}장 중 찾은 사진 {sum(r['photo'] in index for r in rows)}장")
    for place in LANDMARKS:
        n = sum(1 for c in cases if c["place"] == place)
        print(f"  {place} 사진: 케이스 {n}건")
    print(f"케이스 {len(cases)}건 (이미 기록 {len(done & {case_key(c) for c in cases})}건) → 이번에 호출할 것 {len(todo)}건")
    if local:
        print("예상 비용: 0원 (로컬 모델)")
    else:
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
                image_bytes = read_photo(index[case["photo"]])
                if local:
                    record = await judge_local(
                        case, image_bytes, args.base_url, args.model, settings.vlm_api_key, args.format, args.logprobs)
                else:
                    record = await judge(case, image_bytes, args.format)
            except Exception as e:  # noqa: BLE001 — 사진 하나가 깨져도 전체 실행을 멈추지 않는다
                record = {"model_sec": 0.0, "error": f"사진 처리 실패 {type(e).__name__}: {e}"}
            distance = _haversine_km(case["photo_coords"], LANDMARKS[case["target"]]) if case["photo_coords"] else None
            line = {
                "photo": case["photo"], "place": case["place"], "kind": case["kind"], "target": case["target"],
                "distance_km": round(distance, 3) if distance is not None else None, "model": gemini_client.MODEL_NAME,
                "format": args.format, **record,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            f.flush()
            score = line.get("match_score", line.get("grade", line.get("identify", line.get("error", "-"))))
            print(f"[{i}/{len(todo)}] {case['photo']} → {case['target']} [{case['kind']}]: {score} ({line['model_sec']}초)")
    summarize(args.out)


if __name__ == "__main__":
    asyncio.run(main())
