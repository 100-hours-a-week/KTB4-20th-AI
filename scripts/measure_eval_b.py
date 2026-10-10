"""평가셋 B(원본 400장, 보강 172장)를 verify와 같은 입력으로 판정해 사람 정답과 함께 기록한다.

판정 호출은 baseline_verify의 judge·judge_local을 그대로 쓴다(재시도 없이 1회 호출, 같은 프롬프트·리사이즈).
  - B   : generate가 만든 원래 미션 → 정답 대부분 실패, 엉뚱한 사진을 통과시키는지(FP) 본다
  - B_aug: 사진을 보고 쓴 미션 → 정답 대부분 성공, 맞게 찍은 사진을 막는지(FN) 본다

  # 레포 루트에서 실행. Pro는 하루 호출 한도가 있어 --max-calls로 나눠 돌린다
  uv run python -m scripts.measure_eval_b --aihub /Volumes/T7/aihub --set B \\
      --out /Volumes/T7/aihub/eval_b/results/pro_B.jsonl --dry-run
  uv run python -m scripts.measure_eval_b --aihub /Volumes/T7/aihub --set B \\
      --out /Volumes/T7/aihub/eval_b/results/pro_B.jsonl --max-calls 200

  같은 --out으로 다시 실행하면 이미 기록한 건은 건너뛴다.
"""

import argparse
import asyncio
import collections
import json
import pathlib
import zipfile

from app.photomissions import gemini_client
from app.photomissions.pipeline_verify import to_grade
from scripts.baseline_verify import (
    DEFAULT_LOCAL_URL,
    LOCAL_MODELS,
    PRICES,
    judge,
    judge_local,
)

SETS = ("B", "B_aug")


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_cases(manifest: list[dict], missions: list[dict], labels: list[dict], aug_items: list[dict]) -> list[dict]:
    # 두 세트를 같은 모양의 케이스로 만든다. 정답(truth)은 사람이 표시한 등급이고, 표시가 없는 사진은 뺀다
    rows = {row["photo"]: row for row in manifest}
    truth_b = {x["photo"]: x for x in labels}
    cases = []
    for m in missions:
        if "error" in m or m["photo"] not in truth_b:
            continue
        cases.append(_case("B", rows[m["photo"]], truth_b[m["photo"]], m["mission_description"]))
    for x in aug_items:
        cases.append(_case("B_aug", rows[x["photo"]], x, x["mission"]))
    return cases


def _case(eval_set: str, row: dict, truth: dict, mission: str) -> dict:
    return {
        "eval_set": eval_set, "photo": row["photo"], "zip": row["zip"], "member": row["member"],
        "place": row["place_name"], "target": row["place_name"], "region": truth["region"],
        "category": truth["category"], "mission": mission, "truth": truth["label"], "kind": "eval_b",
    }


def case_key(case: dict) -> str:
    return f"{case['eval_set']}|{case['photo']}"


def summarize(out: pathlib.Path) -> None:
    # 등급 일치만 간단히 본다. FP·FN 종류별 집계는 eval_cases 통합 후 오류 분류 함수로 한다
    ok = [r for r in _read_jsonl(out) if "error" not in r and r.get("parsed")]
    print(f"\n[요약] 판정 {len(ok)}건")
    for eval_set in SETS:
        rs = [r for r in ok if r["eval_set"] == eval_set]
        if not rs:
            continue
        table = collections.Counter((r["truth"], to_grade(r["match_score"])) for r in rs)
        agree = sum(n for (t, p), n in table.items() if t == p)
        print(f"  {eval_set}: 정답 일치 {agree}/{len(rs)} ({agree / len(rs):.0%})")
        for truth in ("success", "retry", "fail"):
            row = " / ".join(f"{pred} {table[(truth, pred)]}" for pred in ("success", "retry", "fail"))
            print(f"    정답 {truth:7s} → 판정 {row}")


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aihub", type=pathlib.Path, required=True)
    ap.add_argument("--set", choices=(*SETS, "all"), default="all")
    ap.add_argument("--out", type=pathlib.Path, required=True, help="결과 JSONL (레포 밖에 둘 것)")
    ap.add_argument("--max-calls", type=int, default=0, help="이번 실행의 최대 호출 수 (0이면 제한 없음)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--model", default=gemini_client.MODEL_NAME, choices=sorted(PRICES) + list(LOCAL_MODELS))
    ap.add_argument("--base-url", default=DEFAULT_LOCAL_URL, help="로컬·RunPod 모델의 OpenAI 호환 주소")
    args = ap.parse_args()
    local = args.model in LOCAL_MODELS
    gemini_client.MODEL_NAME = args.model

    eval_b = args.aihub / "eval_b"
    cases = build_cases(
        _read_jsonl(eval_b / "manifest.jsonl"),
        _read_jsonl(eval_b / "missions.jsonl"),
        json.loads((eval_b / "eval_b_labels.json").read_text(encoding="utf-8"))["labels"],
        json.loads((eval_b / "eval_b_aug.json").read_text(encoding="utf-8"))["items"],
    )
    if args.set != "all":
        cases = [c for c in cases if c["eval_set"] == args.set]
    done = {case_key(r) for r in _read_jsonl(args.out) if "error" not in r} if args.out.exists() else set()
    todo = [c for c in cases if case_key(c) not in done]
    if args.max_calls:
        todo = todo[: args.max_calls]

    counts = collections.Counter(c["eval_set"] for c in cases)
    print(f"모델: {args.model}" + (f" (OpenAI 호환, {args.base_url})" if local else ""))
    print(f"케이스 {dict(counts)} (이미 기록 {len(cases) - len([c for c in cases if case_key(c) not in done])}건) → 이번에 호출할 것 {len(todo)}건")
    if not local:
        per_call_won = 27 * PRICES[args.model][0] / PRICES["gemini-3.1-pro-preview"][0]
        print(f"예상 비용: 약 {len(todo) * per_call_won:,.0f}원 (pro-preview 1회 27원 실측을 단가 비율로 환산)")
    if args.dry_run:
        for c in todo[:5]:
            print(f"  - [{c['eval_set']}] {c['place']} · {c['mission']} (정답 {c['truth']})")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    zips: dict[str, zipfile.ZipFile] = {}
    with args.out.open("a", encoding="utf-8") as f:
        for i, case in enumerate(todo, 1):
            try:
                zf = zips.setdefault(case["zip"], zipfile.ZipFile(args.aihub / case["zip"]))
                image = zf.read(case["member"])
                record = await (judge_local(case, image, args.base_url, args.model) if local else judge(case, image))
            except Exception as e:  # noqa: BLE001 — 사진 하나가 깨져도 전체 실행을 멈추지 않는다
                record = {"model_sec": 0.0, "error": f"사진 처리 실패 {type(e).__name__}: {e}"}
            line = {k: case[k] for k in ("eval_set", "photo", "place", "region", "category", "mission", "truth")}
            line.update(model=args.model, **record)
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            f.flush()
            shown = line.get("match_score", line.get("error", "-"))
            print(f"[{i}/{len(todo)}] [{case['eval_set']}] {case['place']}: {shown} (정답 {case['truth']}, {line['model_sec']}초)")
    summarize(args.out)


if __name__ == "__main__":
    asyncio.run(main())
