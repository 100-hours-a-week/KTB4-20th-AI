"""평가셋 A·B·B 보강의 판정 결과를 한 형식(EvalCase)으로 모으고, 판정 오류를 종류별로 집계한다.

Positive = "성공" 판정이다. 사람 정답과 모델 등급을 비교해 각 건을 다음 중 하나로 나눈다.
  TP          정답 성공 → 성공
  TN          정답이 성공이 아님 → 성공이 아님 (재시도·실패끼리 엇갈려도 사용자가 통과하지 못한 것은 같다)
  FN_LIGHT    정답 성공 → 재시도 (다시 찍으면 된다)
  FN_HEAVY    정답 성공 → 실패
  FP_LIGHT    정답 재시도(가림·잘림 등) → 성공
  FP          정답 실패, 목표 장소 사진 → 성공 (장소는 맞는데 미션을 안 함)
  FP_CRITICAL 다른 장소·닮은 장소 사진 → 성공

  # 레포 루트에서 실행. 결과 파일은 측정 스크립트가 만든 JSONL
  uv run python -m scripts.eval_cases \\
      --a /Volumes/T7/aihub/278_동부권/baseline/pro-preview_default.jsonl \\
      --a /Volumes/T7/aihub/278_동부권/baseline/pro-preview_lookalike100.jsonl \\
      --a-labels /Volumes/T7/aihub/labeling/gyeongju_labels.json \\
      --b /Volumes/T7/aihub/eval_b/results/pro_B.jsonl
"""

import argparse
import collections
import json
import pathlib
from typing import Literal

from pydantic import BaseModel

from app.photomissions.pipeline_verify import RETRY_THRESHOLD, SUCCESS_THRESHOLD
from scripts.baseline_verify import LOOKALIKE

Grade = Literal["success", "retry", "fail"]
Outcome = Literal["TP", "TN", "FN_LIGHT", "FN_HEAVY", "FP_LIGHT", "FP", "FP_CRITICAL"]


class EvalCase(BaseModel):
    eval_set: Literal["A", "B", "B_aug"]
    photo: str
    place: str  # 사진을 찍은 장소
    target: str  # 미션의 목표 장소
    case_type: Literal["target_place", "other_place", "lookalike"]
    mission_source: Literal["landmark_fixed", "generate", "photo_written"]
    category: str
    truth: Grade  # 사람 정답
    model: str
    match_score: float | None  # 모델이 답하지 못했으면(에러·형식 불일치) None


def grade(score: float, success: float = SUCCESS_THRESHOLD, retry: float = RETRY_THRESHOLD) -> Grade:
    # 서비스의 to_grade와 같은 규칙. 임계값을 바꿔 보는 개선 ①을 위해 값을 받을 수 있게 한다
    if score >= success:
        return "success"
    return "retry" if score >= retry else "fail"


def classify(case: EvalCase, success: float = SUCCESS_THRESHOLD, retry: float = RETRY_THRESHOLD) -> Outcome | None:
    if case.match_score is None:
        return None  # 판정 자체가 없으면 오류 종류로 세지 않고 따로 센다
    passed = grade(case.match_score, success, retry) == "success"
    if case.case_type != "target_place":
        return "FP_CRITICAL" if passed else "TN"
    if case.truth == "success":
        if passed:
            return "TP"
        return "FN_LIGHT" if grade(case.match_score, success, retry) == "retry" else "FN_HEAVY"
    if not passed:
        return "TN"
    return "FP_LIGHT" if case.truth == "retry" else "FP"


def metrics(cases: list[EvalCase], success: float = SUCCESS_THRESHOLD, retry: float = RETRY_THRESHOLD) -> dict:
    # 지표마다 분모가 다르다. 각 오류가 "생길 수 있는 건"만 분모로 둔다
    judged = [(c, classify(c, success, retry)) for c in cases]
    judged = [(c, o) for c, o in judged if o is not None]
    count = collections.Counter(o for _, o in judged)

    def rate(n: int, d: int) -> float | None:
        return round(n / d, 4) if d else None

    target = [(c, o) for c, o in judged if c.case_type == "target_place"]
    truth_success = sum(c.truth == "success" for c, _ in target)
    by_type = {t: [o for c, o in judged if c.case_type == t] for t in ("other_place", "lookalike")}
    predicted_success = count["TP"] + count["FP_LIGHT"] + count["FP"] + count["FP_CRITICAL"]
    return {
        "judged": len(judged),
        "no_answer": len(cases) - len(judged),
        "counts": dict(count),
        "critical_fp_other": rate(by_type["other_place"].count("FP_CRITICAL"), len(by_type["other_place"])),
        "critical_fp_lookalike": rate(by_type["lookalike"].count("FP_CRITICAL"), len(by_type["lookalike"])),
        "fp": rate(count["FP"], sum(c.truth == "fail" for c, _ in target)),
        "fp_light": rate(count["FP_LIGHT"], sum(c.truth == "retry" for c, _ in target)),
        "fn_heavy": rate(count["FN_HEAVY"], truth_success),
        "fn_light": rate(count["FN_LIGHT"], truth_success),
        "precision": rate(count["TP"], predicted_success),
        "recall": rate(count["TP"], truth_success),
    }


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _score(record: dict) -> float | None:
    return record.get("match_score") if "error" not in record and record.get("parsed") else None


def from_a(records: list[dict], labels: dict[str, Grade]) -> list[EvalCase]:
    # 평가셋 A: 맞는 장소 미션은 사람 정답, 다른 장소 미션은 정답이 항상 실패다
    cases = []
    for r in records:
        if r["kind"] == "positive":
            if r["photo"] not in labels:
                continue
            case_type, truth = "target_place", labels[r["photo"]]
        else:
            case_type = "lookalike" if {r["place"], r["target"]} == LOOKALIKE else "other_place"
            truth = "fail"
        cases.append(EvalCase(
            eval_set="A", photo=r["photo"], place=r["place"], target=r["target"], case_type=case_type,
            mission_source="landmark_fixed", category="역사·문화", truth=truth, model=r["model"], match_score=_score(r),
        ))
    return cases


def from_b(records: list[dict]) -> list[EvalCase]:
    # 평가셋 B·B 보강: 사진을 찍은 장소의 미션만 있다
    return [
        EvalCase(
            eval_set=r["eval_set"], photo=r["photo"], place=r["place"], target=r["place"], case_type="target_place",
            mission_source="generate" if r["eval_set"] == "B" else "photo_written", category=r["category"],
            truth=r["truth"], model=r["model"], match_score=_score(r),
        )
        for r in records
    ]


def dedupe(cases: list[EvalCase]) -> list[EvalCase]:
    # 같은 사진·미션이 여러 번 기록됐으면 하나만 쓴다. 측정 파일은 에러(429 등) 뒤에 다시 잰 결과를 덧붙이므로
    # 답이 있는 기록을 우선하고, 둘 다 답이 있으면 먼저 읽은 것을 쓴다
    chosen: dict[tuple, EvalCase] = {}
    for c in cases:
        key = (c.eval_set, c.photo, c.target, c.mission_source)
        if key not in chosen or (chosen[key].match_score is None and c.match_score is not None):
            chosen[key] = c
    return list(chosen.values())


def _pct(v: float | None) -> str:
    return "-" if v is None else f"{v:.1%}"


def report(cases: list[EvalCase]) -> None:
    rows = [("전체", cases)] + [(s, [c for c in cases if c.eval_set == s]) for s in ("A", "B", "B_aug")]
    rows += [(f"  {cat}", [c for c in cases if c.category == cat]) for cat in sorted({c.category for c in cases})]
    head = ("구분", "건수", "치명FP(다른)", "치명FP(닮은)", "FP", "가벼운FP", "무거운FN", "가벼운FN", "정밀도", "재현율")
    print(" | ".join(head))
    for name, cs in rows:
        if not cs:
            continue
        m = metrics(cs)
        cells = (m["critical_fp_other"], m["critical_fp_lookalike"], m["fp"], m["fp_light"],
                 m["fn_heavy"], m["fn_light"], m["precision"], m["recall"])
        print(f"{name} | {m['judged']} | " + " | ".join(_pct(v) for v in cells))
    print(f"\n판정 없음(에러·형식 불일치): {sum(c.match_score is None for c in cases)}건")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", type=pathlib.Path, action="append", default=[], help="평가셋 A 결과 JSONL (여러 번 가능)")
    ap.add_argument("--a-labels", type=pathlib.Path, help="평가셋 A 사람 정답 (gyeongju_labels.json)")
    ap.add_argument("--b", type=pathlib.Path, action="append", default=[], help="평가셋 B·B 보강 결과 JSONL")
    ap.add_argument("--out", type=pathlib.Path, help="통합한 EvalCase를 JSONL로 저장 (레포 밖에 둘 것)")
    args = ap.parse_args()
    if args.a and not args.a_labels:
        ap.error("--a를 쓰려면 --a-labels가 필요함")

    cases: list[EvalCase] = []
    if args.a:
        labels = {x["photo"]: x["label"] for x in json.loads(args.a_labels.read_text(encoding="utf-8"))["labels"]}
        for path in args.a:
            cases += from_a(_read_jsonl(path), labels)
    for path in args.b:
        cases += from_b(_read_jsonl(path))
    cases = dedupe(cases)
    models = sorted({c.model for c in cases})
    print(f"모델 {models} · 케이스 {len(cases)}건\n")
    report(cases)
    if args.out:
        args.out.write_text("".join(c.model_dump_json() + "\n" for c in cases), encoding="utf-8")


if __name__ == "__main__":
    main()
