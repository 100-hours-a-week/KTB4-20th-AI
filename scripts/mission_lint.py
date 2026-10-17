"""generate가 만든 미션 문구에서 프롬프트 규칙 위반으로 의심되는 표현을 찾는다.

정규식이라 놓치는 표현도, 잘못 걸리는 문구도 있다. 차단용이 아니라 사람이 다시 볼 미션을 고르는 경고용이다.
경고 수를 위반 수로 보고하지 않는다 (예: "단풍나무 간판 찍기"도 계절로 걸린다).
서비스(pipeline_generate)는 이 검사를 쓰지 않는다.

  # 레포 루트에서 실행. 결과 파일은 레포 밖에 둔다
  uv run python -m scripts.mission_lint /Volumes/T7/aihub/eval_b/missions.jsonl
"""

import argparse
import collections
import json
import pathlib
import re

# 프롬프트 규칙(MISSION_SYSTEM_PROMPT)마다 의심 표현. 키는 집계에 그대로 나온다
WARNINGS = {
    # 8번: 들고·앞에서·가까이·다 같이 밖의 동작·자세·과정
    "동작·과정": re.compile(
        r"하며|며 찍|[는은한] 모습|뒷모습|점프|브이|만세|포즈|흔들며|흔드는|흔들고|가리키|앉아|서서|뛰|올려다|걷|착용"
    ),
    # 6번: 사진으로 확인할 수 없는 마음·선택·구매 이력
    "마음·선택": re.compile(
        r"마음에 드는|고른|골라|선택한|사고 싶은|좋아하는|인상적인|직접 산|구매한|구입한|서로 다른|각자 다른"
    ),
    # 5번: 방문 때 있을지 모르는 계절 대상
    "계절": re.compile(r"핀 꽃|꽃 핀|단풍|벚꽃|낙엽|눈 덮인|눈 내리는"),
}
# 7번: "다 같이"는 "○○ 앞에서/배경으로"와 함께 써야 사람이 나와야 하는 미션으로 읽힌다
# 두 표현이 문구에 있는지만 본다. 어순이나 구문 전체가 맞는지는 사람이 본다
TOGETHER = re.compile(r"다 ?같이")
TOGETHER_PLACE = re.compile(r"앞에서|배경으로")


def warnings(description: str) -> list[str]:
    found = [name for name, pattern in WARNINGS.items() if pattern.search(description)]
    if TOGETHER.search(description) and not TOGETHER_PLACE.search(description):
        found.append("다 같이 형태")
    return found


def summarize(rows: list[dict], field: str) -> dict:
    texts = [r[field] for r in rows]
    per_text = [warnings(t) for t in texts]
    by_batch = collections.defaultdict(list)
    for r in rows:
        by_batch[r.get("batch", "")].append(r[field])
    return {
        "missions": len(texts),
        "any_warning": sum(bool(w) for w in per_text),
        "by_warning": dict(collections.Counter(w for ws in per_text for w in ws)),
        # 같은 요청(묶음) 안에서 똑같은 문구가 또 나온 수. 다른 묶음끼리의 반복은 따로 센다
        "dup_in_batch": sum(len(b) - len(set(b)) for b in by_batch.values()),
        "dup_overall": len(texts) - len(set(texts)),
        "scope": dict(collections.Counter(r.get("scope") for r in rows)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("missions", type=pathlib.Path, help="generate_eval_missions.py가 만든 JSONL")
    ap.add_argument("--field", default="mission_description")
    ap.add_argument("--show", action="store_true", help="경고가 난 문구를 모두 출력")
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.missions.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if "error" not in r]
    print(json.dumps(summarize(rows, args.field), ensure_ascii=False, indent=1))
    if args.show:
        for r in rows:
            if found := warnings(r[args.field]):
                print(f"  {', '.join(found)} | {r[args.field]}")


if __name__ == "__main__":
    main()
