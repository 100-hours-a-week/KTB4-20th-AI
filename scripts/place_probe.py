"""장소 구별 실험 (2026-10-15). 학습 전에 Qwen이 닮은 장소를 못 가리는 원인을 나눠 본다.

10/13 제로샷에서 Qwen은 장소 질문에 "아니오"를 169번 중 0번 답했다. 원인 후보는 셋이다.
  ① 생각할 자리가 없다: 답 형식이 칸 4개뿐이라 사진을 보자마자 "yes"를 쓴다
  ② 장소 지식이 없다: 월정교·동궁과월지가 어떻게 생겼는지 모른다
  ③ 맞장구: 요청에 "목표 장소: OO"를 먼저 알려 주니 그 말에 맞춘다

실험 (baseline_verify --format / --logprobs로 고른다)
  A question_obs: 객관식 앞에 "장소를 알 수 있는 특징" 글 칸을 먼저 쓰게 한다 (①)
  D identify: 목표 장소를 알려 주지 않고 경주 4곳·기타·모름 중 어디인지 묻는다 (②·③)
  E --logprobs: 장소 답을 한 단어로 고르게 하지 않고 선택지별 확률을 받아 기준값으로 자른다
  B 추론 모델: 코드는 그대로 두고 --model만 바꾼다. 생각한 글자 수(reasoning_chars)를 남긴다

  # 결과 요약 (레포 루트에서)
  uv run python -m scripts.place_probe <결과 jsonl> [<결과 jsonl> ...]
"""

import argparse
import collections
import json
import math
import pathlib
import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from scripts.question_format import QUESTION_SYSTEM_PROMPT, QuestionAnswer, rule_grade

# A: 객관식 지시문 앞에 관찰 칸 설명만 더한다. 나머지 질문은 question(v1)과 글자까지 같아 효과를 따로 볼 수 있다
_OBSERVATION = """\
[observation] 다른 질문에 답하기 전에 먼저 쓴다.
사진에서 장소를 알아볼 수 있는 특징을 보이는 그대로 1~3문장으로 쓴다.
건물·구조물의 모양과 재질, 다리·연못·탑·무덤 같은 지형, 현판·안내판의 글자가 있으면 적는다.
목표 장소 이름을 쓰지 말고, 사진에 보이는 것만 쓴다. 아래 [place]는 이 관찰을 근거로 답한다.

"""
_FIRST_QUESTION = "[judgeable]"
assert QUESTION_SYSTEM_PROMPT.count(_FIRST_QUESTION) == 1
QUESTION_OBS_SYSTEM_PROMPT = QUESTION_SYSTEM_PROMPT.replace(_FIRST_QUESTION, _OBSERVATION + _FIRST_QUESTION)


class QuestionAnswerObs(BaseModel):
    # 칸 순서가 생성 순서다. 관찰을 먼저 써야 장소 답을 쓰기 전에 생각할 자리가 생긴다
    observation: str = Field(max_length=400)
    judgeable: Literal["yes", "no"]
    target: Literal["enough", "short", "none"] | None
    condition: Literal["yes", "no", "na"] | None
    place: Literal["yes", "no", "unknown", "na"] | None

    @model_validator(mode="after")
    def _same_rules_as_question(self) -> "QuestionAnswerObs":
        self.to_answer()  # 칸이 비어 "성공"으로 새는 답은 question과 똑같이 형식 오류로 막는다
        return self

    def to_answer(self) -> QuestionAnswer:
        return QuestionAnswer.model_validate(self.model_dump(exclude={"observation"}))


def question_obs_fields(a: QuestionAnswerObs) -> dict:
    # answer는 question과 같은 모양으로 남겨 eval_cases가 그대로 셀 수 있게 한다
    answer = a.to_answer()
    return {"observation": a.observation, "answer": answer.model_dump(), "grade": rule_grade(answer)}


# D: 목표 장소를 알려 주지 않는다. 선택지 이름은 영문 코드로 둔다 (한글은 토큰이 바이트로 쪼개져 확률을 묶기 어렵다)
IDENTIFY_PLACES = {
    "cheomseongdae": "첨성대",
    "daereungwon": "대릉원",
    "donggung_wolji": "동궁과월지",
    "woljeonggyo": "월정교",
}
IDENTIFY_SYSTEM_PROMPT = """\
너는 여행 사진을 보고 찍은 장소를 알아보는 AI다. 사진은 경주에서 찍었다.

[observation] 먼저 사진에서 장소를 알아볼 수 있는 특징을 보이는 그대로 1~3문장으로 쓴다.
건물·구조물의 모양과 재질, 다리·연못·탑·무덤 같은 지형, 현판·안내판의 글자가 있으면 적는다.

[answer] 관찰을 근거로 사진을 찍은 장소를 고른다.
- "cheomseongdae": 첨성대
- "daereungwon": 대릉원
- "donggung_wolji": 동궁과월지 (안압지)
- "woljeonggyo": 월정교
- "other": 위 네 곳이 아닌 다른 장소
- "unknown": 사진만으로 알 수 없다
"""
IDENTIFY_USER_TEXT = "이 사진을 찍은 장소는 어디인가?"


class IdentifyAnswer(BaseModel):
    observation: str = Field(max_length=400)
    answer: Literal["cheomseongdae", "daereungwon", "donggung_wolji", "woljeonggyo", "other", "unknown"]


def identify_fields(a: IdentifyAnswer) -> dict:
    # "answer" 칸은 객관식 기록이 쓰는 이름이라 겹치지 않게 identify로 남긴다
    return {"observation": a.observation, "identify": a.answer}


# E: 장소 답 자리의 선택지별 확률. JSON 값 그대로(따옴표 포함) 비교한다
PROB_FIELDS = {
    "question": ("place", ['"yes"', '"no"', '"unknown"', '"na"', "null"]),
    "identify": ("answer", [f'"{k}"' for k in [*IDENTIFY_PLACES, "other", "unknown"]]),
}


def _token_bytes(t: dict) -> bytes:
    # 한글은 토큰 하나가 글자 일부(바이트)일 수 있어 글자 수가 아니라 바이트로 위치를 맞춘다
    return bytes(t["bytes"]) if t.get("bytes") is not None else t["token"].encode()


def option_probs(tokens: list[dict], key: str, options: list[str]) -> dict[str, float] | None:
    # tokens: OpenAI 호환 응답의 choices[0].logprobs.content. "key": 뒤 값이 처음 갈리는 토큰에서 선택지별 확률을 더한다
    # 상위 후보(top_logprobs)에 없는 선택지는 0으로 남고, 어느 선택지에도 안 맞는 후보는 other로 모은다
    text = b"".join(_token_bytes(t) for t in tokens)
    m = re.search(rb'"' + re.escape(key.encode()) + rb'"\s*:\s*', text)
    if not m:
        return None
    start = m.end()
    opts = [o.encode() for o in options]
    pos = 0
    for t in tokens:
        tb = _token_bytes(t)
        end = pos + len(tb)
        if end <= start:
            pos = end
            continue
        lead = tb[: max(0, start - pos)]  # 값 앞(키·콜론) 부분이 같은 토큰에 붙어 있으면 떼어 낸다
        before = text[start:max(start, pos)]  # 이 토큰 전까지 이미 나온 값의 앞부분 (예: 여는 따옴표)
        if (before + tb[len(lead):]).strip(b'"') == b"":
            pos = end  # 따옴표만 나온 토큰이면 아직 어느 선택지인지 갈리지 않았다
            continue
        probs = {o: 0.0 for o in options} | {"other": 0.0}
        for alt in t.get("top_logprobs") or []:
            ab = _token_bytes(alt)
            if not ab.startswith(lead):
                continue
            cand = before + ab[len(lead):]
            hits = [o for o in opts if o.startswith(cand) or cand.startswith(o)]
            bucket = hits[0].decode() if len(hits) == 1 else "other"
            probs[bucket] += math.exp(alt["logprob"])
        return {k: round(v, 6) for k, v in probs.items()}
    return None


def prob_field(fmt: str) -> tuple[str, list[str]]:
    return PROB_FIELDS["identify" if fmt == "identify" else "question"]


# 요약

THRESHOLDS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)


def _load(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} ({n / d:.1%})" if d else "0/0"


def _grade(r: dict) -> str:
    # 기록 당시 grade가 아니라 답 원본에서 지금 규칙으로 다시 정한다 (eval_cases·baseline_verify와 같게)
    return rule_grade(QuestionAnswer.model_validate(r["answer"]))


def summarize(records: list[dict]) -> list[str]:
    ok = [r for r in records if "error" not in r and r.get("parsed")]
    lines = [f"기록 {len(records)}건 / 형식 통과 {len(ok)}건"]
    if not ok:
        return lines
    secs = sorted(r["model_sec"] for r in ok)
    lines.append(f"모델 시간 중앙 {secs[len(secs) // 2]:.2f}초 / 최대 {secs[-1]:.2f}초 · 출력 토큰 중앙 "
                 f"{sorted(r.get('output_tokens', 0) for r in ok)[len(ok) // 2]} · 생각 글자 중앙 "
                 f"{sorted(r.get('reasoning_chars', 0) for r in ok)[len(ok) // 2]}")

    ident = [r for r in ok if "identify" in r]
    if ident:
        # D: 정답은 사진을 찍은 장소. 맞장구 없이 장소를 알아보는지 본다
        for place in sorted({r["place"] for r in ident}):
            rs = [r for r in ident if r["place"] == place]
            right = sum(IDENTIFY_PLACES.get(r["identify"]) == place for r in rs)
            lines.append(f"[알아보기] {place} 사진 → 맞힘 {_pct(right, len(rs))} · 답 분포 "
                         f"{dict(collections.Counter(r['identify'] for r in rs))}")
        return lines

    # A·E: 닮은 장소 오답(negative)은 장소 "아니오"·실패가 맞고, 맞는 장소(positive)는 장소 "예"가 맞다
    for kind in ("negative", "positive"):
        rs = [r for r in ok if r["kind"] == kind and "answer" in r]
        if not rs:
            continue
        places = collections.Counter(r["answer"]["place"] for r in rs)
        grades = collections.Counter(_grade(r) for r in rs)
        lines.append(f"[{kind}] {len(rs)}건 · 장소 답 {dict(places)} · 등급 {dict(grades)}")
    neg = [r for r in ok if r["kind"] == "negative" and "answer" in r]
    pos = [r for r in ok if r["kind"] == "positive" and "answer" in r]
    lines.append(f"닮은 장소 통과(성공 판정) {_pct(sum(_grade(r) == 'success' for r in neg), len(neg))}"
                 f" · 맞는 장소 성공 판정 {_pct(sum(_grade(r) == 'success' for r in pos), len(pos))}")

    probs = [r for r in ok if r.get("place_probs") and "answer" in r]
    if probs:
        # E: 장소 "예" 확률이 기준값 미만이면 장소를 확인 못 한 것으로 보고 실패시킨다고 가정해 다시 센다
        lines.append("[확률 기준] 장소 '예' 확률 < 기준값이면 실패로 볼 때")
        for th in THRESHOLDS:
            def passed(r: dict, th: float = th) -> bool:
                return _grade(r) == "success" and r["place_probs"]['"yes"'] >= th
            n = [r for r in probs if r["kind"] == "negative"]
            p = [r for r in probs if r["kind"] == "positive"]
            lines.append(f"  기준 {th:.2f}: 닮은 장소 통과 {_pct(sum(map(passed, n)), len(n))}"
                         f" · 맞는 장소 성공 {_pct(sum(map(passed, p)), len(p))}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=pathlib.Path, nargs="+")
    args = ap.parse_args()
    for path in args.results:
        print(f"\n== {path.name}")
        for line in summarize(_load(path)):
            print(line)


if __name__ == "__main__":
    main()
