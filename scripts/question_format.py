"""verify 객관식 형식 (개선 ② 실험, 학습 데이터).

모델은 점수 대신 질문 네 개에 답하고, 등급은 서버가 규칙으로 정한다 (2026-10-04 웨일과 결정).
규칙은 docs/photomissions/label_criteria.md와 같다.
  판단 불가 · 다른 장소 · 대상 없음 → 실패 / 대상 부족 · 조건 안 지킴 · 장소 모름 → 재시도 / 나머지 → 성공
장소 모름 → 재시도는 2026-10-16에 더했다. 그 전에는 모름도 성공이었다 (evaluation.md 2절).
같은 규칙이 레포 밖(T7 $AIHUB) 표시 화면 두 곳에도 있다: train/pilot/labeling/index.html의 ruleGrade,
lookalike/relabel.html의 gradeOf. 사진 옆에 두는 로컬 도구라 테스트로 묶여 있지 않으니, 규칙을 바꾸면 두 화면도 직접 맞춘다.
"""

import json
from typing import Literal

from pydantic import BaseModel, model_validator

Grade = Literal["success", "retry", "fail"]

QUESTION_SYSTEM_PROMPT = """\
너는 여행 포토미션 인증 사진을 판정하는 AI다.
주어진 사진과 목표 장소·미션 내용을 보고 아래 질문에 순서대로 답한다. 사진에 실제로 보이는 것만 근거로 한다.
잘 찍었는지는 보지 않는다. 미션을 했는지만 본다.

[judgeable] 사진을 판단할 수 있는가?
- "no": 너무 어둡거나 심하게 흔들려 무엇인지 알 수 없다. 이때 나머지 질문은 모두 null로 둔다.
- "yes": 그 밖의 경우.

[target] 미션이 요구하는 대상이 미션에 충분할 만큼 보이는가?
- "enough": 대상이 분명히 보인다. 사람이 일부 가려도 알아볼 수 있으면 enough다.
  "가까이" 미션이면 대상이 사진에서 가장 크거나 무늬·질감이 보여야 enough다.
- "short": 대상은 있지만 가려졌거나, 너무 작거나, 잘렸거나, 간판·현판 같은 글씨만 보인다.
  (대상 자체가 간판인 미션이면 글씨가 곧 대상이다)
- "none": 대상이 아예 없다. 이때 condition은 null로 둔다.

[condition] 미션의 조건(들고, 다 같이, 각자, 가까이, 앉아서 같은 동작)을 지켰는가?
- "yes" / "no": 조건이 있을 때. "다 같이"는 2명 이상, "각자"는 사진 속 한 사람이 지키면 yes다.
  사람이 해야 하는 조건인데 사진에 사람이 없으면 no다.
  "앞에서"는 사람 없이 대상만 보여도 지킨 것이라 대상이 보이면 yes다.
- "na": 미션에 조건이 없다.

[place] 사진 속 장소가 목표 장소인가?
- "yes": 사진만 보고 목표 장소라고 알아볼 수 있다.
- "no": 사진만 보고 다른 장소라고 확신할 수 있다.
- "unknown": 사진만으로 구별할 수 없다.
- "na": 음식·바다처럼 장소를 알아보는 것이 미션과 상관없다.
"""

# v2: 장소 질문(place)만 바꾼 실험 지시문. 나머지 질문은 v1과 글자까지 같다
# 10/13 제로샷에서 다른·닮은 장소 사진 169건에 place "no"가 0건이었다(다른 장소 통과 61건이 "yes").
# 목표 장소 이름에 맞춰 답하지 말라고 알리고, "no"의 문턱("확신할 수 있다")을 사진 속 특징 기준으로 낮춘다
_PLACE_V1 = """\
[place] 사진 속 장소가 목표 장소인가?
- "yes": 사진만 보고 목표 장소라고 알아볼 수 있다.
- "no": 사진만 보고 다른 장소라고 확신할 수 있다.
- "unknown": 사진만으로 구별할 수 없다.
"""
_PLACE_V2 = """\
[place] 사진 속 장소가 목표 장소인가?
목표 장소 이름은 확인할 대상일 뿐 정답이 아니다. 근처에 비슷하게 생긴 다른 장소가 있을 수 있으니,
이름에 맞추지 말고 사진에 보이는 모습만으로 판단한다.
- "yes": 목표 장소만의 특징(건물·구조물의 모양, 재질, 주변 지형)이 사진에 보여 목표 장소라고 알아볼 수 있다.
- "no": 사진 속 장소의 모습이 목표 장소의 특징과 맞지 않는다.
- "unknown": 사진에 장소를 구별할 만한 특징이 보이지 않는다.
"""
assert QUESTION_SYSTEM_PROMPT.count(_PLACE_V1) == 1
QUESTION_SYSTEM_PROMPT_V2 = QUESTION_SYSTEM_PROMPT.replace(_PLACE_V1, _PLACE_V2)


class QuestionAnswer(BaseModel):
    # 네 칸 모두 필수(값은 null 가능)다. 응답 스키마로 쓰면 서버가 칸을 빼먹지 못하게 막는다
    judgeable: Literal["yes", "no"]
    target: Literal["enough", "short", "none"] | None
    condition: Literal["yes", "no", "na"] | None
    place: Literal["yes", "no", "unknown", "na"] | None

    @model_validator(mode="after")
    def _answered_what_can_be_asked(self) -> "QuestionAnswer":
        # 물을 수 있는 질문이 null이면 rule_grade가 아무 조건에도 안 걸려 "성공"이 된다. 형식 오류로 막는다
        if self.judgeable == "yes" and (self.target is None or self.place is None):
            raise ValueError("판단 가능인데 대상·장소 답이 비어 있음")
        if self.target in ("enough", "short") and self.condition is None:
            raise ValueError("대상이 있는데 조건 답이 비어 있음")
        return self


def rule_grade(a: QuestionAnswer) -> Grade:
    if a.judgeable == "no" or a.place == "no" or a.target == "none":
        return "fail"
    # 사진만으로 목표 장소인지 알 수 없으면 통과시키지 않되, 다시 찍을 기회는 준다 (모름을 성공으로 두면 닮은 장소가 통과)
    if a.target == "short" or a.condition == "no" or a.place == "unknown":
        return "retry"
    return "success"


def normalize(label: dict) -> QuestionAnswer:
    # 표시 화면의 q1~q4를 학습 형식으로 바꾼다. 물을 수 없는 질문은 null로 맞춘다
    # (1차 파일럿 화면은 대상이 없어도 Q3를 물어서 답이 갈렸다)
    if label["q1"] == "no":
        return QuestionAnswer(judgeable="no", target=None, condition=None, place=None)
    if label.get("q2") is None or label.get("q4") is None:
        # 그대로 두면 rule_grade가 아무 조건에도 안 걸려 "성공"이 된다. 틀린 정답이 조용히 들어가지 않게 멈춘다
        raise ValueError(f"판단 가능인데 Q2·Q4 답이 비어 있음: {label.get('mission_id')}")
    return QuestionAnswer(
        judgeable="yes", target=label["q2"], condition=None if label["q2"] == "none" else label["q3"], place=label["q4"],
    )


def user_text(target_place: str, mission: str) -> str:
    # 서비스 score_photo·측정 스크립트(baseline_verify.user_text)와 같은 형식
    return f"목표 장소: {target_place}\n미션 내용: {mission}"


def to_example(mission: dict, label: dict, image_path: str) -> dict:
    # LLaMA-Factory sharegpt 형식 한 건. 모델이 받는 장소는 미션의 목표 장소다 (다른 장소 미션이면 target_place)
    answer = normalize(label)
    return {
        "system": QUESTION_SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": user_text(mission.get("target_place") or mission["place"], mission["mission"]) + "\n<image>"},
            {"role": "assistant", "content": json.dumps(answer.model_dump(), ensure_ascii=False)},
        ],
        "images": [image_path],
    }
