"""verify 객관식 형식의 측정·학습용 도구 (개선 ② 실험, 학습 데이터).

지시문·답 스키마·등급 규칙은 서비스도 쓰므로 app/photomissions/question_format.py에 있고, 여기서는 그대로 다시 내보낸다.
여기에는 실험용 장소 질문 v2 지시문, 표시 화면 답 변환(normalize), 학습 예제 만들기(to_example)만 둔다.
"""

import json

from app.photomissions.question_format import (  # noqa: F401 — 스크립트·테스트가 이 경로로 가져다 쓴다
    QUESTION_SYSTEM_PROMPT,
    Grade,
    QuestionAnswer,
    rule_grade,
)

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
