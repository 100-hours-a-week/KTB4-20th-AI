import json
import math

import httpx
import pytest

from scripts import baseline_verify as bv
from scripts.place_probe import (
    IDENTIFY_SYSTEM_PROMPT,
    QUESTION_OBS_SYSTEM_PROMPT,
    IdentifyAnswer,
    QuestionAnswerObs,
    option_probs,
    question_obs_fields,
    summarize,
)
from scripts.question_format import QUESTION_SYSTEM_PROMPT, QuestionAnswer

pytestmark = pytest.mark.anyio

PLACE_OPTIONS = ['"yes"', '"no"', '"unknown"', '"na"', "null"]


# A. 관찰 먼저 — 관찰 칸 말고는 question(v1)과 같아야 효과를 따로 볼 수 있다

def test_obs_prompt_only_adds_observation_before_first_question():
    head, tail = QUESTION_OBS_SYSTEM_PROMPT.split("[judgeable]")
    v1_head, v1_tail = QUESTION_SYSTEM_PROMPT.split("[judgeable]")
    assert tail == v1_tail
    assert head.startswith(v1_head) and "[observation]" in head


def test_obs_answer_writes_observation_first_and_keeps_question_rules():
    assert next(iter(QuestionAnswerObs.model_json_schema()["properties"])) == "observation"
    with pytest.raises(ValueError):  # 판단 가능인데 장소 답이 비면 question과 똑같이 형식 오류
        QuestionAnswerObs(observation="다리", judgeable="yes", target="enough", condition="na", place=None)


def test_obs_fields_are_readable_by_question_counting():
    a = QuestionAnswerObs(observation="돌다리 위 누각", judgeable="yes", target="none", condition=None, place="no")
    fields = question_obs_fields(a)
    assert QuestionAnswer.model_validate(fields["answer"]).place == "no"
    assert fields["grade"] == "fail" and fields["observation"] == "돌다리 위 누각"


# D. 장소 알아보기 — 목표 장소를 알려 주지 않는다

def test_identify_does_not_tell_the_target():
    case = {"target": "동궁과월지", "mission": "동궁과월지가 잘 보이게 찍기"}
    text = bv.user_text(case, "identify")
    assert "동궁과월지" not in text and "목표" not in text
    assert bv.user_text(case) == "목표 장소: 동궁과월지\n미션 내용: 동궁과월지가 잘 보이게 찍기"  # 다른 형식은 그대로


def test_identify_choices_match_prompt():
    choices = IdentifyAnswer.model_json_schema()["properties"]["answer"]["enum"]
    assert all(f'"{c}"' in IDENTIFY_SYSTEM_PROMPT for c in choices)


# 닮은 장소 케이스 고르기 — 오답만 보면 "아니오"만 하는 모델도 좋아 보이므로 같은 사진의 맞는 장소도 넣는다

CASES = [
    {"photo": "w1", "place": "월정교", "kind": "positive", "target": "월정교"},
    {"photo": "w1", "place": "월정교", "kind": "negative", "target": "동궁과월지"},
    {"photo": "w1", "place": "월정교", "kind": "negative", "target": "첨성대"},
    {"photo": "c1", "place": "첨성대", "kind": "positive", "target": "첨성대"},
]


def test_lookalike_cases_with_and_without_positives():
    assert [c["target"] for c in bv.lookalike_cases(CASES)] == ["동궁과월지"]
    assert [(c["photo"], c["target"]) for c in bv.lookalike_cases(CASES, with_positives=True)] == [
        ("w1", "월정교"), ("w1", "동궁과월지"),
    ]


# E. 확률 — 장소 답 자리에서 선택지별 확률을 더한다

def _tok(text, alts=None):
    return {"token": text, "logprob": 0.0, "bytes": list(text.encode()),
            "top_logprobs": [{"token": t, "logprob": math.log(p), "bytes": list(t.encode())} for t, p in (alts or [])]}


def test_option_probs_when_quote_is_its_own_token():
    tokens = [_tok('{"judgeable":"yes","target":"none","condition":null,"place":'), _tok('"'),
              _tok("no", [("no", 0.6), ("yes", 0.3), ("na", 0.05), ("unknown", 0.04)]), _tok('"}')]
    probs = option_probs(tokens, "place", PLACE_OPTIONS)
    assert probs['"no"'] == pytest.approx(0.6) and probs['"yes"'] == pytest.approx(0.3)
    assert probs['"na"'] == pytest.approx(0.05) and probs['"unknown"'] == pytest.approx(0.04)


def test_option_probs_when_quote_is_glued_to_value():
    tokens = [_tok('{"place": '), _tok('"yes', [('"yes', 0.9), ('"no', 0.07), ('"n', 0.02), ("null", 0.01)]), _tok('"}')]
    probs = option_probs(tokens, "place", PLACE_OPTIONS)
    assert probs['"yes"'] == pytest.approx(0.9) and probs['"no"'] == pytest.approx(0.07)
    assert probs["other"] == pytest.approx(0.02)  # '"n'은 no·na 둘 다라 어느 쪽인지 모른다
    assert probs["null"] == pytest.approx(0.01)


def test_option_probs_counts_bytes_not_characters_after_korean_text():
    # 한글 한 글자가 토큰 두 개로 쪼개져도(바이트) 위치가 어긋나지 않아야 한다
    obs = "누각".encode()
    tokens = [_tok('{"observation":"'), {"token": "bytes:\\xeb", "logprob": 0.0, "bytes": list(obs[:2])},
              {"token": "bytes:rest", "logprob": 0.0, "bytes": list(obs[2:])}, _tok('","place":"'),
              _tok("unknown", [("unknown", 0.7), ("yes", 0.3)]), _tok('"}')]
    probs = option_probs(tokens, "place", PLACE_OPTIONS)
    assert probs['"unknown"'] == pytest.approx(0.7) and probs['"yes"'] == pytest.approx(0.3)


def test_option_probs_without_the_key_is_none():
    assert option_probs([_tok('{"judgeable":"no"}')], "place", PLACE_OPTIONS) is None


# judge_local — 요청에 확률을 달라고 하고, 기록에 확률·생각 글자 수를 남긴다

async def test_judge_local_asks_for_logprobs_and_records_them(monkeypatch):
    sent = []
    answer = '{"judgeable":"yes","target":"enough","condition":"na","place":"yes"}'
    tokens = [_tok('{"judgeable":"yes","target":"enough","condition":"na","place":"'),
              _tok("yes", [("yes", 0.8), ("no", 0.2)]), _tok('"}')]

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": answer, "reasoning_content": "생각" * 10},
                         "logprobs": {"content": tokens}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 30},
        })

    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(bv, "normalize_image", lambda b: b)
    case = {"target": "월정교", "mission": "월정교가 잘 보이게 찍기"}
    record = await bv.judge_local(case, b"jpeg", "http://pod/v1", "qwen", fmt="question", logprobs=True)

    assert sent[0]["logprobs"] is True and sent[0]["top_logprobs"] == 10
    assert record["place_probs"]['"yes"'] == pytest.approx(0.8)
    assert record["reasoning_chars"] == 20 and record["grade"] == "success"


async def test_judge_local_without_logprobs_keeps_request_as_before(monkeypatch):
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}], "usage": {}})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(bv, "normalize_image", lambda b: b)
    record = await bv.judge_local({"target": "월정교", "mission": "m"}, b"jpeg", "http://pod/v1", "qwen", fmt="question")
    assert "logprobs" not in sent[0] and "place_probs" not in record


# 요약

def _q(kind, place, grade, p_yes=None):
    r = {"kind": kind, "parsed": True, "model_sec": 1.0, "output_tokens": 20,
         "answer": {"judgeable": "yes", "target": "enough", "condition": "na", "place": place}, "grade": grade}
    if p_yes is not None:
        r["place_probs"] = {'"yes"': p_yes}
    return r


def test_summary_threshold_blocks_low_confidence_yes():
    records = [_q("negative", "yes", "success", 0.6), _q("negative", "no", "fail", 0.1),
               _q("positive", "yes", "success", 0.95), _q("positive", "yes", "success", 0.85)]
    text = "\n".join(summarize(records))
    assert "닮은 장소 통과(성공 판정) 1/2 (50.0%)" in text
    assert "기준 0.50: 닮은 장소 통과 1/2 (50.0%) · 맞는 장소 성공 2/2 (100.0%)" in text
    assert "기준 0.90: 닮은 장소 통과 0/2 (0.0%) · 맞는 장소 성공 1/2 (50.0%)" in text


def test_summary_identify_accuracy_per_place():
    records = [{"place": "월정교", "parsed": True, "model_sec": 1.0, "identify": "woljeonggyo"},
               {"place": "월정교", "parsed": True, "model_sec": 1.0, "identify": "donggung_wolji"},
               {"place": "동궁과월지", "parsed": True, "model_sec": 1.0, "identify": "unknown"}]
    text = "\n".join(summarize(records))
    assert "[알아보기] 월정교 사진 → 맞힘 1/2 (50.0%)" in text
    assert "[알아보기] 동궁과월지 사진 → 맞힘 0/1 (0.0%)" in text
