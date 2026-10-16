import math
import random

import pytest

from scripts.semantic_id import decode, eval_model, tokens, train_data


def test_place_tokens_use_separate_ranges_per_level():
    a = tokens.place_tokens((5, 5, 5, 0))
    assert len(set(a)) == 4  # 같은 숫자여도 단계마다 다른 토큰
    assert all(0 <= t < tokens.VOCAB_SIZE for t in a)
    for pos, token in enumerate(a):
        assert tokens.token_to_code_part(token)[0] == pos
    with pytest.raises(ValueError):
        tokens.place_tokens((256, 0, 0, 0))


def test_encode_day_masks_only_target_visits():
    pref = dict.fromkeys(tokens.CATEGORIES, 0) | {"ACTIVITY": 1}
    toks, mask = tokens.encode_day(pref, "제주", [(1, 2, 3, 0), (4, 5, 6, 1)], [True, False])
    assert len(toks) == tokens.HEADER_LEN + 8
    assert mask[:tokens.HEADER_LEN] == [0] * tokens.HEADER_LEN
    assert mask[tokens.HEADER_LEN:] == [1, 1, 1, 1, 0, 0, 0, 0]
    assert toks[0] == tokens.BOS


def _fake_next_logprobs(preferred_first_token):
    # 어휘 전체에 같은 확률을 주되 preferred_first_token 하나만 높게 준다
    def fn(prefixes):
        out = []
        for _ in prefixes:
            lp = [math.log(1e-6)] * tokens.VOCAB_SIZE
            lp[preferred_first_token] = math.log(0.9)
            out.append(lp)
        return out
    return fn


def test_beam_search_only_returns_candidates_and_is_deterministic():
    candidates = {"A": (1, 1, 1, 0), "B": (1, 2, 3, 0), "C": (9, 9, 9, 0)}
    fn = _fake_next_logprobs(tokens.place_tokens(candidates["B"])[0])
    first = decode.rank_candidates([tokens.BOS], candidates, fn, 3)
    assert set(first) <= set(candidates)
    assert first == decode.rank_candidates([tokens.BOS], candidates, fn, 3)
    assert first[0] in ("A", "B")  # 첫 토큰이 선호된 쪽이 앞선다
    assert decode.rank_candidates([tokens.BOS], decode.exclude(candidates, ["A", "B"]), fn, 3) == ["C"]
    assert decode.rank_candidates([tokens.BOS], {}, fn, 3) == []


def test_build_examples_drops_days_without_map_and_marks_far_visits():
    visit = {"name": "경복궁", "road_addr": "서울 종로구 사직로 161", "lotno_addr": ""}
    text = train_data.log_visit_text(visit)
    day = {"travel_id": "t", "traveler_id": "x", "region": "서울", "preference": dict.fromkeys(tokens.CATEGORIES, 1), "visits": [visit] * 8}
    codes = {"g1": [1, 2, 3, 0]}
    ex = train_data.build_examples([day], {text: ["g1", 0.5]}, codes, threshold=0.607)
    assert len(ex) == 1 and len(ex[0]["gids"]) == train_data.MAX_VISITS  # 앞 6곳까지만
    assert ex[0]["targets"] == [False] * 6  # 범위 밖 방문은 정답에서 뺀다
    assert train_data.build_examples([day], {}, codes) == []  # 지도에 없으면 하루를 버린다


def test_quality_filter_matches_service_rule():
    assert train_data.passes_quality_filter("OPERATIONAL", 3.1, 10)
    assert not train_data.passes_quality_filter("OPERATIONAL", 3.0, 10)
    assert not train_data.passes_quality_filter("OPERATIONAL", 4.5, 9)
    assert not train_data.passes_quality_filter("CLOSED_TEMPORARILY", 4.5, 100)


def test_rule_slots_are_food_at_2_and_5_and_follow_top_two_preferences():
    pref = {"HISTORY_CULTURE": 1, "NATURE_HEALING": 0, "ACTIVITY": 1, "CONVENIENCE_SHOPPING": 0}
    for seed in range(20):
        slots = eval_model.rule_slot_categories(pref, random.Random(seed))
        assert slots[1] == slots[4] == "FOOD" and slots[5] == "ACTIVITY"
        assert {slots[0], slots[3]} == {"HISTORY_CULTURE", "ACTIVITY"} and slots[0] == slots[2]


def test_rank_of_and_bootstrap_are_consistent():
    assert eval_model.rank_of("b", ["a", "b"], 10) == 2
    assert eval_model.rank_of("c", ["a", "b"], 10) is None
    records = [{"traveler": f"t{i}", "model": {10: 1}, "rule": [None] * 3, "popular": None} for i in range(30)]
    out = eval_model.bootstrap_diff(records, eval_model.metric("model", 10, 10), eval_model.metric("rule", 10))
    assert out["diff"] == 1.0 and out["lo"] == 1.0 and out["travelers"] == 30
