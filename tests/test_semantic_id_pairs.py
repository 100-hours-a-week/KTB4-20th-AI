import pytest

from scripts.semantic_id import embed, eval_retrieval
from scripts.semantic_id.pairs import build_pairs, haversine_m, match_key

PLACE = {"google_place_id": "g1", "name": "경복궁", "lat": 37.5796, "lng": 126.9770}


def _visit(name="경복궁", x=126.9772, y=37.5797, type_cd="2", road="서울 종로구 사직로 161"):
    return {"name": name, "x": x, "y": y, "type_cd": type_cd, "road_addr": road, "lotno_addr": ""}


def test_match_key_ignores_spaces_branch_words_and_foreign_aliases():
    assert match_key("카페 아래헌 / Cafe Arae-Heon") == "카페아래헌"
    assert match_key("명동교자 본점") == match_key("명동교자")


def test_haversine_is_about_right():
    assert haversine_m(37.5796, 126.9770, 37.5796, 126.9770) == 0
    assert 100 < haversine_m(37.5796, 126.9770, 37.5796, 126.9782) < 120  # 경도 0.0012도 ≈ 106m


def test_pair_requires_close_coordinates_and_similar_name():
    near = build_pairs([PLACE], [_visit()])
    assert len(near) == 1 and near[0]["google_place_id"] == "g1" and near[0]["visits"] == 1
    assert build_pairs([PLACE], [_visit(x=127.5)]) == []  # 너무 멂
    assert build_pairs([PLACE], [_visit(name="전혀 다른 곳")]) == []  # 이름이 다름
    assert build_pairs([PLACE], [{**_visit(), "x": None}]) == []  # 좌표 없음


def test_same_text_and_place_are_merged_with_visit_count():
    pairs = build_pairs([PLACE], [_visit(), _visit()])
    assert len(pairs) == 1 and pairs[0]["visits"] == 2
    assert pairs[0]["log_text"] == "경복궁 | 서울 종로구 | 역사·문화"


def test_best_similarity_place_wins_among_neighbours():
    other = {"google_place_id": "g2", "name": "경복궁역", "lat": 37.5796, "lng": 126.9771}
    pairs = build_pairs([other, PLACE], [_visit()])
    assert [p["google_place_id"] for p in pairs] == ["g1"]


def test_build_texts_shares_log_texts_and_can_drop_categories():
    places = [{"google_place_id": "g1", "name": "경복궁", "address": "대한민국 서울특별시 종로구 사직로 161", "categories": ["HISTORY_CULTURE"]}]
    days = [{"visits": [_visit(), _visit(), _visit(name="북촌", type_cd="9")]}]
    items = embed.build_texts(places, days)
    assert [i["source"] for i in items] == ["db", "log", "log"]  # 같은 로그 텍스트는 한 번만
    assert items[0]["text"] == items[1]["text"] == "경복궁 | 서울 종로구 | 역사·문화"
    plain = embed.build_texts(places, days, with_category=False)
    assert plain[0]["text"] == "경복궁 | 서울 종로구"


def test_retrieval_metrics_rank_the_paired_place():
    np = pytest.importorskip("numpy")
    items = [
        {"source": "db", "key": "g1", "text": "a"}, {"source": "db", "key": "g2", "text": "b"},
        {"source": "db", "key": "g3", "text": "c"}, {"source": "log", "key": "q1", "text": "q1"},
        {"source": "log", "key": "q2", "text": "q2"},
    ]
    vectors = np.array([[1, 0], [0, 1], [0.7, 0.7], [0.9, 0.1], [0.6, 0.8]], dtype="float32")
    pairs = [{"log_text": "q1", "google_place_id": "g1", "visits": 3}, {"log_text": "q2", "google_place_id": "g1", "visits": 1}]
    m = eval_retrieval.retrieval_metrics(vectors, items, pairs)
    assert m["pairs"] == 2 and m["db_size"] == 3
    assert m["top1"] == 0.5  # q1은 g1이 1위, q2는 g2나 g3가 위
    assert m["top5"] == 1.0
    assert m["top1_visit_weighted"] == 0.75
