import random
from collections import Counter

from scripts.build_train_550 import (
    ROLE_SEQ,
    Usage,
    blocked_place,
    blocked_travels,
    lookalike_missions,
    pick_general,
    pick_lookalike,
    pilot_keep,
    reserve_travels,
)


def _cands(n_travels=8, per_travel=3, categories=("FOOD",)):
    # 카테고리 × 지역 5곳 × 여행 n개. 여행마다 방문지 per_travel곳, 방문지마다 사진 2장
    rows = []
    for cat in categories:
        for region in ("경주", "부산", "서울", "전주", "제주"):
            for t in range(n_travels):
                for v in range(per_travel):
                    for k in range(2):
                        rows.append({"photo": f"{cat}{region}{t}_{v}_{k}.jpg", "travel_id": f"{region}{t}",
                                     "visit_area_id": f"v{v}", "place_name": f"장소{v}",
                                     "category": cat, "region": region})
    return rows


def test_blocked_place_catches_diagnostic_places_but_not_woljeonggyo():
    # 종묘·김녕·월정리는 진단용이라 막는다. "월정"으로 찾으면 기존 짝인 월정교까지 빠지므로 "월정리"로만 찾는다
    for name in ("종묘", "종묘-창경궁 산책로", "김녕해수욕장", "월정리해수욕장", "월정리 바다"):
        assert blocked_place(name)
    for name in ("월정교", "월정사", "경복궁", "동궁과월지"):
        assert not blocked_place(name)


def test_blocked_travels_use_every_visit_not_only_photo_candidates():
    # 사진 후보에 없는 방문지라도 종묘를 다녀온 여행이면 여행째 뺀다
    visits = [("t1", "종묘"), ("t1", "경복궁"), ("t2", "경복궁"), ("t3", "김녕해수욕장"), ("t4", "월정교")]
    assert blocked_travels(visits, eval_travels={"t9"}, diag_travels={"t2"}) == {"t1", "t2", "t3", "t9"}


def test_excluded_travel_drops_every_photo_of_that_travel():
    cands = _cands()
    banned = cands[0]["travel_id"]
    picked, _ = pick_general(cands, ("FOOD",), Usage(), excluded={banned}, reserved=set(), seed=1)
    assert all(p["travel_id"] != banned for p in picked)


def test_excluded_photo_is_skipped():
    cands = _cands()
    banned = cands[0]["photo"]
    picked, _ = pick_general(cands, ("FOOD",), Usage(), excluded=set(), reserved=set(), seed=1,
                             is_excluded_photo=lambda c: c["photo"] == banned)
    assert banned not in {p["photo"] for p in picked}


def test_caps_count_every_source_together():
    # 파일럿·닮은 장소에서 이미 쓴 여행·방문지도 상한에 들어간다
    cands = _cands(n_travels=20)
    usage = Usage()
    usage.add({"photo": "pilot.jpg", "travel_id": "서울0", "visit_area_id": "v0"})
    usage.add({"photo": "look1.jpg", "travel_id": "서울0", "visit_area_id": "v1"})
    picked, _ = pick_general(cands, ("FOOD",), usage, excluded=set(), reserved=set(), seed=1)
    seoul0 = [p for p in picked if p["travel_id"] == "서울0"]
    assert len(seoul0) <= 1 and all(p["visit_area_id"] == "v2" for p in seoul0)
    per_travel = Counter(p["travel_id"] for p in usage.rows)
    assert max(per_travel.values()) <= 3
    assert len({(p["travel_id"], p["visit_area_id"]) for p in usage.rows}) == len(usage.rows)


def test_spreads_travels_before_reusing_one():
    # 여행당 3장은 상한일 뿐이다. 새 여행이 남아 있으면 먼저 쓴다
    cands = _cands(n_travels=20)
    picked, _ = pick_general(cands, ("FOOD",), Usage(), excluded=set(), reserved=set(), seed=1)
    assert len(picked) == len(ROLE_SEQ)
    assert len({p["travel_id"] for p in picked}) == len(picked)


def test_shortage_is_reported_not_filled_from_other_category():
    cands = _cands(n_travels=2, categories=("FOOD", "ACTIVITY"))
    cands = [c for c in cands if c["category"] == "ACTIVITY" or c["region"] == "경주"]
    picked, short = pick_general(cands, ("FOOD", "ACTIVITY"), Usage(), excluded=set(), reserved=set(), seed=1)
    assert all(p["region"] == "경주" for p in picked if p["category"] == "FOOD")
    assert short["FOOD"] == sum(p["category"] == "FOOD" for p in picked) < len(ROLE_SEQ)


def test_same_seed_same_pick_even_if_input_order_changes():
    cands = _cands(n_travels=20)
    shuffled = cands[:]
    random.Random(0).shuffle(shuffled)
    a, _ = pick_general(cands, ("FOOD",), Usage(), excluded=set(), reserved=set(), seed=3)
    b, _ = pick_general(shuffled, ("FOOD",), Usage(), excluded=set(), reserved=set(), seed=3)
    assert a == b


def test_reserved_travels_are_never_picked_and_exclude_pilot_travels():
    travels = {f"t{i}" for i in range(50)}
    reserved = reserve_travels(travels, keep={"t0", "t1"}, ratio=0.2, seed=1)
    assert len(reserved) == round(48 * 0.2) and not reserved & {"t0", "t1"}
    cands = [{"photo": f"{t}.jpg", "travel_id": t, "visit_area_id": "v", "place_name": "p", "category": "FOOD",
              "region": "서울"} for t in sorted(travels)]
    picked, _ = pick_general(cands, ("FOOD",), Usage(), excluded=set(), reserved=reserved, seed=1)
    assert not {p["travel_id"] for p in picked} & reserved


def test_lookalike_wrong_target_never_jongmyo_and_is_the_pair():
    for place in ("경복궁", "창덕궁", "창경궁", "덕수궁"):
        missions = lookalike_missions({"photo": f"{place}.jpg", "place": place})
        own, wrong = missions
        assert own["target_place"] == place and own["mission_id"] == f"{place}.jpg#own"
        assert wrong["target_place"] in {"경복궁", "창덕궁", "창경궁", "덕수궁"} - {place}
        assert not blocked_place(wrong["target_place"])
    assert lookalike_missions({"photo": "a.jpg", "place": "월정교"})[1]["target_place"] == "동궁과월지"
    assert lookalike_missions({"photo": "b.jpg", "place": "정방폭포"})[1]["target_place"] == "천지연폭포"
    assert lookalike_missions({"photo": "a.jpg", "place": "월정교"})[1]["mission"] == "동궁과월지가 잘 보이게 사진 찍기"


def test_pick_lookalike_respects_caps_and_quota_per_place():
    items = [{"photo": f"{p}{t}_{k}.jpg", "place": p, "travel_id": f"{p}{t}", "visit_area_id": "v", "label": "clear"}
             for p in ("천지연폭포", "정방폭포") for t in range(15) for k in range(2)]
    usage = Usage()
    picked = pick_lookalike(items, {"천지연폭포": 10, "정방폭포": 10}, usage, excluded={"천지연폭포0"}, seed=1)
    assert Counter(p["place"] for p in picked) == {"천지연폭포": 10, "정방폭포": 10}
    assert "천지연폭포0" not in {p["travel_id"] for p in picked}
    assert len({(p["travel_id"], p["visit_area_id"]) for p in picked}) == 20  # 같은 여행·방문지는 1장


def test_pilot_keep_drops_unclear_other_place_and_blocked():
    missions = {
        "a#pw": {"mission_id": "a#pw", "photo": "a", "travel_id": "t1", "place": "불국사", "source": "photo_written"},
        "b#gen": {"mission_id": "b#gen", "photo": "b", "travel_id": "t2", "place": "해동용궁사", "source": "service_generate"},
        "c#other": {"mission_id": "c#other", "photo": "c", "travel_id": "t3", "place": "카페", "target_place": "종묘",
                    "source": "other_place"},
        "d#pw": {"mission_id": "d#pw", "photo": "d", "travel_id": "diag", "place": "추억의달동네", "source": "photo_written"},
        "e#gen": {"mission_id": "e#gen", "photo": "e", "travel_id": "t5", "place": "카페", "target_place": "종묘",
                  "source": "service_generate"},
    }
    labels = [{"mission_id": m, "final_grade": g} for m, g in
              (("a#pw", "success"), ("b#gen", "unclear"), ("c#other", "fail"), ("d#pw", "fail"), ("e#gen", "fail"))]
    kept = pilot_keep(missions, labels, excluded={"diag"})
    assert [m["mission_id"] for m in kept] == ["a#pw"]


def test_second_photo_in_same_visit_only_when_short_and_marked():
    # 닮은 장소만 같은 방문지 2장까지. 새 방문지가 남아 있으면 그쪽을 먼저 쓰고, 두 번째 사진은 표시해 둔다
    items = [{"photo": f"g{t}_{k}.jpg", "place": "경복궁", "travel_id": f"t{t}", "visit_area_id": "v", "label": "clear"}
             for t in range(3) for k in range(3)]
    enough = pick_lookalike(items, {"경복궁": 3}, Usage(), excluded=set(), seed=1, visit_cap=2)
    assert len({p["travel_id"] for p in enough}) == 3 and not any(p.get("second_in_visit") for p in enough)
    short = pick_lookalike(items, {"경복궁": 9}, Usage(), excluded=set(), seed=1, visit_cap=2)
    assert len(short) == 6 and sum(bool(p.get("second_in_visit")) for p in short) == 3
    assert len(pick_lookalike(items, {"경복궁": 9}, Usage(), excluded=set(), seed=1)) == 3  # 기본은 방문지당 1장


def test_spare_list_never_overlaps_main_pick():
    cands = _cands(n_travels=30)
    usage = Usage()
    main, _ = pick_general(cands, ("FOOD",), usage, excluded=set(), reserved=set(), seed=1)
    spare, _ = pick_general(cands, ("FOOD",), usage, excluded=set(), reserved=set(), seed=1,
                            role_seq=("photo_written",) * 5, group="general_spare")
    assert len(spare) == 5 and not {p["photo"] for p in main} & {p["photo"] for p in spare}
    assert all(p["spare"] for p in spare)
