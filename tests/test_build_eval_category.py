from collections import Counter

from scripts.build_eval_category import CATEGORIES, REGIONS, region_of, sample


def test_region_of_service_regions():
    assert region_of("서울특별시 종로구 사직로 161") == "서울"
    assert region_of("부산광역시 해운대구 우동") == "부산"
    assert region_of("제주특별자치도 서귀포시 칠십리로") == "제주"
    assert region_of("경상북도 경주시 첨성로 140-25") == "경주"
    assert region_of("경북 경주시 황남동") == "경주"
    assert region_of("전라북도 전주시 완산구 기린대로") == "전주"
    assert region_of("전북특별자치도 전주시 완산구") == "전주"


def test_region_of_other_regions():
    assert region_of("경기도 수원시 팔달구") is None
    assert region_of("전라북도 완주군 소양면") is None
    assert region_of("") is None


def _photo(name, travel, category="FOOD", region="서울"):
    return {"photo": name, "travel_id": travel, "category": category, "region": region}


def test_sample_one_photo_per_travel_in_a_cell():
    # 한 여행에서 사진 5장이 있어도 한 칸에는 1장만 들어간다
    candidates = [_photo(f"t1_{i}.jpg", "t1") for i in range(5)] + [_photo("t2_0.jpg", "t2")]
    picked, shortage = sample(candidates, per_cell=3, seed=1, eval_travels=set())
    in_cell = [p for p in picked if (p["category"], p["region"]) == ("FOOD", "서울")]
    assert sorted(p["travel_id"] for p in in_cell) == ["t1", "t2"]
    assert shortage[("FOOD", "서울")] == 2


def test_sample_prefers_travels_already_used_for_eval():
    # 평가에 이미 쓴 여행이 있으면 그 여행부터 고른다 (학습에서 새로 빠지는 여행을 줄이기 위해)
    candidates = [_photo(f"new{i}.jpg", f"new{i}") for i in range(10)] + [_photo("a.jpg", "evalA")]
    picked, _ = sample(candidates, per_cell=1, seed=1, eval_travels={"evalA"})
    assert [p["travel_id"] for p in picked] == ["evalA"]


def test_sample_reports_empty_cells_as_shortage():
    # 후보가 0장인 칸도 부족으로 나와야 한다 (zip이 빠져 지역 하나가 통째로 없을 때)
    candidates = [_photo("x.jpg", "t1", category="FOOD", region="서울")]
    picked, shortage = sample(candidates, per_cell=1, seed=1, eval_travels=set())
    assert len(picked) == 1
    assert len(shortage) == len(CATEGORIES) * len(REGIONS) - 1
    assert shortage[("FOOD", "제주")] == 0


def test_sample_is_deterministic():
    candidates = [_photo(f"p{i}.jpg", f"t{i}", category=c, region=r)
                  for i, (c, r) in enumerate((c, r) for c in CATEGORIES for r in REGIONS for _ in range(3))]
    first, _ = sample(candidates, per_cell=2, seed=7, eval_travels=set())
    second, _ = sample(candidates, per_cell=2, seed=7, eval_travels=set())
    assert [p["photo"] for p in first] == [p["photo"] for p in second]
    assert set(Counter((p["category"], p["region"]) for p in first).values()) == {2}
