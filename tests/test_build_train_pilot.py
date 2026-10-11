from collections import Counter

from scripts.build_train_pilot import ROLE_ORDER, pick


def _cands():
    # 카테고리 2개 × 지역 5곳 × 여행 6개, 여행마다 사진 2장
    rows = []
    for cat in ("HISTORY_CULTURE", "FOOD"):
        for region in ("경주", "부산", "서울", "전주", "제주"):
            for t in range(6):
                for k in range(2):
                    rows.append({"photo": f"{cat}{region}{t}_{k}.jpg", "travel_id": f"{cat}{region}{t}",
                                 "category": cat, "region": region})
    return rows


def test_roles_and_counts_per_category():
    picked = pick(_cands(), categories=("HISTORY_CULTURE", "FOOD"), excluded_travels=set(), seed=1)
    for cat in ("HISTORY_CULTURE", "FOOD"):
        roles = Counter(p["role"] for p in picked if p["category"] == cat)
        assert roles == Counter(ROLE_ORDER)


def test_one_photo_per_travel_and_regions_spread():
    picked = pick(_cands(), categories=("HISTORY_CULTURE", "FOOD"), excluded_travels=set(), seed=1)
    assert len({p["travel_id"] for p in picked}) == len(picked)  # 같은 여행 사진은 비슷해서 여행당 1장
    regions = Counter(p["region"] for p in picked if p["category"] == "FOOD")
    assert max(regions.values()) - min(regions.values()) <= 1  # 지역을 돌아가며 뽑는다


def test_excluded_travels_and_photos_are_skipped():
    cands = _cands()
    banned_travel = cands[0]["travel_id"]
    banned_photo = cands[2]["photo"]
    picked = pick(cands, categories=("HISTORY_CULTURE",), excluded_travels={banned_travel}, seed=1,
                  is_excluded_photo=lambda c: c["photo"] == banned_photo)
    assert all(p["travel_id"] != banned_travel and p["photo"] != banned_photo for p in picked)


def test_same_seed_same_pick():
    a = pick(_cands(), categories=("FOOD",), excluded_travels=set(), seed=3)
    b = pick(_cands(), categories=("FOOD",), excluded_travels=set(), seed=3)
    assert a == b
