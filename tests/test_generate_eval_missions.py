from scripts.generate_eval_missions import make_batches, to_request


def _row(photo, region, category="FOOD", place="식당"):
    return {"photo": photo, "region": region, "category": category, "place_name": place}


def test_batches_keep_one_region_and_cover_every_photo():
    # 실제 동선처럼 한 묶음에는 같은 지역 장소만 들어가고, 모든 사진이 한 번씩 들어간다
    rows = [_row(f"s{i}.jpg", "서울") for i in range(7)] + [_row(f"j{i}.jpg", "제주") for i in range(3)]
    batches = make_batches(rows, batch_size=5, seed=1)
    assert [len(b) for b in batches] == [5, 2, 3]
    assert all(len({r["region"] for r in b}) == 1 for b in batches)
    assert sorted(r["photo"] for b in batches for r in b) == sorted(r["photo"] for r in rows)


def test_batches_are_deterministic():
    # 다시 실행해도 같은 묶음이어야 이미 만든 묶음을 건너뛸 수 있다
    rows = [_row(f"p{i}.jpg", "부산") for i in range(12)]
    first = [[r["photo"] for r in b] for b in make_batches(rows, batch_size=5, seed=3)]
    second = [[r["photo"] for r in b] for b in make_batches(list(reversed(rows)), batch_size=5, seed=3)]
    assert first == second


def test_request_uses_photo_as_place_id_and_category_as_preference():
    request = to_request([_row("a.jpg", "경주", "ACTIVITY", "경주엑스포대공원")], "eval_b_000")
    place = request.places[0]
    assert request.itinerary_id == "eval_b_000"
    assert place.id == "a.jpg"
    assert place.displayName.text == "경주엑스포대공원"
    assert place.matched_preferences == ["ACTIVITY"]
