from scripts.write_other_place_missions import pick_other_place


def _place(name, lat, lon, region="경주"):
    return {"place_name": name, "region": region, "place_coords": {"latitude": lat, "longitude": lon}}


PLACES = [
    _place("불국사", 35.790, 129.332),
    _place("석굴암", 35.795, 129.349),  # 불국사에서 약 1.6km
    _place("대릉원", 35.838, 129.212),  # 약 12km
    _place("감천문화마을", 35.097, 129.010, region="부산"),
]


def test_other_place_is_nearby_and_different():
    # 서비스는 사진 좌표가 5km 넘게 떨어지면 AI 전에 거른다. 그래서 5km 안의 다른 장소만 현실적인 "다른 장소" 사례다
    photo = {"photo": "p.jpg", **_place("불국사", 35.790, 129.332)}
    assert pick_other_place(photo, PLACES, seed=1)["place_name"] == "석굴암"


def test_no_other_place_within_range_returns_none():
    photo = {"photo": "p.jpg", **_place("대릉원", 35.838, 129.212)}
    assert pick_other_place(photo, PLACES, seed=1) is None
