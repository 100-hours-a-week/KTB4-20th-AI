from scripts.split_eval import EvalPhoto, split_travels, summarize


def _photos():
    # 여행 40개 × 사진 3장. 여행 절반은 A·B 양쪽에 사진이 있다 (실제로 A 여행 대부분이 B에도 있음)
    photos = []
    for t in range(40):
        photos.append(EvalPhoto(photo=f"a{t}.jpg", travel_id=f"t{t}", stratum="A"))
        photos.append(EvalPhoto(photo=f"b{t}.jpg", travel_id=f"t{t}", stratum="B:음식" if t % 2 else "B:액티비티"))
        if t < 20:
            photos.append(EvalPhoto(photo=f"c{t}.jpg", travel_id=f"t{t}", stratum="B:자연·힐링"))
    return photos


def test_one_travel_never_lands_on_both_sides():
    photos = _photos()
    sides = split_travels(photos, seed=1)
    assert set(sides.values()) == {"dev", "final"}
    for p in photos:
        assert p.travel_id in sides  # 모든 여행이 어느 한쪽에 들어간다


def test_each_stratum_is_close_to_dev_ratio():
    photos = _photos()
    table = summarize(photos, split_travels(photos, seed=1))
    for stratum, (dev, final) in table.items():
        assert abs(dev / (dev + final) - 0.6) <= 0.1, stratum


def test_same_seed_gives_same_split():
    photos = _photos()
    assert split_travels(photos, seed=7) == split_travels(photos, seed=7)
