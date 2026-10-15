import collections
import json

import pytest

from scripts.semantic_id import split

REGIONS = ["서울", "부산", "경주", "전주", "제주"]


def _days(n_travelers=600, seed_offset=0):
    # 여행자마다 하루 1~3개, 지역은 여행자마다 하나(제주가 많음)
    days = []
    for i in range(n_travelers):
        region = REGIONS[(i * 7 + seed_offset) % 5] if i % 3 else "제주"
        for k in range(1 + i % 3):
            days.append({"travel_id": f"t{i}", "traveler_id": f"u{i}", "date": f"2022-10-0{k + 1}", "region": region, "visits": []})
    return days


def test_every_traveler_lands_in_exactly_one_part():
    days = _days()
    parts = split.split_travelers(split.traveler_regions_from_days(days))
    assert set(parts) == {d["traveler_id"] for d in days}
    assert set(parts.values()) == set(split.PARTS)


def test_same_seed_gives_same_split_and_other_seed_differs():
    regions = split.traveler_regions_from_days(_days())
    assert split.split_travelers(regions, seed=1) == split.split_travelers(regions, seed=1)
    assert split.split_travelers(regions, seed=1) != split.split_travelers(regions, seed=2)


def test_ratios_and_region_balance_are_close_to_target():
    days = _days(n_travelers=900)
    parts = split.split_travelers(split.traveler_regions_from_days(days))
    summary = split.summarize(days, parts)
    total = sum(summary["days"][p]["합계"] for p in split.PARTS)
    for p, ratio in split.RATIOS.items():
        assert abs(summary["days"][p]["합계"] / total - ratio) < 0.03
    for region in REGIONS:
        region_total = sum(summary["days"][p][region] for p in split.PARTS)
        for p, ratio in split.RATIOS.items():
            assert abs(summary["days"][p][region] / region_total - ratio) < 0.06, (region, p)


def test_summary_marks_regions_with_few_final_days_as_pending():
    days = _days(n_travelers=60)  # 최종용 하루가 지역마다 100일에 훨씬 못 미침
    parts = split.split_travelers(split.traveler_regions_from_days(days))
    assert set(split.summarize(days, parts)["regions_pending_verdict"]) == set(REGIONS)


def test_split_file_is_never_overwritten(tmp_path):
    days = _days(60)
    parts = split.split_travelers(split.traveler_regions_from_days(days))
    path = tmp_path / "split.json"
    split.write_split(path, parts, split.summarize(days, parts), split.SEED, split.RATIOS)
    with pytest.raises(FileExistsError, match="덮어쓰지 않음"):
        split.write_split(path, parts, split.summarize(days, parts), split.SEED, split.RATIOS)


def test_final_part_is_locked_unless_explicitly_opened(tmp_path):
    days = _days(60)
    parts = split.split_travelers(split.traveler_regions_from_days(days))
    path = tmp_path / "split.json"
    split.write_split(path, parts, split.summarize(days, parts), split.SEED, split.RATIOS)
    assert split.load_part(path, "train")  # train과 dev는 바로 읽힌다
    assert split.load_part(path, "dev")
    with pytest.raises(split.FinalLockedError):
        split.load_part(path, "final")
    assert split.load_part(path, "final", open_final=True)


def test_iter_days_yields_only_the_part_and_respects_the_lock(tmp_path):
    days = _days(60)
    days_path = tmp_path / "days.jsonl"
    days_path.write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in days) + "\n", encoding="utf-8")
    parts = split.split_travelers(split.traveler_regions_from_days(days))
    path = tmp_path / "split.json"
    split.write_split(path, parts, split.summarize(days, parts), split.SEED, split.RATIOS)
    got = list(split.iter_days(days_path, path, "dev"))
    assert got and all(parts[d["traveler_id"]] == "dev" for d in got)
    with pytest.raises(split.FinalLockedError):
        list(split.iter_days(days_path, path, "final"))


def test_main_writes_once_and_then_refuses(tmp_path, capsys):
    days = _days(60)
    days_path = tmp_path / "days.jsonl"
    days_path.write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in days) + "\n", encoding="utf-8")
    out = tmp_path / "split.json"
    assert split.main(["--days", str(days_path), "--out", str(out)]) == 0
    assert "판정 보류" in capsys.readouterr().out
    assert split.main(["--days", str(days_path), "--out", str(out)]) == 1
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert collections.Counter(saved["traveler_part"].values()).keys() <= set(split.PARTS)


def test_unknown_part_is_rejected(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"traveler_part": {}}), encoding="utf-8")
    with pytest.raises(ValueError):
        split.load_part(path, "test")
