import pathlib
import unicodedata
import zipfile

import pytest

from scripts.semantic_id import loader

VISIT_HEADER = "VISIT_AREA_ID,TRAVEL_ID,VISIT_ORDER\n"
TRAVEL_HEADER = "TRAVEL_ID,TRAVELER_ID,TRAVEL_MISSION\n"


def _write_zip(path: pathlib.Path, files: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        for name, text in files.items():
            zf.writestr(name, text.encode("utf-8-sig"))


def _folder(root: pathlib.Path, code: str, region: str, split: str, nfd: bool = False) -> pathlib.Path:
    name = f"{code}.국내_여행로그_데이터({region})_{split}"
    if nfd:
        name = unicodedata.normalize("NFD", name)
    return root / name / "01-1.정식개방데이터" / split / "02.라벨링데이터"


def _make_all(root: pathlib.Path, skip: set[tuple[str, str]] = frozenset(), part0: bool = True) -> None:
    for code, region in loader.REGIONS.items():
        for split in loader.SPLITS:
            if (code, split) in skip:
                continue
            suffix = "_csv.zip.part0" if part0 else "_csv.zip"
            _write_zip(
                _folder(root, code, region, split) / (loader.ZIP_PREFIX[split] + suffix),
                {
                    "/tn_visit_area_info_방문지정보_A.csv": VISIT_HEADER + f"1,t{code}{split[0]},1\n",
                    "/tn_travel_여행_A.csv": TRAVEL_HEADER + f"t{code}{split[0]},x,1;2\n",
                    "/tn_poi_master_POIMaster.csv": "POI_ID\nP1\n",
                },
            )


def test_finds_all_eight_zips_including_part0(tmp_path):
    _make_all(tmp_path)
    zips = loader.find_label_zips(tmp_path)
    assert len(zips) == 8
    assert {(z.code, z.split) for z in zips} == {(c, s) for c in loader.REGIONS for s in loader.SPLITS}


def test_accepts_plain_zip_names(tmp_path):
    _make_all(tmp_path, part0=False)
    assert len(loader.find_label_zips(tmp_path)) == 8


def test_missing_zips_are_all_reported_before_reading(tmp_path):
    _make_all(tmp_path, skip={("278", "Training"), ("280", "Validation")})
    with pytest.raises(loader.DataRootError) as e:
        loader.find_label_zips(tmp_path)
    message = str(e.value)
    assert "278 동부권 Training" in message
    assert "280 제주도_및_도서지역 Validation" in message
    assert "277" not in message.split("\n", 1)[1]  # 있는 것은 문제로 적지 않음


def test_corrupt_zip_is_reported(tmp_path):
    _make_all(tmp_path)
    bad = _folder(tmp_path, "279", "서부권", "Training") / "TL_csv.zip.part0"
    bad.write_bytes(b"not a zip")
    with pytest.raises(loader.DataRootError, match="279 서부권 Training"):
        loader.find_label_zips(tmp_path)


def test_decomposed_hangul_folder_names_are_accepted(tmp_path):
    # macOS가 만드는 분해형(NFD) 폴더 이름도 찾아야 한다
    for code, region in loader.REGIONS.items():
        for split in loader.SPLITS:
            _write_zip(
                _folder(tmp_path, code, region, split, nfd=True) / (loader.ZIP_PREFIX[split] + "_csv.zip.part0"),
                {"/tn_visit_area_info_a.csv": VISIT_HEADER, "/tn_travel_a.csv": TRAVEL_HEADER},
            )
    assert len(loader.find_label_zips(tmp_path)) == 8


def test_missing_data_root(tmp_path):
    with pytest.raises(loader.DataRootError, match="데이터 폴더가 없음"):
        loader.find_label_zips(tmp_path / "nope")


def test_read_table_reads_only_the_prefixed_file_and_drops_repeated_header(tmp_path):
    path = tmp_path / "TL_csv.zip"
    _write_zip(path, {
        "/tn_visit_area_info_방문지정보_A.csv": VISIT_HEADER + "1,a,1\nVISIT_AREA_ID,TRAVEL_ID,VISIT_ORDER\n2,b,2\n",
        "/tn_poi_master_POIMaster.csv": "POI_ID\n" + "P\n" * 3,
    })
    rows = loader.read_table(path, "tn_visit_area_info")
    assert [r["TRAVEL_ID"] for r in rows] == ["a", "b"]  # BOM이 있어도 칸 이름이 맞고, 헤더 반복 줄은 빠짐


def test_travel_prefix_does_not_match_traveller_master(tmp_path):
    path = tmp_path / "TL_csv.zip"
    _write_zip(path, {
        "/tn_travel_여행_A.csv": TRAVEL_HEADER + "a,x,1\n",
        "/tn_traveller_master_여행객 Master_A.csv": "TRAVELER_ID\nx\n",
    })
    assert len(loader.read_table(path, "tn_travel_")) == 1


def test_prefix_must_match_exactly_one_file(tmp_path):
    path = tmp_path / "TL_csv.zip"
    _write_zip(path, {"/a.csv": "x\n1\n"})
    with pytest.raises(loader.DataRootError, match="0개"):
        loader.read_table(path, "tn_visit_area_info")


def test_load_rows_merges_training_and_validation(tmp_path):
    _make_all(tmp_path)
    zips = loader.find_label_zips(tmp_path)
    rows = loader.load_rows(zips, "tn_visit_area_info")
    assert len(rows) == 8
    assert len({r["TRAVEL_ID"] for r in rows}) == 8


def test_resolve_data_root_prefers_argument_then_env(tmp_path):
    arg = tmp_path / "arg"
    assert loader.resolve_data_root(arg, {"AIHUB_ROOT": "/x"}) == arg
    assert loader.resolve_data_root(None, {"AIHUB_ROOT": "/x"}) == pathlib.Path("/x")
    assert loader.resolve_data_root(None, {}) is None


def test_zip_from_path_fills_region_and_split_from_folder(tmp_path):
    _make_all(tmp_path)
    path = _folder(tmp_path, "280", "제주도_및_도서지역", "Training") / "TL_csv.zip.part0"
    z = loader.zip_from_path(path)
    assert (z.code, z.region, z.split) == ("280", "제주도_및_도서지역", "Training")


def test_main_prints_counts_and_checks(tmp_path, capsys):
    _make_all(tmp_path)
    assert loader.main(["--data-root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "라벨링 zip 8개" in out
    assert "합계: 방문 8건" in out
    assert "불일치" in out  # 합성 데이터라 실제 데이터의 기대 건수와 다르다


def test_main_returns_error_when_zips_are_missing(tmp_path, capsys):
    _make_all(tmp_path, skip={("277", "Training")})
    assert loader.main(["--data-root", str(tmp_path)]) == 1
    assert "277 수도권 Training" in capsys.readouterr().err
