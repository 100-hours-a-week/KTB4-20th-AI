"""AI-Hub 국내 여행로그의 라벨링 CSV zip을 찾아 읽는다 (M1 T1-1).

권역(277 수도권, 278 동부권, 279 서부권, 280 제주도 및 도서지역) × Training·Validation, 모두 8개 zip이 있어야 한다.
Training과 Validation은 같은 여행의 방문을 나눈 것이라 합쳐야 여행이 온전해진다 (ADR-0005).
다운로드한 `.zip.part0`는 단일 조각이어서 zip으로 바로 읽힌다.

  # 레포 루트에서 실행. --data-root가 없으면 환경 변수 AIHUB_ROOT를 쓴다
  uv run python -m scripts.semantic_id.loader --data-root "$AIHUB_ROOT"

  # 일부만 읽을 때(재현·점검용). 8개 확인은 건너뛴다
  uv run python -m scripts.semantic_id.loader --label-zip <TL_csv.zip.part0> --label-zip <VL_csv.zip.part0>
"""

import argparse
import csv
import io
import os
import pathlib
import re
import sys
import unicodedata
import zipfile
from typing import NamedTuple

REGIONS = {"277": "수도권", "278": "동부권", "279": "서부권", "280": "제주도_및_도서지역"}
SPLITS = ("Training", "Validation")
ZIP_PREFIX = {"Training": "TL", "Validation": "VL"}  # TL_csv.zip, VL_csv.zip
ZIP_SUFFIXES = ("_csv.zip", "_csv.zip.part0")
FOLDER_RE = re.compile(r"^(?P<code>27[789]|280)\.국내_여행로그_데이터\((?P<region>[^)]+)\)_(?P<split>Training|Validation)$")
ENV_NAME = "AIHUB_ROOT"

# 2026-10-15에 내려받은 데이터의 건수. 다른 판이면 달라질 수 있어 점검 출력에만 쓴다
EXPECTED_VISITS = 184_719
EXPECTED_TRIPS_IN_VISITS = 16_000
EXPECTED_TRIPS_WITH_META = 14_400


class DataRootError(Exception):
    pass


class LabelZip(NamedTuple):
    code: str  # 권역 번호 (277~280). 직접 지정한 zip은 빈 문자열
    region: str
    split: str  # Training / Validation. 직접 지정한 zip은 빈 문자열
    path: pathlib.Path

    @property
    def label(self) -> str:
        return f"{self.code} {self.region} {self.split}".strip() or self.path.name


def _nfc(name: str) -> str:
    # macOS는 한글 파일 이름을 분해 형태(NFD)로 저장하는 경우가 있어 비교 전에 맞춘다
    return unicodedata.normalize("NFC", name)


def _find_zip(folder: pathlib.Path, split: str) -> pathlib.Path | None:
    # 폴더 안 어디에 있든 TL_csv.zip(.part0) / VL_csv.zip(.part0)를 찾는다
    wanted = {ZIP_PREFIX[split] + suffix for suffix in ZIP_SUFFIXES}
    for root, _dirs, files in os.walk(folder):
        for name in sorted(files):
            if _nfc(name) in wanted:
                return pathlib.Path(root) / name
    return None


def find_label_zips(data_root: pathlib.Path) -> list[LabelZip]:
    # 8개가 모두 있고 zip으로 열 수 있는지 읽기 전에 확인한다. 하나라도 문제면 문제를 모두 모아 알린다
    if not data_root.is_dir():
        raise DataRootError(f"데이터 폴더가 없음: {data_root}")
    found: dict[tuple[str, str], LabelZip] = {}
    for entry in sorted(data_root.iterdir()):
        m = FOLDER_RE.match(_nfc(entry.name))
        if not m or not entry.is_dir():
            continue
        path = _find_zip(entry, m["split"])
        if path is not None:
            found[(m["code"], m["split"])] = LabelZip(m["code"], REGIONS[m["code"]], m["split"], path)

    problems = []
    result = []
    for code, region in REGIONS.items():
        for split in SPLITS:
            item = found.get((code, split))
            if item is None:
                names = " 또는 ".join(ZIP_PREFIX[split] + s for s in ZIP_SUFFIXES)
                problems.append(f"{code} {region} {split}: 폴더 또는 {names} 없음")
            elif not zipfile.is_zipfile(item.path):
                problems.append(f"{code} {region} {split}: zip으로 열 수 없음 (없거나 손상됨) {item.path.name}")
            else:
                result.append(item)
    if problems:
        raise DataRootError("라벨링 zip 8개가 모두 필요함 (" + str(data_root) + ")\n  - " + "\n  - ".join(problems))
    return result


def zip_from_path(path: pathlib.Path) -> LabelZip:
    # 직접 지정한 zip. 폴더 이름에서 권역과 구분을 알 수 있으면 채운다
    if not zipfile.is_zipfile(path):
        raise DataRootError(f"zip으로 열 수 없음 (없거나 손상됨): {path}")
    for parent in path.parents:
        m = FOLDER_RE.match(_nfc(parent.name))
        if m:
            return LabelZip(m["code"], REGIONS[m["code"]], m["split"], path)
    return LabelZip("", "", "", path)


def read_table(zip_path: pathlib.Path, prefix: str) -> list[dict]:
    # zip 안의 파일 이름은 한글이 깨져 있어 앞부분 영문 이름(예: tn_visit_area_info)으로 찾는다
    # 같은 zip에 1.8GB짜리 POI 마스터가 들어 있어서, 찾은 파일 하나만 읽는다
    with zipfile.ZipFile(zip_path) as zf:
        matches = [n for n in zf.namelist() if n.strip("/").startswith(prefix)]
        if len(matches) != 1:
            raise DataRootError(f"{zip_path.name} 안에서 '{prefix}'로 시작하는 CSV가 {len(matches)}개 있음 (1개여야 함)")
        text = zf.read(matches[0]).decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    first = reader.fieldnames[0] if reader.fieldnames else None
    # 일부 CSV는 헤더 줄이 데이터 중간에 한 번 더 들어 있어 걸러낸다
    return [r for r in reader if first is None or r.get(first) != first]


def load_rows(zips: list[LabelZip], prefix: str) -> list[dict]:
    rows: list[dict] = []
    for z in zips:
        rows.extend(read_table(z.path, prefix))
    return rows


def resolve_data_root(arg: pathlib.Path | None, env: dict[str, str] | None = None) -> pathlib.Path | None:
    if arg is not None:
        return arg
    value = (env if env is not None else os.environ).get(ENV_NAME, "").strip()
    return pathlib.Path(value).expanduser() if value else None


def summarize(zips: list[LabelZip]) -> list[str]:
    lines = []
    visit_total = 0
    trips_in_visits: set[str] = set()
    trips_with_meta: set[str] = set()
    for z in zips:
        visits = read_table(z.path, "tn_visit_area_info")
        meta = read_table(z.path, "tn_travel_")
        visit_total += len(visits)
        trips_in_visits.update(r["TRAVEL_ID"] for r in visits)
        trips_with_meta.update(r["TRAVEL_ID"] for r in meta)
        lines.append(f"  {z.label}: 방문 {len(visits)}건, 여행 메타 {len(meta)}건")
    lines.append(f"합계: 방문 {visit_total}건, 방문 기록이 있는 여행 {len(trips_in_visits)}건, 여행 메타가 있는 여행 {len(trips_with_meta)}건")
    lines.append(f"  메타가 없는 여행 {len(trips_in_visits - trips_with_meta)}건 (취향 벡터를 만들 수 없어 제외됨)")
    checks = [
        ("방문", visit_total, EXPECTED_VISITS),
        ("방문 기록이 있는 여행", len(trips_in_visits), EXPECTED_TRIPS_IN_VISITS),
        ("여행 메타가 있는 여행", len(trips_with_meta), EXPECTED_TRIPS_WITH_META),
    ]
    for name, actual, expected in checks:
        status = "일치" if actual == expected else f"불일치 (기대 {expected})"
        lines.append(f"  점검 {name} {actual}건: {status}")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=pathlib.Path, help=f"AI-Hub 폴더 (없으면 환경 변수 {ENV_NAME})")
    ap.add_argument("--label-zip", type=pathlib.Path, action="append", help="라벨링 zip 직접 지정 (여러 번). 8개 확인을 건너뜀")
    args = ap.parse_args(argv)
    try:
        if args.label_zip:
            zips = [zip_from_path(p) for p in args.label_zip]
        else:
            root = resolve_data_root(args.data_root)
            if root is None:
                ap.error(f"--data-root 또는 환경 변수 {ENV_NAME}가 필요함")
            zips = find_label_zips(root)
    except DataRootError as e:
        print(e, file=sys.stderr)
        return 1
    print(f"라벨링 zip {len(zips)}개")
    for line in summarize(zips):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
