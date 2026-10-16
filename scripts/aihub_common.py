"""AI Hub 국내 여행로그 데이터를 읽는 공용 도구.

포토미션 평가셋·학습 데이터 스크립트(build_eval_category, build_train_pilot)와
trips 추천 모델 스크립트(measure_travel_sequences, semantic_id)가 함께 쓴다.
여기를 바꾸면 양쪽 결과가 함께 바뀌므로, 고칠 때는 위 스크립트의 테스트를 모두 돌린다.
"""

import csv
import io
import pathlib
import zipfile

# AI Hub 데이터셋 폴더(번호_권역)마다 라벨 CSV zip과 사진 zip 위치. 받은 사진 zip만 쓴다
DATASETS = {
    "277_수도권": {
        "labels": "277_수도권/277.국내 여행로그 데이터(수도권)/01-1.정식개방데이터/Training/02.라벨링데이터/TL_csv.zip",
        "photos": ["277_수도권/277.국내 여행로그 데이터(수도권)-사진/01-1.정식개방데이터/Training/01.원천데이터/TS_photo_4.zip"],
    },
    "278_동부권": {
        "labels": "278_동부권_추가/278.국내 여행로그 데이터(동부권)/01-1.정식개방데이터/Training/02.라벨링데이터/TL_csv.zip",
        "photos": [
            "278_동부권/Training/278.국내 여행로그 데이터(동부권)/01-1.정식개방데이터/Training/01.원천데이터/TS_photo_2.zip",
            "278_동부권_추가/278.국내 여행로그 데이터(동부권)/01-1.정식개방데이터/Training/01.원천데이터/TS_photo_5.zip",
        ],
    },
    "279_서부권": {
        "labels": "279_서부권/279.국내 여행로그 데이터(서부권)/01-1.정식개방데이터/Training/02.라벨링데이터/TL_csv.zip",
        "photos": [
            "279_서부권/279.국내 여행로그 데이터(서부권)/01-1.정식개방데이터/Training/01.원천데이터/TS_photo_4.zip",
            "279_서부권/279.국내 여행로그 데이터(서부권)/01-1.정식개방데이터/Training/01.원천데이터/TS_photo_5.zip",
        ],
    },
    "280_제주": {
        "labels": "280_제주/280.국내 여행로그 데이터(제주도 및 도서지역)/01-1.정식개방데이터/Training/02.라벨링데이터/TL_csv.zip",
        # 다운로드 중 연결이 끊겨 원본 zip이 손상돼, 받아 둔 앞부분(약 88%)에서 CRC가 맞는 사진만 모아 다시 만든 zip
        "photos": ["280_제주/280.국내 여행로그 데이터(제주도 및 도서지역)/01-1.정식개방데이터/Training/01.원천데이터/TS_photo_7_recovered.zip"],
    },
}


def region_of(address: str) -> str | None:
    # 서비스 지역 5곳만 남긴다. 주소 앞부분으로 지역 판정 (제주는 서귀포 포함)
    if address.startswith("서울"):
        return "서울"
    if address.startswith("부산"):
        return "부산"
    if address.startswith("제주"):
        return "제주"
    head = address[:12]
    if "경주" in head:
        return "경주"
    if "전주" in head:
        return "전주"
    return None

# zip 안의 CSV 읽기
def read_csv_in_zip(zip_path: pathlib.Path, prefix: str) -> list[dict]:
    # AI Hub zip 안의 파일명은 한글이 깨져 있어 앞부분 영문 이름으로 찾는다
    with zipfile.ZipFile(zip_path) as zf:
        name = next((n for n in zf.namelist() if n.strip("/").startswith(prefix)), None)
        if name is None:
            raise SystemExit(f"{zip_path} 안에서 {prefix} CSV를 찾을 수 없음")
        text = zf.read(name).decode("utf-8-sig")
    # 일부 CSV는 헤더 줄이 데이터 중간에 한 번 더 들어 있어 걸러낸다
    return [r for r in csv.DictReader(io.StringIO(text)) if r.get("TRAVEL_ID") != "TRAVEL_ID"]
