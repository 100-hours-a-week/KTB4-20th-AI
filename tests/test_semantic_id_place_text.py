import pytest

from scripts.semantic_id.place_text import (
    category_text,
    db_place_text,
    district_of,
    log_visit_text,
)


@pytest.mark.parametrize("address, expected", [
    ("대한민국 전북특별자치도 전주시 완산구 태조로 51", "전북 전주시 완산구"),
    ("대한민국 서울특별시 서초구 신반포로 30", "서울 서초구"),
    ("대한민국 제주특별자치도 제주시 화북일동 4474", "제주 제주시"),
    ("대한민국 경상북도 경주시 첨성로 169-5", "경북 경주시"),
    ("서울 종로구 사직로 161", "서울 종로구"),  # 로그의 줄인 시도 이름
    ("제주특별자치도 서귀포시 표선면 민속해안로 537", "제주 서귀포시"),
    ("경기 용인시 처인구 포곡읍 에버랜드로 199", "경기 용인시 처인구"),
    ("대한민국 부산광역시 기장군 기장읍", "부산 기장군"),
    ("세종특별자치시 한누리대로 2130", "세종"),
    ("대한민국 경주시 건천읍 건천리", "경북 경주시"),  # 시도가 빠진 주소
    ("대한민국 고양시 덕양구 행신동", "경기 고양시 덕양구"),
    ("대한민국 서울특별시 KR 서울특별시 금천구 가산디지털1로 25", "서울 금천구"),  # 시도가 반복된 주소
    ("327 번길 6, 시민대로 동안구 안양시 경기도 대한민국", "경기 안양시 동안구"),  # 거꾸로 적힌 주소
    ("대한민국 서울특별시 신림동 서울대학교 미술관", "서울"),  # 시군구를 알 수 없으면 시도만
    ("", ""),
    ("Seoul Jongno-gu", ""),  # 시도를 알아볼 수 없으면 비움
])
def test_district_of(address, expected):
    assert district_of(address) == expected


def test_db_and_log_give_the_same_text_for_the_same_place():
    db = db_place_text("경복궁", "대한민국 서울특별시 종로구 사직로 161", ["HISTORY_CULTURE"])
    log = log_visit_text({"name": "경복궁", "road_addr": "서울 종로구 사직로 161", "lotno_addr": "", "type_cd": "2"})
    assert db == log == "경복궁 | 서울 종로구 | 역사·문화"


def test_missing_parts_are_left_out():
    assert log_visit_text({"name": "온담", "road_addr": "", "lotno_addr": "", "type_cd": "9"}) == "온담"
    assert log_visit_text({"name": "온담", "road_addr": "", "lotno_addr": "전북 전주시 완산구 풍남동", "type_cd": "9"}) == "온담 | 전북 전주시 완산구"


def test_road_address_is_preferred_then_lot_number_address():
    visit = {"name": "카페", "road_addr": "", "lotno_addr": "부산 해운대구 우동 1", "type_cd": "11"}
    assert log_visit_text(visit) == "카페 | 부산 해운대구 | 음식"


def test_multiple_categories_use_a_fixed_order():
    assert category_text(["ACTIVITY", "FOOD"]) == "음식, 액티비티"
    assert category_text(["UNKNOWN"]) == ""


def test_name_is_normalized():
    text = db_place_text("카페 아래헌 / Cafe Arae-Heon", "대한민국 경상북도 경주시 황남동", ["FOOD"])
    assert text == "카페 아래헌 | 경북 경주시 | 음식"
