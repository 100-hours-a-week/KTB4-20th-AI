import unicodedata

import pytest

from scripts.semantic_id.text import (
    TYPE_CODE_CATEGORY,
    category_for_type,
    normalize_name,
)


@pytest.mark.parametrize("raw, expected", [
    ("카페 아래헌 / Cafe Arae-Heon / カフェアレホン / 咖啡馆 Arae-heon", "카페 아래헌"),
    ("고집돌우럭 중문점 (Gozipdol Ureok Jungmun Branch)", "고집돌우럭 중문점"),
    ("스타벅스 (Starbucks) 서울역점", "스타벅스 서울역점"),
    ("Gymboxx Fitness Pyeongchon (평촌점)", "Gymboxx Fitness Pyeongchon 평촌점"),
    ("Starbucks Gangnam", "Starbucks Gangnam"),  # 한글이 없으면 그대로
    ("AK& 홍대", "AK& 홍대"),  # 브랜드 약어는 그대로
    ("버거킹 평촌금성GS점", "버거킹 평촌금성GS점"),
    ("  경복궁   ", "경복궁"),
    ("해운대（海雲臺）해수욕장", "해운대 해수욕장"),  # 전각 괄호와 한자 병기
    ("", ""),
])
def test_normalize_name(raw, expected):
    assert normalize_name(raw) == expected


def test_normalize_name_is_nfc_and_idempotent():
    decomposed = unicodedata.normalize("NFD", "월정교 (Woljeonggyo)")
    out = normalize_name(decomposed)
    assert out == "월정교"
    assert unicodedata.is_normalized("NFC", out)
    assert normalize_name(out) == out


def test_type_code_table_uses_only_service_categories():
    assert set(TYPE_CODE_CATEGORY.values()) <= {
        "FOOD", "CONVENIENCE_SHOPPING", "HISTORY_CULTURE", "ACTIVITY", "NATURE_HEALING",
    }


@pytest.mark.parametrize("code, expected", [("11", "FOOD"), ("4", "CONVENIENCE_SHOPPING"), ("1", "NATURE_HEALING"),
                                            ("7", None), ("9", None), ("13", None), ("", None)])
def test_category_for_type(code, expected):
    assert category_for_type(code) == expected
