import re

from app.core.config import Settings

# 이름에 이 말이 들어가는 새 필드가 생기면 아래 목록 테스트가 실패해, 비밀값인지 검토하고 repr=False를 정하게 한다.
# 모든 비밀값을 자동으로 찾는 장치는 아니다(credential, database_url 같은 이름은 놓친다)
SECRET_NAME = re.compile(r"key|token|password|dsn|secret")
SECRET_FIELDS = [
    "gemini_api_key", "internal_service_token", "google_places_api_key", "kma_api_key",
    "db_password", "sentry_dsn", "vlm_api_key", "runpod_model_api_key",
]
# 비밀 키는 아니지만 Pod·엔드포인트 ID가 들어 있어 같이 가린다
LOCATION_FIELDS = ["vlm_base_url", "runpod_model_url"]
HIDDEN_FIELDS = SECRET_FIELDS + LOCATION_FIELDS


def test_secret_name_rule_matches_known_fields():
    assert sorted(name for name in Settings.model_fields if SECRET_NAME.search(name)) == sorted(SECRET_FIELDS)


def test_hidden_fields_are_excluded_from_repr():
    for name in HIDDEN_FIELDS:
        assert Settings.model_fields[name].repr is False, name


def test_repr_and_str_hide_values_but_settings_keep_them():
    values = {name: f"DUMMY-{name}" for name in HIDDEN_FIELDS}
    settings = Settings(_env_file=None, vlm_model="qwen-test", **values)
    for text in (repr(settings), str(settings)):
        assert "qwen-test" in text  # 일반 설정은 그대로 보인다
        assert not [name for name, value in values.items() if value in text]
    # 출력만 가리고, 코드가 쓰는 값은 그대로다
    assert {name: getattr(settings, name) for name in HIDDEN_FIELDS} == values
