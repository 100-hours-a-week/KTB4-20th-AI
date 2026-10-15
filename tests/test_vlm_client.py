import pytest

from app.core.config import Settings, settings
from app.main import app, lifespan
from app.photomissions import vlm_client

pytestmark = pytest.mark.anyio

SELF_OK = {
    "verify_backend": "self",
    "vlm_base_url": "https://vlm.example.com/v1",
    "vlm_model": "qwen",
    "vlm_deadline_seconds": 60,
    "self_success_threshold": 80,
    "self_retry_threshold": 40,
    "self_landmark_threshold": 50,
}


def _settings(**overrides) -> Settings:
    # .env를 읽지 않아야 로컬 설정에 따라 결과가 달라지지 않는다
    return Settings(_env_file=None, **{**SELF_OK, **overrides})


# 1. 시작 검사 — self에 필요한 설정이 빠지거나 틀리면 모두 잡는다

def test_complete_self_settings_pass():
    assert vlm_client.settings_errors(_settings()) == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"vlm_base_url": ""},
        {"vlm_base_url": "   "},
        {"vlm_base_url": "vlm.example.com/v1"},  # http(s)://가 빠짐
        {"vlm_base_url": "ftp://vlm.example.com/v1"},
        {"vlm_model": " "},
        {"vlm_deadline_seconds": None},
        {"vlm_deadline_seconds": 0},
        {"self_success_threshold": None},
        {"self_retry_threshold": None},
        {"self_landmark_threshold": None},
        {"self_retry_threshold": 90},  # 재시도 기준이 성공 기준보다 높음
        {"self_success_threshold": 120},
        {"self_retry_threshold": -1},
        {"self_landmark_threshold": 101},
    ],
)
def test_wrong_self_settings_are_reported(overrides):
    assert vlm_client.settings_errors(_settings(**overrides))


def test_check_settings_reports_every_problem_at_once():
    # 배포할 때 한 번에 다 고칠 수 있게 하나씩 멈추지 않는다
    with pytest.raises(RuntimeError) as exc:
        vlm_client.check_settings(_settings(vlm_model="", vlm_deadline_seconds=None))
    assert "VLM_MODEL" in str(exc.value) and "VLM_DEADLINE_SECONDS" in str(exc.value)


# 2. 앱 시작 — 공용 test_client(테스트 전체에서 한 번만 시작)와 섞이지 않게 lifespan을 직접 실행한다

async def test_self_backend_with_missing_settings_does_not_start(monkeypatch):
    monkeypatch.setattr(settings, "verify_backend", "self")
    monkeypatch.setattr(settings, "vlm_base_url", "")
    with pytest.raises(RuntimeError):
        async with lifespan(app):
            pass


async def test_gemini_backend_starts_without_vlm_settings(monkeypatch, caplog):
    monkeypatch.setattr(settings, "verify_backend", "gemini")
    monkeypatch.setattr(settings, "vlm_base_url", "")
    monkeypatch.setattr(settings, "self_success_threshold", None)
    with caplog.at_level("INFO", logger="uvicorn.error"):
        async with lifespan(app):
            pass
    assert "verify 판정 백엔드: gemini" in caplog.text


# 3. 시작 로그 — 개발 중에 지금 어느 모델로 판정하는지 바로 보이게 한다

def test_self_backend_summary_shows_model_and_thresholds_without_secrets():
    summary = vlm_client.backend_summary(_settings(vlm_api_key="secret-key"))
    assert summary == "verify 판정 백엔드: self (모델 qwen, 마감 60초, 기준 성공 80 / 재시도 40 / 랜드마크 50)"
    assert "secret-key" not in summary and "vlm.example.com" not in summary
