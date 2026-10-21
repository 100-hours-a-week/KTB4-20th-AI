# tests/conftest.py
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.photomissions import pipeline_generate
from app.trips.services.model import runtime


@pytest.fixture(autouse=True)
def no_real_model_route(monkeypatch):
    # 2·3안은 항상 모델 경로(RunPod, 유료)를 먼저 시도한다. .env에 주소가 있어도 테스트가 실제로 부르지 않게 막는다.
    # 모델 경로를 쓰는 테스트는 runtime.get_router를 직접 바꾼다
    def unavailable():
        raise RuntimeError("테스트에서는 모델 추론 경로를 쓰지 않음")

    monkeypatch.setattr(runtime, "get_router", unavailable)


@pytest.fixture(scope="session")
def test_client():
    with TestClient(app) as client_instance:
        yield client_instance


@pytest.fixture
def request_headers():
    return {"X-Internal-Token": settings.internal_service_token}


@pytest.fixture
def generate_started():
    # 앱 없이 generate_missions를 직접 부르는 테스트용. 테스트마다 새 동시 호출 상한을 만들고, 끝나면 이전 값으로 돌린다
    # (TestClient로 띄운 앱은 lifespan이 만든 상한을 그대로 쓰게 둔다)
    previous = pipeline_generate._call_slots
    pipeline_generate.start()
    yield
    pipeline_generate._call_slots = previous
