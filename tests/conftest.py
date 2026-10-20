# tests/conftest.py
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.photomissions import pipeline_generate


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
