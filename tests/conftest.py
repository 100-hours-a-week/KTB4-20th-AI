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


@pytest.fixture(autouse=True)
def _generate_started():
    # 앱 없이 generate_missions를 직접 부르는 테스트도 있어, 테스트마다 새 동시 호출 상한을 만든다
    # (테스트마다 이벤트 루프가 달라 이전 테스트의 세마포어를 이어 쓰지 않게)
    pipeline_generate.start()
    yield
    pipeline_generate.close()
