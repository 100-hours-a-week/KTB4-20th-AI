# tests/conftest.py
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app


@pytest.fixture(scope="session")
def test_client():
    with TestClient(app) as client_instance:
        yield client_instance


@pytest.fixture
def request_headers():
    return {"X-Internal-Token": settings.internal_service_token}