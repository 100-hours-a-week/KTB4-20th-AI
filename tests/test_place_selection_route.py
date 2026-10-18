import copy
import json
from datetime import timedelta
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.time import get_service_today
from app.main import app
from app.trips import router as trips_router
from app.trips.schemas.schemas import Display_Name, Location, Place
from app.trips.services.model import runtime

SURVEYS = Path(__file__).resolve().parent / "trips" / "data" / "representative_surveys.json"


def _place(i):
    return Place(
        id=f"g{i}", displayName=Display_Name(text=f"장소{i}", languageCode="ko"),
        location=Location(latitude=37.5, longitude=127.0), types=[], rating=4.5, userRatingCount=100,
    )


def _body(days=0):
    members = copy.deepcopy(json.loads(SURVEYS.read_text(encoding="utf-8"))[0]["members"])
    today = get_service_today()
    return {"region": "서울", "start_date": today.isoformat(), "end_date": (today + timedelta(days=days)).isoformat(), "members": members}


def test_first_place_seed_comes_from_the_request_and_k_from_settings(monkeypatch):
    """DB 없이 엔드포인트를 불러, 모델 경로에 넘기는 시드와 K를 확인한다."""
    seen = []

    def fake_model(*args, **kwargs):
        seen.append({"first_k": kwargs.get("first_k"), "seed": kwargs.get("seed")})
        return [_place(i) for i in range(5)], "gpu"

    monkeypatch.setattr(settings, "internal_service_token", "t")
    monkeypatch.setattr(settings, "place_selection_mode", "model")
    monkeypatch.setattr(settings, "model_first_place_top_k", 30)
    monkeypatch.setattr(runtime, "get_router", lambda: object())
    monkeypatch.setattr(trips_router, "select_places_by_model", fake_model)
    client = TestClient(app)
    for payload in (_body(), _body(), _body(days=1)):
        response = client.post("/trips/place-selection", json=payload, headers={"x-internal-token": "t"})
        assert response.status_code == 200 and len(response.json()["data"]["places"]) == 5
    assert all(item["first_k"] == 30 for item in seen)
    assert seen[0]["seed"] == seen[1]["seed"]  # 같은 요청은 같은 시드(결정론)
    assert seen[0]["seed"] != seen[2]["seed"]  # 요청이 다르면 시드도 다르다


def test_model_failure_still_answers_from_the_rule_path(monkeypatch):
    """모델 경로가 예외를 내면 규칙 경로 결과로 응답한다(엔드포인트 수준 폴백)."""
    def broken_model(*args, **kwargs):
        raise RuntimeError("모델 경로 실패")

    monkeypatch.setattr(settings, "internal_service_token", "t")
    monkeypatch.setattr(settings, "place_selection_mode", "model")
    monkeypatch.setattr(runtime, "get_router", lambda: object())
    monkeypatch.setattr(trips_router, "select_places_by_model", broken_model)
    monkeypatch.setattr(trips_router, "select_places_by_slot_plan", lambda *a, **k: [_place(i) for i in range(5)])
    response = TestClient(app).post("/trips/place-selection", json=_body(), headers={"x-internal-token": "t"})
    assert response.status_code == 200 and len(response.json()["data"]["places"]) == 5
