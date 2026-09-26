import io
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException
from google.genai import errors
from PIL import Image

from app.core.config import settings
from app.photomissions import gemini_client, pipeline_generate, pipeline_verify
from app.photomissions.schemas import (
    Coordinates,
    DisplayName,
    MissionBatch,
    MissionDescription,
    MissionPlace,
    PhotoMissionGenerateRequest,
    VerifyRequest,
    VlmResult,
)

pytestmark = pytest.mark.anyio

VLM_RESULT = VlmResult(detected_labels=[], landmark_confidence=90, match_score=85, retry_hint="")
PLACE = MissionPlace(
    id="places/ChIJ123",
    displayName=DisplayName(text="첨성대", languageCode="ko"),
    selected_for=[],
    matched_preferences=["HISTORY_CULTURE"],
)


def _api_error(code: int) -> errors.APIError:
    return errors.APIError(code, {"error": {"code": code, "message": "fake"}})


def _raising(exc: Exception):
    async def fake(**kwargs):
        raise exc
    return fake


def _returning(value):
    async def fake(**kwargs):
        return value
    return fake


# 1. score_photo / write_description — Gemini 에러를 HTTP 상태 코드로 바꾸는지
# generate_structured는 각 파이프라인 모듈로 import돼 있어서, 쓰는 쪽 모듈의 이름을 바꿔야 한다

async def _score():
    return await pipeline_verify.score_photo(b"jpeg", "첨성대", "첨성대 정면이 보이게 찍기")


async def _describe():
    return await pipeline_generate.write_descriptions([PLACE])


CALLERS = [
    pytest.param(pipeline_verify, _score, id="score_photo"),
    pytest.param(pipeline_generate, _describe, id="write_descriptions"),
]


@pytest.mark.parametrize(("module", "call"), CALLERS)
async def test_rate_limit_maps_to_429_with_retry_after(monkeypatch, module, call):
    monkeypatch.setattr(module, "generate_structured", _raising(_api_error(429)))
    with pytest.raises(HTTPException) as exc:
        await call()
    assert exc.value.status_code == 429
    assert exc.value.headers == {"Retry-After": "30"}


@pytest.mark.parametrize(("module", "call"), CALLERS)
async def test_other_api_error_maps_to_502(monkeypatch, module, call):
    monkeypatch.setattr(module, "generate_structured", _raising(_api_error(500)))
    with pytest.raises(HTTPException) as exc:
        await call()
    assert exc.value.status_code == 502


@pytest.mark.parametrize(("module", "call"), CALLERS)
async def test_timeout_maps_to_504(monkeypatch, module, call):
    monkeypatch.setattr(module, "generate_structured", _raising(httpx.ReadTimeout("fake")))
    with pytest.raises(HTTPException) as exc:
        await call()
    assert exc.value.status_code == 504


async def test_score_photo_schema_mismatch_after_retry_maps_to_502(monkeypatch):
    # 재시도 후에도 응답이 스키마와 맞지 않으면 generate_structured가 ValueError를 올린다
    monkeypatch.setattr(pipeline_verify, "generate_structured", _raising(ValueError("fake")))
    with pytest.raises(HTTPException) as exc:
        await _score()
    assert exc.value.status_code == 502


async def test_score_photo_returns_vlm_result(monkeypatch):
    monkeypatch.setattr(pipeline_verify, "generate_structured", _returning(VLM_RESULT))
    assert await _score() == VLM_RESULT


async def test_write_descriptions_returns_result(monkeypatch):
    result = MissionBatch(missions=[
        MissionDescription(number=1, description="첨성대 정면이 보이게 찍기", scope="GROUP"),
    ])
    monkeypatch.setattr(pipeline_generate, "generate_structured", _returning(result))
    assert await _describe() == result


async def test_write_descriptions_invalid_after_retry_maps_to_502(monkeypatch):
    # 재시도 후에도 장소와 짝지을 수 없으면 generate_structured가 ValueError를 올린다
    monkeypatch.setattr(pipeline_generate, "generate_structured", _raising(ValueError("fake")))
    with pytest.raises(HTTPException) as exc:
        await _describe()
    assert exc.value.status_code == 502


# 2. fetch_image — 실제 네트워크 대신 MockTransport가 응답한다

IMAGE_URL = "https://my-bucket.s3.ap-northeast-2.amazonaws.com/photo.jpg"


@pytest.fixture
def fake_s3(monkeypatch):
    real_client = httpx.AsyncClient

    def install(handler):
        monkeypatch.setattr(
            httpx, "AsyncClient",
            lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
        )
    return install


async def test_fetch_image_returns_bytes(fake_s3):
    fake_s3(lambda request: httpx.Response(200, content=b"jpeg-bytes"))
    assert await pipeline_verify.fetch_image(IMAGE_URL) == b"jpeg-bytes"


async def test_fetch_image_error_status_maps_to_502(fake_s3):
    fake_s3(lambda request: httpx.Response(404))
    with pytest.raises(HTTPException) as exc:
        await pipeline_verify.fetch_image(IMAGE_URL)
    assert exc.value.status_code == 502


async def test_fetch_image_timeout_maps_to_504(fake_s3):
    def handler(request):
        raise httpx.ReadTimeout("fake", request=request)

    fake_s3(handler)
    with pytest.raises(HTTPException) as exc:
        await pipeline_verify.fetch_image(IMAGE_URL)
    assert exc.value.status_code == 504


async def test_fetch_image_connection_error_maps_to_502(fake_s3):
    # 응답 자체를 못 받은 경우(DNS·네트워크 장애)도 404 같은 실패 응답과 똑같이 502여야 한다
    def handler(request):
        raise httpx.ConnectError("fake", request=request)

    fake_s3(handler)
    with pytest.raises(HTTPException) as exc:
        await pipeline_verify.fetch_image(IMAGE_URL)
    assert exc.value.status_code == 502


# 3. generate_structured — 재시도 횟수. 가짜 클라이언트가 호출 횟수를 센다

class FakeGemini:
    def __init__(self, *outcomes):
        # outcomes: 호출마다 돌려줄 결과. Exception이면 raise, 아니면 parsed 값
        self.outcomes = list(outcomes)
        self.calls = 0
        self.aio = SimpleNamespace(models=SimpleNamespace(generate_content=self._generate))

    async def _generate(self, **kwargs):
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(parsed=outcome)


async def _call_generate_structured():
    return await gemini_client.generate_structured(
        system="system", user="user", response_schema=VlmResult,
    )


@pytest.fixture
def fake_gemini(monkeypatch):
    def install(*outcomes):
        fake = FakeGemini(*outcomes)
        monkeypatch.setattr(gemini_client, "_get_client", lambda: fake)
        return fake
    return install


async def test_success_on_first_try(fake_gemini):
    fake = fake_gemini(VLM_RESULT)
    assert await _call_generate_structured() == VLM_RESULT
    assert fake.calls == 1


async def test_429_is_not_retried(fake_gemini):
    fake = fake_gemini(_api_error(429), VLM_RESULT)
    with pytest.raises(errors.APIError) as exc:
        await _call_generate_structured()
    assert exc.value.code == 429
    assert fake.calls == 1


async def test_500_is_retried_once_then_succeeds(fake_gemini):
    fake = fake_gemini(_api_error(500), VLM_RESULT)
    assert await _call_generate_structured() == VLM_RESULT
    assert fake.calls == 2


async def test_schema_mismatch_is_retried(fake_gemini):
    fake = fake_gemini(None, VLM_RESULT)  # parsed=None: 응답이 스키마와 안 맞음
    assert await _call_generate_structured() == VLM_RESULT
    assert fake.calls == 2


async def test_invalid_content_is_retried(fake_gemini):
    fake = fake_gemini(VLM_RESULT, VLM_RESULT)
    checked = iter([False, True])  # 첫 응답은 검사 불통과, 두 번째는 통과
    result = await gemini_client.generate_structured(
        system="system", user="user", response_schema=VlmResult,
        is_valid=lambda _: next(checked),
    )
    assert result == VLM_RESULT
    assert fake.calls == 2


async def test_invalid_content_gives_up_with_value_error(fake_gemini):
    fake = fake_gemini(VLM_RESULT, VLM_RESULT)
    with pytest.raises(ValueError):
        await gemini_client.generate_structured(
            system="system", user="user", response_schema=VlmResult,
            is_valid=lambda _: False,
        )
    assert fake.calls == gemini_client.MAX_ATTEMPTS


async def test_gives_up_after_max_attempts(fake_gemini):
    fake = fake_gemini(_api_error(500), _api_error(503))
    with pytest.raises(errors.APIError) as exc:
        await _call_generate_structured()
    assert exc.value.code == 503  # 마지막 에러를 올린다
    assert fake.calls == gemini_client.MAX_ATTEMPTS


async def test_timeout_propagates_as_timeout_after_retry(fake_gemini):
    # 재시도 후에도 타임아웃이면 원래 예외 타입 그대로 올라가야 score_photo가 504로 바꿀 수 있다
    fake = fake_gemini(httpx.ReadTimeout("fake"), httpx.ReadTimeout("fake"))
    with pytest.raises(httpx.TimeoutException):
        await _call_generate_structured()
    assert fake.calls == 2


# 4. verify_photo — 위치 사전 판정이 사진 다운로드·VLM 호출보다 먼저 일어나는지

CHEOMSEONGDAE = Coordinates(latitude=35.8347, longitude=129.2192)


def _verify_request(photo_coordinates: Coordinates | None) -> VerifyRequest:
    return VerifyRequest(
        image_url=IMAGE_URL,
        place_id="places/ChIJ123",
        place_name="첨성대",
        place_coordinates=CHEOMSEONGDAE,
        mission_description="첨성대 정면이 보이게 찍기",
        photo_coordinates=photo_coordinates,
    )


@pytest.fixture
def fake_verify_steps(monkeypatch):
    # 다운로드와 VLM 판정을 가짜로 바꾸고, 각각 불린 횟수를 센다
    monkeypatch.setattr(settings, "allowed_image_hosts", "my-bucket.s3.ap-northeast-2.amazonaws.com")
    calls = {"fetch_image": 0, "score_photo": 0}
    buf = io.BytesIO()
    Image.new("RGB", (10, 10)).save(buf, format="JPEG")

    async def fake_fetch(image_url):
        calls["fetch_image"] += 1
        return buf.getvalue()

    async def fake_score(image_bytes, place_name, mission_description):
        calls["score_photo"] += 1
        return VLM_RESULT

    monkeypatch.setattr(pipeline_verify, "fetch_image", fake_fetch)
    monkeypatch.setattr(pipeline_verify, "score_photo", fake_score)
    return calls


async def test_far_photo_rejected_before_download_and_vlm(fake_verify_steps):
    seoul = Coordinates(latitude=37.5665, longitude=126.9780)
    res = await pipeline_verify.verify_photo(_verify_request(seoul))
    assert res.result == "fail"
    assert res.reason == "location_mismatch"
    assert fake_verify_steps == {"fetch_image": 0, "score_photo": 0}


async def test_nearby_photo_goes_to_vlm(fake_verify_steps):
    nearby = Coordinates(latitude=35.8350, longitude=129.2192)  # 약 40m
    res = await pipeline_verify.verify_photo(_verify_request(nearby))
    assert res.reason is None
    assert fake_verify_steps == {"fetch_image": 1, "score_photo": 1}


async def test_missing_photo_coordinates_goes_to_vlm(fake_verify_steps):
    res = await pipeline_verify.verify_photo(_verify_request(None))
    assert res.reason is None
    assert fake_verify_steps == {"fetch_image": 1, "score_photo": 1}


# 5. generate_missions — 장소 전부를 한 번에 부르고, 번호로 장소와 짝짓는지

def _place(place_id: str, name: str) -> MissionPlace:
    return MissionPlace(
        id=place_id,
        displayName=DisplayName(text=name, languageCode="ko"),
        selected_for=[],
        matched_preferences=[],
    )


async def test_generate_missions_one_call_matched_by_number(monkeypatch):
    places = [_place("places/A", "첨성대"), _place("places/B", "대릉원"), _place("places/C", "월정교")]
    calls = []

    async def fake_generate(**kwargs):
        calls.append(kwargs)
        # 모델이 순서를 섞어서 돌려준 경우
        return MissionBatch(missions=[
            MissionDescription(number=3, description="월정교 미션", scope="GROUP"),
            MissionDescription(number=1, description="첨성대 미션", scope="PERSONAL"),
            MissionDescription(number=2, description="대릉원 미션", scope="GROUP"),
        ])

    monkeypatch.setattr(pipeline_generate, "generate_structured", fake_generate)
    request = PhotoMissionGenerateRequest(itinerary_id="it_1", places=places)
    res = await pipeline_generate.generate_missions(request)

    assert len(calls) == 1
    assert [(m.place_id, m.description) for m in res.missions] == [
        ("places/A", "첨성대 미션"),
        ("places/B", "대릉원 미션"),
        ("places/C", "월정교 미션"),
    ]


async def test_generate_missions_retries_when_place_missing(fake_gemini):
    # 실제 generate_structured를 거쳐, 장소가 빠진 응답이면 다시 부르는지 확인한다
    places = [_place("places/A", "첨성대"), _place("places/B", "대릉원")]
    missing = MissionBatch(missions=[
        MissionDescription(number=1, description="첨성대 미션", scope="GROUP"),
    ])
    complete = MissionBatch(missions=[
        MissionDescription(number=1, description="첨성대 미션", scope="GROUP"),
        MissionDescription(number=2, description="대릉원 미션", scope="GROUP"),
    ])
    fake = fake_gemini(missing, complete)
    request = PhotoMissionGenerateRequest(itinerary_id="it_1", places=places)

    res = await pipeline_generate.generate_missions(request)

    assert fake.calls == 2
    assert [m.place_id for m in res.missions] == ["places/A", "places/B"]
