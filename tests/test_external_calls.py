import io
from types import SimpleNamespace

import httpx
import pytest
import sentry_sdk
from fastapi import HTTPException
from google.genai import errors
from PIL import Image

from app.core.config import settings
from app.photomissions import (
    gemini_client,
    pipeline_generate,
    pipeline_verify,
    vlm_client,
)
from app.photomissions.pipeline_verify import ScoreResult
from app.photomissions.question_format import QuestionAnswer
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


@pytest.fixture(autouse=True)
def _gemini_backend(monkeypatch):
    # 이 파일은 Gemini 호출을 확인한다. 로컬 .env가 self여도 결과가 같도록 Gemini로 고정한다
    monkeypatch.setattr(settings, "verify_backend", "gemini")


def _api_error(code: int) -> errors.APIError:
    return errors.APIError(code, {"error": {"code": code, "message": "fake"}})


def _raising(exc: Exception):
    async def fake(*args, **kwargs):
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


async def test_score_photo_returns_result_answered_by_gemini(monkeypatch):
    monkeypatch.setattr(pipeline_verify, "generate_structured", _returning(VLM_RESULT))
    assert await _score() == ScoreResult(VLM_RESULT, "gemini")


# VERIFY_BACKEND 분기 — 설정이 정한 모델만 불리고, 미션 생성은 항상 Gemini다

async def test_gemini_backend_does_not_call_self_model(monkeypatch):
    monkeypatch.setattr(pipeline_verify, "generate_structured", _returning(VLM_RESULT))
    monkeypatch.setattr(vlm_client, "score", _raising(AssertionError("자체 모델을 부르면 안 됨")))
    assert await _score() == ScoreResult(VLM_RESULT, "gemini")


async def test_self_backend_success_does_not_call_gemini(monkeypatch):
    monkeypatch.setattr(settings, "verify_backend", "self")
    monkeypatch.setattr(pipeline_verify, "generate_structured", _raising(AssertionError("Gemini를 부르면 안 됨")))

    async def fake_score(image_bytes, place_name, mission_description):
        assert (image_bytes, place_name, mission_description) == (b"jpeg", "첨성대", "첨성대 정면이 보이게 찍기")
        return VLM_RESULT

    monkeypatch.setattr(vlm_client, "score", fake_score)
    assert await _score() == ScoreResult(VLM_RESULT, "self")


@pytest.mark.parametrize("status", [429, 502, 503, 504])
async def test_self_backend_error_is_returned_as_is(monkeypatch, status):
    # 이 단계에서는 폴백하지 않는다. 자체 모델 에러가 헤더까지 그대로 나가고 Gemini는 불리지 않는다
    monkeypatch.setattr(settings, "verify_backend", "self")
    monkeypatch.setattr(pipeline_verify, "generate_structured", _raising(AssertionError("Gemini를 부르면 안 됨")))
    error = HTTPException(status_code=status, headers={"Retry-After": "7"} if status == 429 else None)
    monkeypatch.setattr(vlm_client, "score", _raising(error))
    with pytest.raises(HTTPException) as exc:
        await _score()
    assert exc.value is error
    assert exc.value.headers == error.headers


async def test_generate_uses_gemini_even_when_verify_backend_is_self(monkeypatch):
    monkeypatch.setattr(settings, "verify_backend", "self")
    result = MissionBatch(missions=[
        MissionDescription(number=1, description="첨성대 정면이 보이게 찍기", scope="GROUP"),
    ])
    monkeypatch.setattr(pipeline_generate, "generate_structured", _returning(result))
    monkeypatch.setattr(vlm_client, "score", _raising(AssertionError("자체 모델을 부르면 안 됨")))
    assert await _describe() == result


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


class CountingStream(httpx.AsyncByteStream):
    # 본문을 청크로 나눠 주면서 몇 개를 읽었는지, 연결을 닫았는지 센다. 중간에 오류를 낼 수도 있다
    def __init__(self, chunks, fail_at=None, error=None):
        self.chunks, self.fail_at, self.error = chunks, fail_at, error
        self.read, self.closed = 0, False

    async def __aiter__(self):
        for i, chunk in enumerate(self.chunks):
            if i == self.fail_at:
                raise self.error
            self.read += 1
            yield chunk

    async def aclose(self):
        self.closed = True


@pytest.fixture
def small_limit(monkeypatch):
    # 실제 10MiB 대신 10바이트로 경계를 확인한다
    monkeypatch.setattr(pipeline_verify, "MAX_IMAGE_BYTES", 10)


async def test_fetch_image_asks_for_uncompressed_body(fake_s3):
    seen = {}

    def handler(request):
        seen["accept_encoding"] = request.headers.get("Accept-Encoding")
        return httpx.Response(200, content=b"jpeg")

    fake_s3(handler)
    await pipeline_verify.fetch_image(IMAGE_URL)
    assert seen["accept_encoding"] == "identity"


async def test_fetch_image_accepts_exactly_the_limit(fake_s3, small_limit):
    fake_s3(lambda request: httpx.Response(200, content=b"x" * 10))
    assert await pipeline_verify.fetch_image(IMAGE_URL) == b"x" * 10


async def test_fetch_image_declared_too_large_is_rejected_before_reading(fake_s3, small_limit):
    stream = CountingStream([b"x" * 11])
    fake_s3(lambda request: httpx.Response(200, headers={"Content-Length": "11"}, stream=stream))
    with pytest.raises(HTTPException) as exc:
        await pipeline_verify.fetch_image(IMAGE_URL)
    assert exc.value.status_code == 422
    assert exc.value.detail == "이미지 파일 크기가 제한을 초과했습니다"
    assert stream.read == 0 and stream.closed


@pytest.mark.parametrize(
    "headers", [{}, {"Content-Length": "4"}, {"Content-Length": "abc"}, {"Content-Length": "9" * 5000}],
    ids=["none", "too-small", "not-a-number", "too-long"],
)
async def test_fetch_image_stops_reading_once_over_the_limit(fake_s3, small_limit, headers):
    # 헤더가 없거나, 작게 적혀 있거나, 숫자가 아니어도 실제로 받은 양으로 막고 남은 청크는 읽지 않는다
    stream = CountingStream([b"x" * 6, b"x" * 6, b"x" * 6, b"x" * 6])
    fake_s3(lambda request: httpx.Response(200, headers=headers, stream=stream))
    with pytest.raises(HTTPException) as exc:
        await pipeline_verify.fetch_image(IMAGE_URL)
    assert exc.value.status_code == 422
    assert stream.read == 2 and stream.closed


async def test_fetch_image_error_status_is_502_even_with_a_large_body(fake_s3, small_limit):
    # 크기 검사보다 상태 코드를 먼저 본다. 큰 404 본문도 읽지 않는다
    stream = CountingStream([b"x" * 100])
    fake_s3(lambda request: httpx.Response(404, headers={"Content-Length": "100"}, stream=stream))
    with pytest.raises(HTTPException) as exc:
        await pipeline_verify.fetch_image(IMAGE_URL)
    assert exc.value.status_code == 502
    assert stream.read == 0


async def test_fetch_image_compressed_response_is_502_without_reading(fake_s3):
    stream = CountingStream([b"gzip-bytes"])
    fake_s3(lambda request: httpx.Response(200, headers={"Content-Encoding": "gzip"}, stream=stream))
    with pytest.raises(HTTPException) as exc:
        await pipeline_verify.fetch_image(IMAGE_URL)
    assert exc.value.status_code == 502
    assert stream.read == 0


@pytest.mark.parametrize(
    ("error", "code"),
    [(httpx.ReadTimeout("fake"), 504), (httpx.ReadError("fake"), 502), (httpx.RemoteProtocolError("fake"), 502)],
)
async def test_fetch_image_error_while_reading_body(fake_s3, error, code):
    # 본문을 받는 도중 끊겨도 처음부터 못 받은 경우와 같은 상태 코드다
    fake_s3(lambda request: httpx.Response(200, stream=CountingStream([b"x", b"x"], fail_at=1, error=error)))
    with pytest.raises(HTTPException) as exc:
        await pipeline_verify.fetch_image(IMAGE_URL)
    assert exc.value.status_code == code


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
        return ScoreResult(VLM_RESULT, "gemini")

    monkeypatch.setattr(pipeline_verify, "fetch_image", fake_fetch)
    monkeypatch.setattr(pipeline_verify, "score_photo", fake_score)
    return calls


async def test_far_photo_rejected_before_download_and_vlm(fake_verify_steps):
    seoul = Coordinates(latitude=37.5665, longitude=126.9780)
    res = await pipeline_verify.verify_photo(_verify_request(seoul))
    assert res.result == "fail"
    assert res.reason == "location_mismatch"
    assert fake_verify_steps == {"fetch_image": 0, "score_photo": 0}


async def test_far_photo_is_fail_even_with_zero_self_thresholds(fake_verify_steps, monkeypatch):
    # 예전에는 0점을 등급 기준에 넣어서, self 재시도 기준이 0이면 위치 반려가 재시도가 됐다
    monkeypatch.setattr(settings, "verify_backend", "self")
    monkeypatch.setattr(settings, "self_success_threshold", 0.0)
    monkeypatch.setattr(settings, "self_retry_threshold", 0.0)
    monkeypatch.setattr(settings, "self_landmark_threshold", 0.0)
    seoul = Coordinates(latitude=37.5665, longitude=126.9780)
    res = await pipeline_verify.verify_photo(_verify_request(seoul))
    assert (res.result, res.reason, res.match_score) == ("fail", "location_mismatch", 0)
    assert fake_verify_steps == {"fetch_image": 0, "score_photo": 0}


async def test_backend_tag_is_set_even_when_rejected_before_vlm(fake_verify_steps, monkeypatch):
    # 위치로 반려돼 모델을 부르지 않은 요청도 Sentry에서 백엔드별로 볼 수 있어야 한다
    tags = {}
    monkeypatch.setattr(sentry_sdk, "set_tag", lambda key, value: tags.update({key: value}))
    seoul = Coordinates(latitude=37.5665, longitude=126.9780)
    await pipeline_verify.verify_photo(_verify_request(seoul))
    assert tags == {"verify_backend": "gemini"}


async def test_nearby_photo_goes_to_vlm(fake_verify_steps):
    nearby = Coordinates(latitude=35.8350, longitude=129.2192)  # 약 40m
    res = await pipeline_verify.verify_photo(_verify_request(nearby))
    assert res.reason is None
    assert fake_verify_steps == {"fetch_image": 1, "score_photo": 1}


async def test_missing_photo_coordinates_goes_to_vlm(fake_verify_steps):
    res = await pipeline_verify.verify_photo(_verify_request(None))
    assert res.reason is None
    assert fake_verify_steps == {"fetch_image": 1, "score_photo": 1}


@pytest.mark.parametrize(
    ("answered_by", "expected"),
    [("self", ("success", 55, None)), ("gemini", ("retry", None, "정면에서 다시 찍어보세요"))],
)
async def test_verify_grades_with_the_answering_models_thresholds(fake_verify_steps, monkeypatch, answered_by, expected):
    # 설정이 self여도 등급·랜드마크 노출·안내 문구는 실제로 답한 모델의 기준을 따른다
    # 85점·랜드마크 55: self 기준(70/30/50)이면 성공·표시·안내 숨김, Gemini 기준(90/40/60)이면 재시도·숨김·안내 표시
    monkeypatch.setattr(settings, "verify_backend", "self")
    monkeypatch.setattr(settings, "self_success_threshold", 70.0)
    monkeypatch.setattr(settings, "self_retry_threshold", 30.0)
    monkeypatch.setattr(settings, "self_landmark_threshold", 50.0)

    vlm = VlmResult(detected_labels=[], landmark_confidence=55, match_score=85, retry_hint="정면에서 다시 찍어보세요")

    async def fake_score(image_bytes, place_name, mission_description):
        return ScoreResult(vlm, answered_by)

    monkeypatch.setattr(pipeline_verify, "score_photo", fake_score)
    res = await pipeline_verify.verify_photo(_verify_request(None))
    assert (res.result, res.landmark_confidence, res.retry_hint) == expected
    assert res.match_score == 85


async def test_verify_uses_question_rule_without_score_thresholds(fake_verify_steps, monkeypatch):
    # 객관식 답이면 점수 기준을 읽지 않는다. self 기준이 비어 있어도(객관식 설정) 응답을 만든다
    monkeypatch.setattr(settings, "verify_backend", "self")
    monkeypatch.setattr(settings, "vlm_format", "question")
    monkeypatch.setattr(settings, "self_success_threshold", None)
    monkeypatch.setattr(settings, "self_retry_threshold", None)
    monkeypatch.setattr(settings, "self_landmark_threshold", None)
    answer = QuestionAnswer(judgeable="yes", target="short", condition="na", place="yes")

    async def fake_score(image_bytes, place_name, mission_description):
        return ScoreResult(answer, "self")

    monkeypatch.setattr(pipeline_verify, "score_photo", fake_score)
    res = await pipeline_verify.verify_photo(_verify_request(None))
    assert (res.result, res.match_score, res.retry_hint) == ("retry", 50.0, "미션 대상이 더 잘 보이게 다시 찍어주세요")


async def test_gemini_answer_uses_gemini_thresholds_even_in_question_mode(fake_verify_steps, monkeypatch):
    # 나중에 폴백으로 Gemini가 답하면 점수 응답이다. 설정이 객관식이어도 Gemini 기준으로 등급을 정한다
    monkeypatch.setattr(settings, "verify_backend", "self")
    monkeypatch.setattr(settings, "vlm_format", "question")
    monkeypatch.setattr(settings, "self_success_threshold", None)

    async def fake_score(image_bytes, place_name, mission_description):
        return ScoreResult(VLM_RESULT, "gemini")  # 85점

    monkeypatch.setattr(pipeline_verify, "score_photo", fake_score)
    res = await pipeline_verify.verify_photo(_verify_request(None))
    assert (res.result, res.match_score) == ("retry", 85)


async def test_gemini_backend_ignores_vlm_format(monkeypatch):
    # VLM_FORMAT은 자체 모델에만 쓴다. gemini면 객관식 설정이어도 Gemini 점수 판정이다
    monkeypatch.setattr(settings, "vlm_format", "question")
    monkeypatch.setattr(pipeline_verify, "generate_structured", _returning(VLM_RESULT))
    monkeypatch.setattr(vlm_client, "score", _raising(AssertionError("자체 모델을 부르면 안 됨")))
    assert await _score() == ScoreResult(VLM_RESULT, "gemini")


async def test_unreadable_photo_is_422_without_calling_vlm(fake_verify_steps, monkeypatch):
    async def fake_fetch(image_url):
        fake_verify_steps["fetch_image"] += 1
        return b"not an image"

    monkeypatch.setattr(pipeline_verify, "fetch_image", fake_fetch)
    with pytest.raises(HTTPException) as exc:
        await pipeline_verify.verify_photo(_verify_request(None))
    assert exc.value.status_code == 422
    assert fake_verify_steps == {"fetch_image": 1, "score_photo": 0}


async def test_vlm_gets_the_same_bytes_as_normalize_image(fake_verify_steps, monkeypatch):
    # 정상 사진은 이번 수정 전과 같은 전처리 결과가 모델로 간다(측정 입력과 같아야 한다)
    seen = {}

    async def fake_score(image_bytes, place_name, mission_description):
        seen["bytes"] = image_bytes
        return ScoreResult(VLM_RESULT, "gemini")

    monkeypatch.setattr(pipeline_verify, "score_photo", fake_score)
    raw = await pipeline_verify.fetch_image(IMAGE_URL)
    await pipeline_verify.verify_photo(_verify_request(None))
    assert seen["bytes"] == pipeline_verify.normalize_image(raw)


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
