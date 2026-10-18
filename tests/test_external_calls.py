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
REAL_SCORE_PHOTO = pipeline_verify.score_photo
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

    async def fake_score(image_bytes, place_name, mission_description, *, deadline=None, trace=None):
        assert (image_bytes, place_name, mission_description) == (b"jpeg", "첨성대", "첨성대 정면이 보이게 찍기")
        assert deadline is None  # 폴백을 켜지 않으면 마감은 자체 모델 설정 그대로다
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

    async def fake_score(image_bytes, place_name, mission_description, trace=None):
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

    async def fake_score(image_bytes, place_name, mission_description, trace=None):
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

    async def fake_score(image_bytes, place_name, mission_description, trace=None):
        return ScoreResult(answer, "self")

    monkeypatch.setattr(pipeline_verify, "score_photo", fake_score)
    res = await pipeline_verify.verify_photo(_verify_request(None))
    assert (res.result, res.match_score, res.retry_hint) == ("retry", 50.0, "미션 대상이 더 잘 보이게 다시 찍어주세요")


async def test_gemini_answer_uses_gemini_thresholds_even_in_question_mode(fake_verify_steps, monkeypatch):
    # 나중에 폴백으로 Gemini가 답하면 점수 응답이다. 설정이 객관식이어도 Gemini 기준으로 등급을 정한다
    monkeypatch.setattr(settings, "verify_backend", "self")
    monkeypatch.setattr(settings, "vlm_format", "question")
    monkeypatch.setattr(settings, "self_success_threshold", None)

    async def fake_score(image_bytes, place_name, mission_description, trace=None):
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

    async def fake_score(image_bytes, place_name, mission_description, trace=None):
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



# 6. 폴백 — self에서 자체 모델이 실패하거나 운영 시간 밖이면 Gemini로, 모델 단계 전체 예산 안에서

GEMINI_RESULT = VlmResult(detected_labels=[], landmark_confidence=70, match_score=95, retry_hint="")


@pytest.fixture
def fallback(monkeypatch):
    # self + 폴백 설정. 운영 시간 판단 시계는 운영 시간 안(10:00)으로 고정한다
    from datetime import datetime

    from app.core.time import SERVICE_TIMEZONE

    for key, value in {
        "verify_backend": "self", "vlm_format": "score", "verify_model_budget_seconds": 20.0,
        "vlm_serving_hours": "09:00-17:00", "vlm_deadline_seconds": 60.0,
        "self_success_threshold": 70.0, "self_retry_threshold": 30.0, "self_landmark_threshold": 50.0,
    }.items():
        monkeypatch.setattr(settings, key, value)
    clock = {"now": datetime(2026, 10, 20, 10, 0, tzinfo=SERVICE_TIMEZONE)}
    monkeypatch.setattr(pipeline_verify, "_now_in_service_timezone", lambda: clock["now"])
    calls = {"self": [], "gemini": 0}

    def set_self(outcome):
        async def fake_score(image_bytes, place_name, mission_description, *, deadline=None, trace=None):
            calls["self"].append(deadline)
            if trace is not None:
                trace.update(attempts=1, upstream_status=None, last_error="ReadTimeout")
            if isinstance(outcome, BaseException):
                raise outcome
            if callable(outcome):
                return await outcome()
            return outcome
        monkeypatch.setattr(vlm_client, "score", fake_score)

    def set_gemini(outcome):
        async def fake_generate(**kwargs):
            calls["gemini"] += 1
            if isinstance(outcome, BaseException):
                raise outcome
            if callable(outcome):
                return await outcome()
            return outcome
        monkeypatch.setattr(pipeline_verify, "generate_structured", fake_generate)

    set_gemini(GEMINI_RESULT)
    return SimpleNamespace(calls=calls, clock=clock, set_self=set_self, set_gemini=set_gemini)


async def _score_traced():
    trace = pipeline_verify.VerifyTrace()
    result = await pipeline_verify.score_photo(b"jpeg", "첨성대", "첨성대 정면이 보이게 찍기", trace)
    return result, trace


async def test_self_success_does_not_fall_back(fallback):
    fallback.set_self(VLM_RESULT)
    result, trace = await _score_traced()
    assert result == ScoreResult(VLM_RESULT, "self")
    assert fallback.calls["gemini"] == 0
    assert (trace.route, trace.answered_by, trace.fallback_enabled, trace.fallback_attempted) == ("self", "self", True, False)


@pytest.mark.parametrize("status", [429, 502, 503, 504])
async def test_self_failure_falls_back_to_gemini(fallback, status):
    fallback.set_self(HTTPException(status_code=status))
    result, trace = await _score_traced()
    assert result == ScoreResult(GEMINI_RESULT, "gemini")
    assert fallback.calls["gemini"] == 1
    assert (trace.route, trace.answered_by, trace.self_error_status) == ("self", "gemini", status)
    assert (trace.fallback_attempted, trace.fallback_succeeded, trace.fallback_skip_reason) == (True, True, None)
    assert trace.self_call["last_error"] == "ReadTimeout"  # 자체 모델 쪽 원인도 남는다


async def test_self_waits_less_to_leave_room_for_gemini(fallback):
    import asyncio

    fallback.set_self(VLM_RESULT)
    started = asyncio.get_running_loop().time()
    await _score_traced()
    left_for_gemini = started + 20.0 - fallback.calls["self"][0]
    expected = pipeline_verify.vlm_client.MIN_GEMINI_START_BUDGET_SECONDS + pipeline_verify.vlm_client.BUDGET_MARGIN_SECONDS
    assert left_for_gemini == pytest.approx(expected, abs=0.05)


@pytest.mark.parametrize(
    "outcome",
    [
        pytest.param(HTTPException(status_code=422), id="other-http-error"),
        pytest.param(RuntimeError("클라이언트 미시작"), id="program-error"),
    ],
)
async def test_other_errors_do_not_fall_back(fallback, outcome):
    fallback.set_self(outcome)
    with pytest.raises(type(outcome)):
        await _score_traced()
    assert fallback.calls["gemini"] == 0


@pytest.mark.parametrize("judged", ["retry", "fail"])
async def test_normal_retry_or_fail_answer_does_not_fall_back(fallback, judged):
    score = {"retry": 40, "fail": 10}[judged]
    fallback.set_self(VlmResult(detected_labels=[], landmark_confidence=10, match_score=score, retry_hint=""))
    result, trace = await _score_traced()
    assert result.answered_by == "self" and fallback.calls["gemini"] == 0
    assert trace.answered_by == "self"


async def test_fallback_gemini_error_is_returned_as_gemini_error(fallback):
    fallback.set_self(HTTPException(status_code=504))
    fallback.set_gemini(_api_error(429))
    with pytest.raises(HTTPException) as exc:
        await _score_traced()
    assert (exc.value.status_code, exc.value.headers) == (429, {"Retry-After": "30"})  # Gemini 쪽 헤더


async def test_fallback_gemini_failure_leaves_no_answerer(fallback):
    fallback.set_self(HTTPException(status_code=504))
    fallback.set_gemini(_api_error(500))
    trace = pipeline_verify.VerifyTrace()
    with pytest.raises(HTTPException):
        await pipeline_verify.score_photo(b"jpeg", "첨성대", "첨성대 정면이 보이게 찍기", trace)
    assert (trace.answered_by, trace.fallback_attempted, trace.fallback_succeeded) == (None, True, False)


async def test_not_enough_time_keeps_the_self_error(fallback, monkeypatch):
    # 자체 모델이 늦게 실패해 Gemini 몫이 모자라면, 원인을 504로 바꾸지 않고 자체 모델 에러를 돌려준다
    import asyncio

    monkeypatch.setattr(settings, "verify_model_budget_seconds", 0.4)
    monkeypatch.setattr(pipeline_verify.vlm_client, "MIN_GEMINI_START_BUDGET_SECONDS", 0.3)
    monkeypatch.setattr(pipeline_verify.vlm_client, "BUDGET_MARGIN_SECONDS", 0.0)

    async def late_502():
        await asyncio.sleep(0.2)
        raise HTTPException(status_code=502)

    fallback.set_self(late_502)
    trace = pipeline_verify.VerifyTrace()
    with pytest.raises(HTTPException) as exc:
        await pipeline_verify.score_photo(b"jpeg", "첨성대", "첨성대 정면이 보이게 찍기", trace)
    assert exc.value.status_code == 502
    assert fallback.calls["gemini"] == 0
    assert trace.fallback_skip_reason == "insufficient_budget"


async def test_gemini_past_the_budget_is_504(fallback, monkeypatch):
    import asyncio

    monkeypatch.setattr(settings, "verify_model_budget_seconds", 0.3)
    monkeypatch.setattr(pipeline_verify.vlm_client, "MIN_GEMINI_START_BUDGET_SECONDS", 0.1)
    monkeypatch.setattr(pipeline_verify.vlm_client, "BUDGET_MARGIN_SECONDS", 0.0)

    async def slow_gemini():
        await asyncio.sleep(5)
        return GEMINI_RESULT

    fallback.set_self(HTTPException(status_code=502))
    fallback.set_gemini(slow_gemini)
    started = asyncio.get_running_loop().time()
    with pytest.raises(HTTPException) as exc:
        await _score_traced()
    assert exc.value.status_code == 504
    assert asyncio.get_running_loop().time() - started < 2


@pytest.mark.parametrize("hour", [8, 17, 23])
async def test_off_hours_go_straight_to_gemini(fallback, hour):
    fallback.clock["now"] = fallback.clock["now"].replace(hour=hour)
    fallback.set_self(AssertionError("운영 시간 밖에는 자체 모델을 부르지 않음"))
    result, trace = await _score_traced()
    assert result == ScoreResult(GEMINI_RESULT, "gemini")
    assert (trace.route, trace.fallback_attempted, trace.answered_by) == ("off_hours", False, "gemini")


async def test_off_hours_gemini_is_also_bounded_by_the_budget(fallback, monkeypatch):
    import asyncio

    monkeypatch.setattr(settings, "verify_model_budget_seconds", 0.3)
    fallback.clock["now"] = fallback.clock["now"].replace(hour=20)

    async def slow_gemini():
        await asyncio.sleep(5)
        return GEMINI_RESULT

    fallback.set_gemini(slow_gemini)
    with pytest.raises(HTTPException) as exc:
        await _score_traced()
    assert exc.value.status_code == 504


async def test_serving_hours_use_korean_time_not_utc(fallback):
    # UTC 00:30은 한국 09:30이라 운영 시간 안이다(날짜가 바뀌는 경우)
    from datetime import UTC, datetime

    from app.core.time import SERVICE_TIMEZONE

    fallback.clock["now"] = datetime(2026, 10, 20, 0, 30, tzinfo=UTC).astimezone(SERVICE_TIMEZONE)
    fallback.set_self(VLM_RESULT)
    result, _ = await _score_traced()
    assert result.answered_by == "self"


async def test_without_budget_self_errors_are_returned_as_before(fallback, monkeypatch):
    monkeypatch.setattr(settings, "verify_model_budget_seconds", None)
    monkeypatch.setattr(settings, "vlm_serving_hours", "")
    fallback.set_self(HTTPException(status_code=504))
    trace = pipeline_verify.VerifyTrace()
    with pytest.raises(HTTPException) as exc:
        await pipeline_verify.score_photo(b"jpeg", "첨성대", "첨성대 정면이 보이게 찍기", trace)
    assert exc.value.status_code == 504
    assert fallback.calls["gemini"] == 0 and fallback.calls["self"] == [None]
    assert (trace.route, trace.fallback_enabled) == ("self", False)


async def test_gemini_backend_ignores_fallback_settings(fallback, monkeypatch):
    monkeypatch.setattr(settings, "verify_backend", "gemini")
    fallback.set_self(AssertionError("자체 모델을 부르면 안 됨"))
    result, trace = await _score_traced()
    assert result == ScoreResult(GEMINI_RESULT, "gemini")
    assert (trace.route, trace.fallback_enabled) == ("gemini_backend", False)


async def test_cancel_during_self_is_not_a_fallback(fallback):
    import asyncio

    async def hang():
        await asyncio.sleep(5)

    fallback.set_self(hang)
    task = asyncio.create_task(_score_traced())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert fallback.calls["gemini"] == 0


async def test_off_hours_question_mode_grades_with_gemini_thresholds(fallback, fake_verify_steps, monkeypatch):
    # 객관식·점수 기준 없음 설정이어도, 운영 시간 밖 Gemini 답은 Gemini 기준(90/40)이다. 95점 → 성공
    monkeypatch.setattr(settings, "vlm_format", "question")
    monkeypatch.setattr(settings, "self_success_threshold", None)
    monkeypatch.setattr(pipeline_verify, "score_photo", REAL_SCORE_PHOTO)  # fake_verify_steps가 바꾼 것을 되돌린다
    fallback.clock["now"] = fallback.clock["now"].replace(hour=22)
    res = await pipeline_verify.verify_photo(_verify_request(None))
    assert (res.result, res.match_score) == ("success", 95)


# 7. 요청별 판정 기록 한 줄 (uvicorn.error 로거, JSON)

def _verify_logs(caplog) -> list[dict]:
    import json

    return [json.loads(r.getMessage().removeprefix("verify ")) for r in caplog.records if r.getMessage().startswith("verify {")]


@pytest.fixture
def full_fallback(fallback, fake_verify_steps, monkeypatch):
    # 다운로드는 가짜, 모델 판정은 실제 score_photo(폴백 흐름)로
    monkeypatch.setattr(pipeline_verify, "score_photo", REAL_SCORE_PHOTO)
    return fallback


async def test_log_for_self_answer(full_fallback, caplog):
    full_fallback.set_self(VLM_RESULT)
    with caplog.at_level("INFO", logger="uvicorn.error"):
        res = await pipeline_verify.verify_photo(_verify_request(None))
    [line] = _verify_logs(caplog)
    assert line["route"] == "self" and line["answered_by"] == "self" and line["result"] == res.result
    assert (line["fallback_enabled"], line["fallback_attempted"], line["final_status"], line["outcome"]) == (True, False, 200, "ok")
    assert line["self_sec"] is not None and line["gemini_sec"] is None and line["budget_sec"] == 20.0


async def test_log_for_fallback(full_fallback, caplog, monkeypatch):
    tags = {}
    monkeypatch.setattr(sentry_sdk, "set_tag", lambda key, value: tags.update({key: value}))
    full_fallback.set_self(HTTPException(status_code=504))
    with caplog.at_level("INFO", logger="uvicorn.error"):
        await pipeline_verify.verify_photo(_verify_request(None))
    [line] = _verify_logs(caplog)
    assert (line["route"], line["answered_by"], line["self_error_status"], line["self_last_error"]) == ("self", "gemini", 504, "ReadTimeout")
    assert (line["fallback_attempted"], line["fallback_succeeded"], line["final_status"]) == (True, True, 200)
    assert line["gemini_sec"] is not None
    assert tags["verify_answered_by"] == "gemini"


async def test_log_for_off_hours_is_not_a_fallback(full_fallback, caplog):
    full_fallback.clock["now"] = full_fallback.clock["now"].replace(hour=20)
    with caplog.at_level("INFO", logger="uvicorn.error"):
        await pipeline_verify.verify_photo(_verify_request(None))
    [line] = _verify_logs(caplog)
    assert (line["route"], line["answered_by"], line["fallback_attempted"], line["self_sec"]) == ("off_hours", "gemini", False, None)


async def test_log_when_fallback_gemini_also_fails(full_fallback, caplog):
    full_fallback.set_self(HTTPException(status_code=502))
    full_fallback.set_gemini(_api_error(500))
    with caplog.at_level("INFO", logger="uvicorn.error"), pytest.raises(HTTPException) as exc:
        await pipeline_verify.verify_photo(_verify_request(None))
    assert exc.value.status_code == 502  # 기록한 뒤 에러는 그대로 올라간다
    [line] = _verify_logs(caplog)
    assert (line["answered_by"], line["fallback_attempted"], line["fallback_succeeded"]) == (None, True, False)
    assert (line["final_status"], line["outcome"], line["result"]) == (502, "error", None)


async def test_log_for_location_mismatch_has_no_model(full_fallback, caplog, monkeypatch):
    tags = {}
    monkeypatch.setattr(sentry_sdk, "set_tag", lambda key, value: tags.update({key: value}))
    seoul = Coordinates(latitude=37.5665, longitude=126.9780)
    with caplog.at_level("INFO", logger="uvicorn.error"):
        await pipeline_verify.verify_photo(_verify_request(seoul))
    [line] = _verify_logs(caplog)
    assert (line["route"], line["answered_by"], line["reason"], line["result"]) == ("not_started", None, "location_mismatch", "fail")
    assert line["model_stage_sec"] is None and line["fallback_attempted"] is False
    assert "verify_answered_by" not in tags


async def test_log_for_unreadable_photo(full_fallback, caplog, monkeypatch):
    async def fake_fetch(image_url):
        return b"not an image"

    monkeypatch.setattr(pipeline_verify, "fetch_image", fake_fetch)
    with caplog.at_level("INFO", logger="uvicorn.error"), pytest.raises(HTTPException):
        await pipeline_verify.verify_photo(_verify_request(None))
    [line] = _verify_logs(caplog)
    assert (line["route"], line["final_status"], line["outcome"]) == ("not_started", 422, "error")


async def test_log_for_cancelled_request(full_fallback, caplog):
    import asyncio

    async def hang():
        await asyncio.sleep(5)

    full_fallback.set_self(hang)
    with caplog.at_level("INFO", logger="uvicorn.error"):
        task = asyncio.create_task(pipeline_verify.verify_photo(_verify_request(None)))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    [line] = _verify_logs(caplog)
    assert (line["outcome"], line["final_status"], line["answered_by"]) == ("cancelled", None, None)


async def test_log_has_no_secrets_and_each_request_starts_fresh(full_fallback, caplog):
    full_fallback.set_self(HTTPException(status_code=504))
    with caplog.at_level("INFO", logger="uvicorn.error"):
        await pipeline_verify.verify_photo(_verify_request(None))
        full_fallback.set_self(VLM_RESULT)
        await pipeline_verify.verify_photo(_verify_request(None))
    first, second = _verify_logs(caplog)
    assert first["fallback_attempted"] is True
    assert (second["fallback_attempted"], second["self_error_status"], second["answered_by"]) == (False, None, "self")
    text = caplog.text
    assert IMAGE_URL not in text and "my-bucket" not in text
