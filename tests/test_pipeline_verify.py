import io

import pytest
from fastapi import HTTPException
from PIL import Image, PngImagePlugin

from app.core.config import settings
from app.photomissions import pipeline_verify
from app.photomissions.pipeline_verify import (
    CLEAR_MISMATCH_KM,
    LANDMARK_THRESHOLD,
    RETRY_THRESHOLD,
    SUCCESS_THRESHOLD,
    _haversine_km,
    assert_allowed_source,
    build_response,
    is_clear_mismatch,
    normalize_image,
    prepare_image,
    to_grade,
)
from app.photomissions.schemas import Coordinates, DetectedLabel

ALLOWED_HOST = "my-bucket.s3.ap-northeast-2.amazonaws.com"
CHEOMSEONGDAE = Coordinates(latitude=35.8347, longitude=129.2190)


@pytest.fixture(autouse=True)
def _gemini_backend(monkeypatch):
    # 이 파일의 모든 테스트를 Gemini 기준(90/40/60)으로 고정한다. 로컬 .env가 self여도 결과가 같다
    monkeypatch.setattr(settings, "verify_backend", "gemini")


def _jpeg_bytes(size=(10, 10), exif=None) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size).save(buf, format="JPEG", **({"exif": exif} if exif else {}))
    return buf.getvalue()


# 1. assert_allowed_source

@pytest.fixture
def allowed_host(monkeypatch):
    monkeypatch.setattr(settings, "allowed_image_hosts", ALLOWED_HOST)


def test_allowed_host_passes(allowed_host):
    assert_allowed_source(f"https://{ALLOWED_HOST}/photo.jpg?X-Amz-Signature=abc")


def test_uppercase_host_passes(allowed_host):
    assert_allowed_source(f"https://{ALLOWED_HOST.upper()}/photo.jpg")


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.com/photo.jpg",
        f"https://evil.com/?next={ALLOWED_HOST}",  # 쿼리에만 허용 호스트가 있는 경우
        f"https://{ALLOWED_HOST}@evil.com/photo.jpg",  # @ 앞은 사용자 정보라 실제 호스트는 evil.com
        f"https://{ALLOWED_HOST}.evil.com/photo.jpg",
    ],
)
def test_disallowed_host_rejected(allowed_host, url):
    with pytest.raises(HTTPException) as exc:
        assert_allowed_source(url)
    assert exc.value.status_code == 422


# 2. is_clear_mismatch

def test_haversine_same_point_is_zero():
    assert _haversine_km(CHEOMSEONGDAE, CHEOMSEONGDAE) == pytest.approx(0)


def test_haversine_one_degree_latitude():
    # 경도가 같으면 위도 1도 = 2πR/360 ≈ 111.195km
    a = Coordinates(latitude=35.0, longitude=129.0)
    b = Coordinates(latitude=36.0, longitude=129.0)
    assert _haversine_km(a, b) == pytest.approx(111.195, abs=0.01)


def test_nearby_photo_is_not_mismatch():
    nearby = Coordinates(latitude=35.8350, longitude=129.2192)  # 약 40m
    assert is_clear_mismatch(nearby, CHEOMSEONGDAE) is False


def test_far_photo_is_mismatch():
    seoul = Coordinates(latitude=37.5665, longitude=126.9780)
    assert is_clear_mismatch(seoul, CHEOMSEONGDAE) is True


def test_mismatch_boundary():
    # 위도 차이로 거리를 만들어 CLEAR_MISMATCH_KM 바로 안쪽/바깥쪽을 확인
    km_per_degree = 111.195
    inside = Coordinates(
        latitude=CHEOMSEONGDAE.latitude + (CLEAR_MISMATCH_KM - 0.1) / km_per_degree,
        longitude=CHEOMSEONGDAE.longitude,
    )
    outside = Coordinates(
        latitude=CHEOMSEONGDAE.latitude + (CLEAR_MISMATCH_KM + 0.1) / km_per_degree,
        longitude=CHEOMSEONGDAE.longitude,
    )
    assert is_clear_mismatch(inside, CHEOMSEONGDAE) is False
    assert is_clear_mismatch(outside, CHEOMSEONGDAE) is True


# 4. normalize_image

def test_normalize_shrinks_long_side_and_keeps_ratio():
    buf = io.BytesIO()
    Image.new("RGBA", (3000, 2000)).save(buf, format="PNG")
    out = Image.open(io.BytesIO(normalize_image(buf.getvalue())))
    assert out.format == "JPEG"
    assert out.mode == "RGB"
    assert out.size == (1024, 683)


def test_normalize_does_not_upscale_small_image():
    out = Image.open(io.BytesIO(normalize_image(_jpeg_bytes(size=(300, 200)))))
    assert out.size == (300, 200)


def test_normalize_applies_exif_orientation():
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation: 90도 회전해서 봐야 하는 사진
    out = Image.open(io.BytesIO(normalize_image(_jpeg_bytes(size=(200, 100), exif=exif))))
    assert out.size == (100, 200)


# 4-1. prepare_image — 서비스 전처리. 열 수 없거나 너무 큰 사진은 422, 정상 사진은 normalize_image와 같은 결과

def _image_bytes(fmt, mode="RGB", size=(64, 48), **save) -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, "red").save(buf, format=fmt, **save)
    return buf.getvalue()


def _rotated_jpeg() -> bytes:
    exif = Image.Exif()
    exif[0x0112] = 6
    return _jpeg_bytes(size=(200, 100), exif=exif)


@pytest.mark.parametrize(
    "raw",
    [_image_bytes("JPEG"), _image_bytes("PNG", "RGBA"), _image_bytes("WEBP"), _image_bytes("WEBP", "RGBA"), _rotated_jpeg()],
    ids=["jpeg", "png-rgba", "webp", "webp-rgba", "jpeg-exif-rotated"],
)
def test_prepare_keeps_normalize_output_for_supported_formats(raw):
    # 업로드 허용 형식(JPEG·PNG·WebP)은 이번 수정 전과 같은 결과가 모델로 간다
    assert prepare_image(raw) == normalize_image(raw)


def _big_text_png() -> bytes:
    info = PngImagePlugin.PngInfo()
    info.add_text("k", "a" * (2 * 1024 * 1024), zip=True)  # Pillow가 ValueError를 내는 PNG
    return _image_bytes("PNG", size=(8, 8), pnginfo=info)


def _broken_idat_png() -> bytes:
    png = bytearray(_image_bytes("PNG"))
    png[png.find(b"IDAT") + 6] ^= 0xFF
    return bytes(png)


@pytest.mark.parametrize(
    "raw",
    [b"not an image", b"", _image_bytes("JPEG")[:300], _image_bytes("WEBP")[:40], _broken_idat_png(), _big_text_png()],
    ids=["text", "empty", "cut-jpeg", "cut-webp", "broken-png-data", "png-huge-text"],
)
def test_prepare_unreadable_image_is_422(raw):
    with pytest.raises(HTTPException) as exc:
        prepare_image(raw)
    assert exc.value.status_code == 422
    assert exc.value.detail == "이미지를 열 수 없습니다"


def _must_not_decode(image_bytes):
    raise AssertionError("픽셀 상한을 넘으면 normalize_image(디코딩)를 부르면 안 된다")


def test_prepare_over_pixel_limit_is_422_before_decoding(monkeypatch):
    monkeypatch.setattr(pipeline_verify, "MAX_IMAGE_PIXELS", 99)
    monkeypatch.setattr(pipeline_verify, "normalize_image", _must_not_decode)
    with pytest.raises(HTTPException) as exc:
        prepare_image(_jpeg_bytes(size=(10, 10)))
    assert exc.value.status_code == 422
    assert exc.value.detail == "이미지 해상도가 제한을 초과했습니다"


def test_prepare_allows_exactly_the_pixel_limit(monkeypatch):
    monkeypatch.setattr(pipeline_verify, "MAX_IMAGE_PIXELS", 100)
    raw = _jpeg_bytes(size=(10, 10))
    assert prepare_image(raw) == normalize_image(raw)


def test_prepare_pillow_decompression_bomb_is_422(monkeypatch):
    # Pillow 자체 한도(기준의 2배)를 넘으면 열 때 오류가 난다. 실제 1.79억 픽셀 대신 한도를 낮춰 확인한다
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)
    monkeypatch.setattr(pipeline_verify, "normalize_image", _must_not_decode)
    with pytest.raises(HTTPException) as exc:
        prepare_image(_jpeg_bytes(size=(10, 10)))
    assert exc.value.status_code == 422
    assert exc.value.detail == "이미지 해상도가 제한을 초과했습니다"


def test_prepare_allows_pillow_warning_zone(monkeypatch):
    # 경고 구간(기준~2배)은 지금처럼 통과시킨다. 메모리 상한은 따로 정한다
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 60)
    raw = _jpeg_bytes(size=(10, 10))
    with pytest.warns(Image.DecompressionBombWarning):
        assert prepare_image(raw) == normalize_image(raw)


# 6. to_grade — 임계값은 임시값이라 숫자 대신 상수로 경계를 확인

def test_grade_success_at_threshold():
    assert to_grade(SUCCESS_THRESHOLD) == "success"
    assert to_grade(100) == "success"


def test_grade_retry_between_thresholds():
    assert to_grade(SUCCESS_THRESHOLD - 0.1) == "retry"
    assert to_grade(RETRY_THRESHOLD) == "retry"


def test_grade_fail_below_retry_threshold():
    assert to_grade(RETRY_THRESHOLD - 0.1) == "fail"
    assert to_grade(0) == "fail"


# 7. build_response

@pytest.mark.parametrize(
    "score",
    [0, RETRY_THRESHOLD - 0.1, RETRY_THRESHOLD, SUCCESS_THRESHOLD - 0.1, SUCCESS_THRESHOLD, 100],
)
def test_result_follows_score(score):
    # 등급은 점수로만 정해진다. 경계값에서도 to_grade와 같아야 한다
    assert build_response(score).result == to_grade(score)


def test_landmark_confidence_hidden_below_threshold():
    res = build_response(90, raw_landmark_confidence=LANDMARK_THRESHOLD - 0.1)
    assert res.landmark_confidence is None


def test_landmark_confidence_shown_at_threshold():
    res = build_response(90, raw_landmark_confidence=LANDMARK_THRESHOLD)
    assert res.landmark_confidence == LANDMARK_THRESHOLD


def test_retry_hint_only_on_retry():
    hint = "정면에서 다시 찍어보세요"
    assert build_response(RETRY_THRESHOLD, raw_retry_hint=hint).retry_hint == hint
    assert build_response(SUCCESS_THRESHOLD, raw_retry_hint=hint).retry_hint is None
    assert build_response(RETRY_THRESHOLD - 0.1, raw_retry_hint=hint).retry_hint is None


def test_self_backend_uses_its_own_thresholds(monkeypatch):
    # 자체 모델은 Gemini 기준(90/40/60)이 아니라 SELF_* 설정값으로 판정한다
    monkeypatch.setattr(settings, "verify_backend", "self")
    monkeypatch.setattr(settings, "self_success_threshold", 70.0)
    monkeypatch.setattr(settings, "self_retry_threshold", 30.0)
    monkeypatch.setattr(settings, "self_landmark_threshold", 50.0)
    assert to_grade(70) == "success"  # Gemini 기준이면 재시도
    assert to_grade(35) == "retry"  # Gemini 기준이면 실패
    assert to_grade(29.9) == "fail"
    assert build_response(80, raw_landmark_confidence=50).landmark_confidence == 50  # Gemini 기준이면 숨김


def test_location_mismatch_response():
    res = build_response(match_score=0, reason="location_mismatch")
    assert res.result == "fail"
    assert res.reason == "location_mismatch"
    assert res.detected_labels == []
    assert res.landmark_confidence is None
    assert res.retry_hint is None


def test_detected_labels_passed_through():
    input_labels = [DetectedLabel(name="석탑", matched=True)]
    expected_labels = [DetectedLabel(name="석탑", matched=True)]
    result = build_response(90, detected_labels=input_labels).detected_labels
    assert result == expected_labels
