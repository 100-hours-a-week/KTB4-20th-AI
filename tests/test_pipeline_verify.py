import io

import pytest
from fastapi import HTTPException
from PIL import Image

from app.core.config import settings
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
    to_grade,
)
from app.photomissions.schemas import Coordinates, DetectedLabel

ALLOWED_HOST = "my-bucket.s3.ap-northeast-2.amazonaws.com"
CHEOMSEONGDAE = Coordinates(latitude=35.8347, longitude=129.2190)


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

def test_landmark_confidence_hidden_below_threshold():
    res = build_response("success", 90, raw_landmark_confidence=LANDMARK_THRESHOLD - 0.1)
    assert res.landmark_confidence is None


def test_landmark_confidence_shown_at_threshold():
    res = build_response("success", 90, raw_landmark_confidence=LANDMARK_THRESHOLD)
    assert res.landmark_confidence == LANDMARK_THRESHOLD


def test_retry_hint_only_on_retry():
    hint = "정면에서 다시 찍어보세요"
    assert build_response("retry", 50, raw_retry_hint=hint).retry_hint == hint
    assert build_response("success", 90, raw_retry_hint=hint).retry_hint is None
    assert build_response("fail", 10, raw_retry_hint=hint).retry_hint is None


def test_location_mismatch_response():
    res = build_response("fail", match_score=0, reason="location_mismatch")
    assert res.result == "fail"
    assert res.reason == "location_mismatch"
    assert res.detected_labels == []
    assert res.landmark_confidence is None
    assert res.retry_hint is None


def test_detected_labels_passed_through():
    input_labels = [DetectedLabel(name="석탑", matched=True)]
    expected_labels = [DetectedLabel(name="석탑", matched=True)]
    result = build_response("success", 90, detected_labels=input_labels).detected_labels
    assert result == expected_labels
