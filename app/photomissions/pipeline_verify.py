import io
import math
from urllib.parse import urlparse

from fastapi import HTTPException, status
from PIL import Image, ImageOps

from app.core.config import settings
from app.photomissions.schemas import Coordinates

_GPS_IFD_TAG = 0x8825  # EXIF 표준 태그 번호
_GPS_LAT_REF, _GPS_LAT, _GPS_LON_REF, _GPS_LON = 1, 2, 3, 4  # GPS IFD 내부 태그 번호

_EARTH_RADIUS_KM = 6371.0
CLEAR_MISMATCH_KM = 5.0  # TODO: 2단계 평가셋으로 확정 전 임시값

_MAX_DIMENSION = 1024  # TODO: baseline 측정(2단계) 후 토큰·정확도 트레이드오프로 조정
_JPEG_QUALITY = 85


def assert_allowed_source(image_url: str) -> None:
    # 1. 입력 검증 - 허용된 호스트가 아니면 거부한다 (SSRF 방지)
    host = urlparse(image_url).hostname
    if host not in settings.allowed_image_hosts_list:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="허용되지 않은 이미지 호스트입니다",
        )


def _dms_to_decimal(dms: tuple[float, float, float], ref: str) -> float:
    # 도/분/초(degrees, minutes, seconds) 표기를 십진수 좌표로 변환
    degrees, minutes, seconds = dms
    decimal = degrees + minutes / 60 + seconds / 3600
    return -decimal if ref in ("S", "W") else decimal


def extract_gps(image_bytes: bytes) -> Coordinates | None:
    # 3. EXIF에서 GPS 좌표를 추출한다. 없으면 None 
    image = Image.open(io.BytesIO(image_bytes))
    exif = image.getexif()
    gps_ifd = exif.get_ifd(_GPS_IFD_TAG)

    lat = gps_ifd.get(_GPS_LAT)
    lat_ref = gps_ifd.get(_GPS_LAT_REF)
    lon = gps_ifd.get(_GPS_LON)
    lon_ref = gps_ifd.get(_GPS_LON_REF)
    if not (lat and lat_ref and lon and lon_ref):
        return None

    return Coordinates(
        latitude=_dms_to_decimal(lat, lat_ref),
        longitude=_dms_to_decimal(lon, lon_ref),
    )


def _haversine_km(a: Coordinates, b: Coordinates) -> float:
    # 두 좌표 사이의 직선거리(km). 지구를 구로 근사하는 Haversine 공식.
    lat1, lon1 = math.radians(a.latitude), math.radians(a.longitude)
    lat2, lon2 = math.radians(b.latitude), math.radians(b.longitude)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(h))


def is_clear_mismatch(exif_gps: Coordinates | None, place_coordinates: Coordinates) -> bool:
    # 4. 위치 사전 판정 - GPS가 없으면 판정 불가이므로 통과시킨다(사진 자체는 VLM이 봄)
    if exif_gps is None:
        return False
    return _haversine_km(exif_gps, place_coordinates) > CLEAR_MISMATCH_KM


def normalize_image(image_bytes: bytes) -> bytes:
    # 5. 이미지 전처리 - 리사이즈·포맷 정규화로 입력 크기 축소 (prefill 병목 완화)
    image = Image.open(io.BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image)  # EXIF 방향을 실제 픽셀에 반영 (세로사진이 눕는 것 방지)
    image = image.convert("RGB")  # PNG 투명배경 등도 JPEG로 통일

    image.thumbnail((_MAX_DIMENSION, _MAX_DIMENSION), Image.LANCZOS)  # 비율 유지, 더 작으면 확대 안 함

    output = io.BytesIO()
    image.save(output, format="JPEG", quality=_JPEG_QUALITY)
    return output.getvalue()
