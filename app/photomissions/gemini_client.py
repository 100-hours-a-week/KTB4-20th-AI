# Gemini SDK 호출을 한 곳에 모은다 (provider 교체 시 영향 범위 한정)

# TODO: generate_content 호출 시그니처는 실제 SDK로 검증 전이라 확정 아님.

import functools

from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from app.core.config import settings

# LLM API 모델 — 실제 API로 확인함(client.models.list())
MODEL_NAME = "gemini-3.1-pro-preview"

MAX_ATTEMPTS = 2  # 최초 1회 + 재시도 1회


@functools.lru_cache(maxsize=1)
def _get_client() -> genai.Client:
    # generate_structured가 실제 호출될 때 처음 만듦 — import 시점엔 안 만듦(API 키 없어도 import는 되게)
    return genai.Client(api_key=settings.gemini_api_key)


async def generate_structured[T: BaseModel](
    system: str,
    user: types.ContentListUnion,
    response_schema: type[T],
) -> T:
    # 구조화 출력 호출. user는 텍스트 하나 또는 [텍스트, 이미지] 리스트
    # 실패(예외 또는 스키마 불일치) 시 1회 재시도, 그래도 실패하면 마지막 에러를 그대로 올림

    config = types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        response_schema=response_schema,
    )
    last_error: Exception = ValueError("generate_structured: 알 수 없는 실패")
    for _ in range(MAX_ATTEMPTS):
        try:
            response = await _get_client().aio.models.generate_content(
                model=MODEL_NAME, contents=user, config=config,
            )
        except errors.APIError as e:
            if e.code == 429:  # 할당량 초과 - 바로 재시도해도 또 막히므로 즉시 포기
                raise
            last_error = e
            continue
        except Exception as e:  # noqa: BLE001 — 재시도 대상이라 SDK 예외 종류를 가리지 않고 잡음
            last_error = e
            continue
        if isinstance(response.parsed, response_schema):
            return response.parsed
        last_error = ValueError("Gemini 응답이 스키마와 맞지 않음")
    raise last_error
