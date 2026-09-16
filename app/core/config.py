from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """trips·photomissions 공통 설정. 도메인 전용 값(예: photomissions의
    ALLOWED_IMAGE_HOSTS)은 각 기능 브랜치에서 이 클래스에 추가한다."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    gemini_api_key: str = ""
    internal_service_token: str = ""


settings = Settings()
