from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # trips·photomissions 공통 설정. 도메인 전용 값은 각 기능 브랜치에서 이 클래스에 추가한다

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    gemini_api_key: str = ""
    internal_service_token: str = ""
    allowed_image_hosts: str = ""  # 쉼표로 구분한 허용 호스트 목록 (SSRF 방지)
    google_places_api_key: str = ""
    db_host: str = "localhost"
    db_port: int = 3306
    db_user: str = ""
    db_password: str = ""
    db_name: str = ""

    @property
    def allowed_image_hosts_list(self) -> list[str]:
        return [h.strip().lower() for h in self.allowed_image_hosts.split(",") if h.strip()]


settings = Settings()
