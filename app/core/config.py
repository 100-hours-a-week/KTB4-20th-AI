from typing import Literal

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
    sentry_dsn: str = ""  # 비어 있으면 Sentry를 켜지 않음 (로컬·테스트)
    sentry_environment: str = "local"  # local / dev / prod
    sentry_traces_sample_rate: float = 0.0  # 성능 추적 비율. 0이면 끔
    git_sha: str = ""  # 배포 이미지의 커밋 SHA. CI가 빌드할 때 넣는다 (로컬은 비어 있음)
    vlm_api_key: str = ""  # OpenAI 호환 모델 서버(RunPod 등) API 키. 키가 필요 없는 로컬 Ollama는 비워 둔다

    # verify 판정 모델. generate는 이 값과 상관없이 항상 Gemini다
    # self로 바꾸면 아래 값들이 모두 필요하다. 형식 외의 검사는 앱 시작 때 한다(측정 스크립트도 settings를 쓰므로)
    verify_backend: Literal["gemini", "self"] = "gemini"
    vlm_base_url: str = ""  # 자체 모델 서버 주소 (OpenAI 호환, 끝의 /v1까지)
    vlm_model: str = ""  # 자체 모델 서버의 모델 이름
    vlm_deadline_seconds: float | None = None  # 판정 한 번에 자체 모델을 기다리는 전체 시간(재시도 포함)
    # 자체 모델용 판정 기준. Gemini 기준(pipeline_verify의 상수)을 조용히 쓰지 않도록 따로 받는다
    self_success_threshold: float | None = None
    self_retry_threshold: float | None = None
    self_landmark_threshold: float | None = None

    @property
    def allowed_image_hosts_list(self) -> list[str]:
        return [h.strip().lower() for h in self.allowed_image_hosts.split(",") if h.strip()]


settings = Settings()
