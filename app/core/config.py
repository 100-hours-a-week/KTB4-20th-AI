from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # trips·photomissions 공통 설정. 도메인 전용 값은 각 기능 브랜치에서 이 클래스에 추가한다

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    # 키·토큰·비밀번호·DSN과 Pod·엔드포인트 ID가 든 주소는 repr=False로 둔다. 테스트 실패나 로그에 설정 객체가 찍혀도
    # 값이 드러나지 않게 한다. repr·str만 가린다(model_dump·ValidationError 메시지에는 그대로 나온다)

    gemini_api_key: str = Field(default="", repr=False)
    internal_service_token: str = Field(default="", repr=False)
    allowed_image_hosts: str = ""  # 쉼표로 구분한 허용 호스트 목록 (SSRF 방지)
    google_places_api_key: str = Field(default="", repr=False)
    kma_api_key: str = Field(default="", repr=False)  # 기상청(공공데이터포털) 단기·중기 예보 인증키. 비어 있으면 precheck가 503 (ADR-0009)
    kma_timeout_seconds: float = 10.0  # 기상청 호출 한 번의 대기 시간
    precheck_cache_ttl_seconds: float = 1800.0  # 예보 캐시 유효 시간. 06시 트리거가 몰려도 같은 예보를 한 번만 받는다
    db_host: str = "localhost"
    db_port: int = 3306
    db_user: str = ""
    db_password: str = Field(default="", repr=False)
    db_name: str = ""
    sentry_dsn: str = Field(default="", repr=False)  # 비어 있으면 Sentry를 켜지 않음 (로컬·테스트)
    sentry_environment: str = "local"  # local / dev / prod
    sentry_traces_sample_rate: float = 0.0  # 성능 추적 비율. 0이면 끔
    git_sha: str = ""  # 배포 이미지의 커밋 SHA. CI가 빌드할 때 넣는다 (로컬은 비어 있음)
    vlm_api_key: str = Field(default="", repr=False)  # OpenAI 호환 모델 서버(RunPod 등) API 키. 키가 필요 없는 로컬 Ollama는 비워 둔다

    # verify 판정 모델. generate는 이 값과 상관없이 항상 Gemini다
    # self로 바꾸면 아래 값들이 모두 필요하다. 형식 외의 검사는 앱 시작 때 한다(측정 스크립트도 settings를 쓰므로)
    verify_backend: Literal["gemini", "self"] = "gemini"
    vlm_base_url: str = Field(default="", repr=False)  # 자체 모델 서버 주소 (OpenAI 호환, 끝의 /v1까지)
    vlm_model: str = ""  # 자체 모델 서버의 모델 이름
    vlm_deadline_seconds: float | None = None  # 판정 한 번에 자체 모델을 기다리는 전체 시간(재시도 포함)
    # 자체 모델 답 형식. score = 점수(0~100) + 아래 기준, question = 객관식 4문항 + 등급 규칙(question_format.rule_grade)
    vlm_format: Literal["score", "question"] = "score"
    # 자체 모델이 실패하거나 운영 시간 밖일 때 Gemini로 판정한다. 이 값(자체 모델 + Gemini를 합친 모델 판정 단계의
    # 전체 상한, 초)을 설정해야 폴백이 켜진다. 비우면 자체 모델 결과·에러를 그대로 돌려준다
    verify_model_budget_seconds: float | None = None
    # 자체 모델 운영 시간(한국 시간 HH:MM-HH:MM, 시작 포함·끝 제외). 밖이면 자체 모델을 부르지 않고 Gemini로 판정한다.
    # 비우면 항상 자체 모델을 먼저 부른다. 설정하면 위 예산도 필요하다. Pod를 켜고 끄는 일은 하지 않는다
    vlm_serving_hours: str = ""
    # 자체 모델용 판정 기준(score 형식만). Gemini 기준(pipeline_verify의 상수)을 조용히 쓰지 않도록 따로 받는다
    self_success_threshold: float | None = None
    self_retry_threshold: float | None = None
    self_landmark_threshold: float | None = None

    # generate 한 번의 전체 상한(초). 장소 묶음 대기·호출·재시도를 모두 포함한다. 백엔드 읽기 타임아웃(30초)보다 짧게
    generate_deadline_seconds: float = 25.0

    # 장소 선정 모델(ADR-0003). 2·3안은 항상 모델 경로로 만들고, 경로가 실패하면 그 안만 규칙 변형으로 채운다(ADR-0010)
    model_dir: str = ""  # model.onnx와 codes.json이 있는 폴더(CPU 경로)
    model_cpu_max_concurrency: int = 2
    model_first_place_top_k: int = 30  # 첫 장소를 모델 상위 K곳 중 무작위로 고른다(ADR-0007). 1이면 모델 1위로 고정
    runpod_model_url: str = Field(default="", repr=False)  # 장소 선정 모델 전용 RunPod 엔드포인트 주소(비어 있으면 GPU 경로를 쓰지 않음)
    runpod_model_api_key: str = Field(default="", repr=False)
    runpod_model_timeout_seconds: float = 20.0  # 백엔드 → AI 읽기 타임아웃(30초)보다 짧게
    model_breaker_failures: int = 3  # GPU 호출이 이만큼 연속 실패하면 잠시 막는다
    model_breaker_cooldown_seconds: float = 60.0

    @property
    def allowed_image_hosts_list(self) -> list[str]:
        return [h.strip().lower() for h in self.allowed_image_hosts.split(",") if h.strip()]


settings = Settings()
