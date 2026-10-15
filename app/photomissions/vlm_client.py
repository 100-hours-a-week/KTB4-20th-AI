# 자체 모델(OpenAI 호환 서버, RunPod vLLM 등) 호출을 한 곳에 모은다. VERIFY_BACKEND=self일 때만 쓴다

from urllib.parse import urlparse

from app.core.config import Settings


def settings_errors(s: Settings) -> list[str]:
    # self 판정에 필요한 설정이 빠지거나 틀린 것을 모두 모은다. 하나라도 있으면 앱이 시작되지 않는다
    # 서버는 켜졌는데 판정마다 502가 나는 것보다, 배포할 때 바로 멈추는 편이 원인을 찾기 쉽다
    errors = []
    url = urlparse(s.vlm_base_url.strip())
    if url.scheme not in ("http", "https") or not url.hostname:
        errors.append("VLM_BASE_URL은 http(s)://로 시작하는 서버 주소여야 함")
    if not s.vlm_model.strip():
        errors.append("VLM_MODEL이 비어 있음")
    if s.vlm_deadline_seconds is None or s.vlm_deadline_seconds <= 0:
        errors.append("VLM_DEADLINE_SECONDS는 0보다 커야 함")

    success, retry, landmark = s.self_success_threshold, s.self_retry_threshold, s.self_landmark_threshold
    if success is None or retry is None or landmark is None:
        # Gemini 기준(90/40/60)은 Pro로 정한 값이라 자체 모델에 그대로 쓰면 판정 의미가 달라진다
        errors.append("SELF_SUCCESS_THRESHOLD·SELF_RETRY_THRESHOLD·SELF_LANDMARK_THRESHOLD가 모두 필요함")
    else:
        if not 0 <= retry <= success <= 100:
            errors.append("판정 기준은 0 ≤ SELF_RETRY_THRESHOLD ≤ SELF_SUCCESS_THRESHOLD ≤ 100이어야 함")
        if not 0 <= landmark <= 100:
            errors.append("SELF_LANDMARK_THRESHOLD는 0~100이어야 함")
    return errors


def check_settings(s: Settings) -> None:
    if errors := settings_errors(s):
        raise RuntimeError("VERIFY_BACKEND=self 설정 오류: " + " / ".join(errors))


def backend_summary(s: Settings) -> str:
    # 서버를 켤 때 어느 모델로 판정하는지 한 줄로 보여 준다. 키와 서버 주소(Pod ID가 들어 있음)는 넣지 않는다
    if s.verify_backend != "self":
        return "verify 판정 백엔드: gemini"
    return (
        f"verify 판정 백엔드: self (모델 {s.vlm_model}, 마감 {s.vlm_deadline_seconds:g}초, "
        f"기준 성공 {s.self_success_threshold:g} / 재시도 {s.self_retry_threshold:g} / "
        f"랜드마크 {s.self_landmark_threshold:g})"
    )
