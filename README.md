# KTB4-20th-AI

카카오테크 부트캠프 20조 AI 개발

## 실행 방법

의존성 관리는 [uv](https://docs.astral.sh/uv/)를 쓴다.

```bash
# 1. 의존성 설치 (uv가 알아서 가상환경도 만듦)
uv sync

# 2. 환경변수 설정
cp .env.example .env
# .env 열어서 실제 값 채우기 (GEMINI_API_KEY 등)

# 3. 서버 실행
uv run uvicorn app.main:app --reload

# 확인: http://127.0.0.1:8000/health -> {"status": "ok"}
```

## 디렉토리 구조

```
app/
├── main.py            # FastAPI 앱 진입점
├── core/               # 도메인 공통(설정 등)
├── photomissions/      # 포토미션 도메인
└── trips/               # 장소 추천·동선 도메인
```

## 개발 도구

```bash
uv run pytest    # 테스트
uv run ruff check .    # 린트
```
