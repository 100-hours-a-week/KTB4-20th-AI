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

## 전체 조합 테스트 (place-selection)

조합 테스트는 실제 수집 데이터 기반으로 place-selection 응답의 invariant를 전수 검증하는 테스트이며, CI에서는 제외되므로 슬롯 구조, 장소 조회, 회피 조건 매핑을 변경할 때 로컬에서 직접 실행해야 한다.

### 최근 검증 결과

표 1. 최근 검증 결과
| 항목 | 값 |
|---|---|
| 검증 일자 | 2026-10-01 (KST) |
| 커밋 | `cf41974` (cf41974ec78aef8e58213eb51881c9ef3bca2931) |
| 결과 | 7,680 / 7,680 통과 |
| DB 데이터 기준 | 수집 장소 10,715건 (평점 또는 리뷰 수가 NULL인 장소는 조회에서 제외) |

코드 또는 DB 데이터 변경 후 재실행하면 이 표를 갱신한다.

### 목적

모든 요청 조합에서 응답이 아래 invariant를 만족하는지 검증한다.

표 2. 응답 invariant
| 번호 | 조건 |
|---|---|
| 1 | 반환 장소 수가 5 또는 6 |
| 2 | 2번째(점심), 5번째(저녁) 장소가 FOOD |
| 3 | 회피 조건에 해당하는 type을 가진 장소 없음 |
| 4 | 장소 중복 없음 |

### 검증 범위

표 3. 조합 구성
| 차원 | 경우의 수 | 내용 |
|---|---|---|
| 지역 | 5 | E_Region 전체 |
| 회피 조건 | 64 | E_Breaker 6개의 부분집합 전체 (공집합 포함) |
| 설문 대표 요청값 | 24 | 응답값 24개(주간 1위·2위 순열 12 × 저녁 이후 슬롯 여부 2)별 대표 멤버 설문 |
| 합계 | 7,680 | |

설문 대표 요청값은 멤버 수 1~8명을 분산 배정하며, 다수 멤버 요청값은 표준편차 페널티를 적용해야만 기대 순위가 나오도록 구성되어 있다.

### 실행 조건

- 실제 수집 데이터(`ai_places`, `ai_place_categories`, `ai_place_types`)가 적재된 DB가 필요하다. 빈 DB에서는 invariant 1이 항상 실패한다.
- 전체 실행에 약 25~30분이 소요된다.
- `pyproject.toml`의 `addopts`에서 `exhaustive` marker를 기본 제외하므로, 옵션 없는 `pytest` 실행과 CI에서는 실행되지 않는다.

### 실행 방법

표 4. 실행 명령
| 목적 | 명령 |
|---|---|
| 전체 실행 | 전체 조합 테스트 실행 (`uv run pytest -m exhaustive`) |
| 결과 파일 저장 | 실패 정보를 한 줄씩 파일로 저장 (`uv run pytest -m exhaustive --tb=line > combination_result.txt 2>&1`) |
| 실패 건만 재실행 | 직전 실패 건만 실행하고, 기록이 없으면 실행 안 함 (`uv run pytest -m exhaustive --lf --lfnf=none --tb=line`) |

- 실패 건 재실행 시에도 `-m exhaustive`를 지정해야 한다. 지정하지 않으면 `addopts`에 의해 조합 테스트가 제외된다.
- 실행 중단은 Ctrl+C로 한다. 강제 종료하면 결과 요약이 기록되지 않는다.

### 실행 시점

- 슬롯 구조 변경 (`slot_plan.py`)
- 장소 조회 및 후보 부족 처리 변경 (`place_selection.py`)
- 회피 조건 매핑 변경 (`DIRECT_EXCLUDE_MAP`)
- 수집 데이터 재적재

### 대표 요청값 재생성

그룹 점수 공식(가중치 포함), 설문 문항 구조, 슬롯 배치 규칙이 바뀌면 대표 요청값을 재생성한다 (`uv run python -m tests.trips.tools.generate_representative_surveys`).

생성 스크립트는 검증 대상 코드를 import하지 않고 그룹 점수 공식을 독립 구현한다. 서비스의 공식이 바뀌면 스크립트의 공식도 함께 수정해야 한다.
