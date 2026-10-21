# ADR 목차

| 링크 | 제목 | 상태 |
|---|---|---|
| [ADR-0000](adr-0000.md) | 장소 선정을 LLM 없이 설문 점수 기반 결정론 규칙과 DB 조회로 구현 | Superseded (by ADR-0003) |
| [ADR-0001](adr-0001.md) | 장소 후보를 hex grid로 사전 수집한 자체 DB에서 조회 | Accepted |
| [ADR-0002](adr-0002.md) | 그룹 취향 점수를 평균에서 표준편차 페널티를 뺀 값으로 통합 | Accepted |
| [ADR-0003](adr-0003.md) | 장소 선정을 Semantic ID 생성 모델과 후보 제한 디코딩으로 구현 | Accepted |
| [ADR-0004](adr-0004.md) | 모델 서빙을 RunPod 서버리스 GPU와 컨테이너 CPU 폴백의 두 경로로 구성 | Accepted |
| [ADR-0005](adr-0005.md) | 장소 선정 모델의 학습 데이터와 평가를 AI-Hub 여행로그의 여행자 단위 분할로 구성 | Accepted |
| [ADR-0006](adr-0006.md) | 모델 경로에서 FOOD가 아닌 슬롯의 후보를 취향 상위 두 카테고리로 제한 | Accepted |
| [ADR-0007](adr-0007.md) | 모델 경로의 첫 장소를 모델 상위 30곳 중 요청 기반 시드 무작위로 선택 | Accepted |
| [ADR-0008](adr-0008.md) | place-selection 응답을 안·일자·장소의 3차원 배열로 확장하고 안마다 규칙 경로 1번과 모델 경로 2번으로 구성 | Accepted |
| [ADR-0009](adr-0009.md) | precheck를 기상청 단기·중기 예보로 5개 지역의 9일 치 날씨 위험을 요청 시 계산하고 캐시하는 엔드포인트로 구현 | Proposed |
| [ADR-0010](adr-0010.md) | 2·3안을 항상 모델 경로로 만들고 장소 선정 모드 전환을 제거하되 실패한 안은 규칙 변형으로 채움 | Proposed |
| [ADR-0011](adr-0011.md) | 1안을 평점 정렬 상위 10곳 중 요청 기반 시드로 선택 | Proposed |
| [ADR-0012](adr-0012.md) | place-selection 응답을 1안 places와 2·3안 routes로 나누고 routes에 낮 활동 장소의 취향 반영도를 담음 | Proposed |
