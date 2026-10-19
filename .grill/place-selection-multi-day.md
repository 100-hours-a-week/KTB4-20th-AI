# Grill: place-selection 여러 날 일정, 3가지 안, 외부 요인 반영
Date: 2026-10-19

## Intent
`POST /trips/place-selection`이 하루 일정 하나(`places: list[Place]`)가 아니라, 최대 8박 9일의 일정을 3가지 안으로 돌려주게 한다. 요청에 외부 요인(`external_factors`, 현재는 날씨)을 받아 날씨 신호가 있는 날에는 야외 후보를 제외한다. 이번 작업은 GPU 서버(RunPod) 호출을 기준으로 한 전처리(후보·슬롯 계획·요청 구성)와 후처리(안·일자·장소 조립)를 바꾸는 것이다.

## Constraints
- BE의 place-selection 응답 대기 시간은 30초이고 변경 가능하다
- 일정 상한은 8박 9일이며 BE와 합의했다. 응답 모양(`places[안][일자][장소]`)과 안의 인덱스 순서도 BE와 협의했다
- 운영 기본값은 아직 `PLACE_SELECTION_MODE=rule`이다. 모델 경로를 켜기 전에도 3개 안이 나와야 한다
- 학습 모델은 하루의 앞 6곳까지만 학습했다(`MAX_VISITS = 6`). 모델 입력은 하루 단위로 유지한다
- RunPod 호출은 유료이므로 호출 전에 사용자 승인을 받는다
- 서버리스 핸들러는 현재 슬롯 1~8개만 받는다(`serving/runpod/handler.py:40`)
- 이번 범위는 place-selection만이다. precheck 구현은 별도 작업이다

## Key decisions
- Decision: `external_factors`는 precheck 응답과 같은 모양(`generated_at`, `forecasts[날짜].regions[지역]`)으로 5개 지역 전체를 BE가 그대로 전달하고, AI 서버가 요청의 지역과 여행 날짜만 골라 쓴다. `has_issue`가 true인 날짜는 모든 안에서 `OUTDOOR_ACTIVITY` 제외 목록(`DIRECT_EXCLUDE_MAP`)을 후보에 적용한다. 값이 없으면 신호 없음으로 처리한다. Reason: BE에 가공 로직을 두지 않고 요청 크기도 작다. Alternative considered: BE가 지역·날짜를 잘라 보내는 안, 요청에 지역·날짜만 받고 AI 서버가 날씨를 직접 가져오는 안
- Decision: precheck 요청에서 `region`을 뺀다. 5개 지역을 모두 판정하고, 기준 시각은 요청이 들어온 날짜·시간이다. 응답 모양은 `generated_at`과 `forecasts` 구조다. 구현은 이번 범위 밖이다. Reason: 사용자 결정
- Decision: 시드는 `region`, 날짜, `members`, 그리고 이 여행의 지역·날짜에 해당하는 `has_issue` 신호만으로 계산한다. `generated_at`과 다른 지역의 신호는 시드에서 뺀다. Reason: 현재 시드가 요청 본문 전체의 해시(`router.py:64`)라서, 외부 요인이 바뀌면 같은 여행 조건에도 다른 일정이 나온다. Alternative considered: 요청 본문 전체 해시를 유지
- Decision: 3개 안은 `places[0]` = 규칙 경로(평점순, Spring이 재정렬해 최단 거리 안으로 쓰는 원본), `places[1]` = 모델 경로의 취향 안(첫 장소 시드 무작위 포함), `places[2]` = 모델 경로 + 3위·4위 카테고리를 주간 후보에 추가한 다양성 안이다. 1안과 2안은 매일 취향 상위 2개 카테고리를 그대로 쓴다. Reason: 지난주 태스크 목록(1안 Spring 거리, 2안 취향 우선 Serverless 모델, 3안 미탐색 카테고리)과 일치한다. Alternative considered: 세 안 모두 모델 경로에서 시드만 다르게 하는 안(처음에 내가 잘못 해석한 구조)
- Decision: 응답은 이름 없는 3차원 배열 `places[안][일자][장소]`이다. 일자 수는 항상 `종료일 − 시작일 + 1`이고, 날씨 제외로 후보가 모자라는 날은 해당 일자 배열이 짧아지거나 비는 것으로 가정한다. 기존 `places: list[Place]`는 대체한다. Reason: BE와 협의했다. Alternative considered: 이름 있는 객체(`plans[{course_type, days[{date, places}]}]`)
- Decision: 일자별 슬롯 구성은 첫날과 중간 날짜 모두 현재 하루 템플릿(5~6곳), 마지막 날만 MORNING·LUNCH 2곳(점심까지)이다. 시작일 = 종료일인 당일치기는 현재 종일 템플릿 5~6곳을 유지한다. Reason: 사용자 결정, 당일치기는 현재 동작 유지. Alternative considered: 도착일 오후만 / 출발일 오전만, 시각을 BE가 추가로 보내는 안
- Decision: 저녁 이후(EVENING) 슬롯은 야간 점수 조건을 만족하면 마지막 날을 뺀 모든 날에 시도하고, 후보가 소진되면 그 날은 슬롯을 제거한다(앞 일자부터 채워짐). 로컬 DB 기준 야간 후보는 서울 63, 부산 36, 제주 14, 전주 6, 경주 2곳이다. Reason: 사용자 결정. Alternative considered: 이틀에 한 번, 여행 전체 최대 N회
- Decision: 장소 중복은 한 안 안에서 여행 전체에 걸쳐 제거한다. 안 사이의 중복 제거는 하지 않는다. Reason: 안 사이 제거는 호출을 직렬로 만든다. Alternative considered: 안을 순서대로 호출하며 앞 안의 장소를 제외, 병렬 호출 뒤 겹친 슬롯만 재호출
- Decision: 안 사이 차이 기준은 일자당 최소 2곳 이상 다름이다(`[0]`과 `[1]`을 비교, EVENING 슬롯은 후보가 모자랄 수 있어 기준에서 뺀다). 구현 뒤 대표 설문 × 지역 5곳 × 여행 길이로 측정하고, 기준에 못 미치면 안 사이 중복 제거 방식을 도입한다. Reason: 첫 장소 시드만으로는 안 사이 차이를 알 수 없고, 기준은 제안값이라 측정 뒤 조정하는 것을 전제로 한다
- Decision: GPU 호출은 안마다 한 번씩(총 2번, `[1]`과 `[2]`), 호출 하나에 `days` 구조(일자별 슬롯 후보 목록)를 담는다. 핸들러가 일자마다 문맥을 초기화하고 중복 방지 집합은 여행 전체에서 공유한다. 모델 입력은 학습 때처럼 하루 단위다. Reason: 호출 수를 줄이고 학습 분포를 유지하며, 여행 전체 중복 제거를 핸들러 안에서 해결한다. Alternative considered: 안·일자마다 호출(최대 27회), 45~54슬롯을 일자 구분 없이 한 번에 생성
- Decision: 규칙 `[0]`의 DB 조회와 `[1]`, `[2]`의 후보 조회·GPU 호출을 처음부터 동시에 시작한다. GPU 호출 제한은 20초를 유지한다. Reason: 사용자 결정. 구현 뒤 전체 응답 시간을 측정해 최악이 BE 제한 30초 안에 들어오는지 본다. 모델 호출 p95가 15초를 넘으면 일자를 나눠 부르는 방식으로 재설계한다(15초는 내가 정한 제안값). Alternative considered: 처음에는 순서대로 만들고 측정 뒤 병렬화
- Decision: 모델 경로를 쓸 수 없으면(운영 기본값 `rule` 포함) 규칙 경로로 `[1]`(후보 상위 K곳에서 시드로 고른 조합)과 `[2]`(3위·4위 카테고리 추가)를 만들어 항상 3개 안을 채운다. Reason: 운영 기본값이 규칙이라 기획의 3가지 일정이 규칙 모드에서도 지켜져야 한다. Alternative considered: 빈 배열로 두기, 요청 전체 실패
- Decision: 종료일 − 시작일이 8일을 넘는 요청은 요청 검증에서 422로 거부한다. Reason: BE와 8박 9일로 합의했고, 응답 일자 수 가정을 지키기 위해서다. Alternative considered: 9일까지만 처리, 제한 없음

## Surfaced assumptions
- `Positive Signal`은 `has_issue: true`이다. 사용자 확인
- 세 안의 의미는 처음에 `E_Course_Type`(SHORTEST, PREFERENCE, DIVERSITY)과 같다고 가정했고, 이후 1안 = 규칙, 2안 = 모델 취향, 3안 = 다양성으로 정리되었다. 최단 거리는 place-selection이 아니라 Spring이 만든다
- 응답 `places`의 필드 이름은 유지한다고 이해했다(사용자가 명시적으로 이의를 제기하지 않았지만 별도로 확정하지도 않음)
- 날씨 제외로 후보가 모자라는 날은 일자 배열이 짧아지거나 빈다고 가정했다. BE가 같은 가정인지는 확인하지 못함
- 모델이 이미 소진된 야간 후보를 안 사이에서 공유하는 것은 허용한다(경주 2곳 등)
- 규칙 `[0]`을 항상 평점순 1개로 두는 것이 Spring의 재정렬 요구에 충분한지 확인하지 못함

## Open questions
- `[2]` 다양성 안에서 3위·4위 카테고리를 후보에 얼마나 섞는지(비율)
- 규칙 변형 `[1]`에서 시드로 고르는 상위 후보 수 K (구현 때 DB 후보 수를 보고 정함)
- `reasons.detail`(`heavy_rain` 등)과 `closure_risk`를 place-selection이 읽을지. 지금은 `has_issue`만 쓴다
- precheck 응답의 지역 표기(`REGION-SEOUL`)와 현재 `E_Region`(`"서울"`)의 불일치 해소 방법
- 예시 응답의 두 번째 날짜(10/12)에 지역이 2개뿐인 점. 설계에서는 없는 지역·날짜를 신호 없음으로 본다
- 여행 날짜가 예보 범위(단기·중기) 밖일 때의 처리. 설계에서는 신호 없음
- 9일치 모델 호출 한 번의 지연과 안 사이 겹침(일자당 2곳 이상 다름)의 실측. RunPod 호출이라 유료, 호출 전 승인 필요
- `[2]` 다양성 안의 품질(학습 분포 밖 후보라 측정한 적 없음)
- 병렬 실행 방식(스레드풀 등). 라우터가 동기 함수(`def place_selection`)이다
- 이 설계가 ADR-0003, 0006, 0007의 결정을 어떻게 바꾸는지. 결정이 바뀌므로 새 ADR이 필요하다(기존 ADR은 불변, Status만 갱신)
- BE가 place-selection 응답 `places`를 현재 어떻게 쓰는지(장소 수 5~6에 의존하는지)와 새 구조 협의 내용의 문서화

## Out of scope
- precheck 구현(기상청 API 연동, `heavy_rain` 등의 판정 기준, 배치, `region` 제거 코드). 이번에는 place-selection이 precheck 응답과 같은 모양을 받기만 한다
- 최단 거리 안 생성(Spring이 `[0]`을 재정렬)
- `closure_risk` 등 날씨 외 외부 요인의 반영
- 품질 하한 완화, 모델 경로 운영 활성화 결정, 큰 모델 B 평가
