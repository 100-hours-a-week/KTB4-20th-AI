# 핸드오프: Semantic ID 장소 선정 모델 작업

새 Claude 세션이 이 작업을 이어받기 위한 문서다. 사실과 수치는 ADR과 로그를 가리키는 링크로만 두고 복제하지 않는다(복제하면 한쪽이 낡는다).

## 1. 변동 상태 (이 절만 자주 바뀐다)

- 마지막 갱신: 2026-10-21(#95·#97·#99는 2026-10-19까지 머지됨, 열린 PR: #109 모드 제거, #110 1안 시드, 그 위에 쌓은 응답 구조·취향 반영도 PR). 기준 커밋(`dev`)은 적지 않는다. 아래 점검 명령으로 확인한다
- 마감: 최종 발표·제출·시연은 2026-10-19(월) 빠르면, 20일(화) 늦어도
- 구현 상태: M1~M5 완료. 모델 경로(Semantic ID 생성 + 후보 제한 빔 서치)와 서빙(RunPod GPU 우선, 실패 시 규칙)이 dev에 들어가 있다. PR #109(리뷰 중, ADR-0010 Proposed)가 머지되면 2·3안은 항상 모델 경로를 먼저 시도하고 실패한 안만 규칙 변형으로 채운다. 머지 전까지 운영 기본값은 규칙(`PLACE_SELECTION_MODE=rule`)이라 사용자가 받는 일정은 규칙 결과다. 서버 반영 여부는 확인하지 못했다
- ADR 상태: 0000 Superseded, 0002~0008 Accepted, **0009 Proposed**(precheck). 0009는 ADR의 "배포 서버 확인 목록"을 실행한 뒤 Accepted로 바꾼다
- 서빙: 엔드포인트 `zspx96xu5zs7ge`, 이미지 `whale2200d/place-model-runpod:v3`(`days` 요청을 아는 이미지). 사용자가 v3 푸시와 콘솔 교체를 마쳤다고 알렸고 digest는 확인하지 못했다. Active 0, FlashBoot 끔, GPU는 Ada/Ampere 16GB 그룹. **Idle Timeout은 5초로 알고 있었으나 2026-10-19 측정 때 워커가 마지막 사용 뒤 12분 이상 내려가지 않았다(콘솔 값은 확인하지 못함).** 측정을 위해 Max Workers를 2로 열어 둔 상태이며 0으로 되돌렸는지는 확인하지 못했다(사용자가 콘솔에서 변경). **Max Workers가 0이면 엔드포인트가 일시 정지되어 모든 요청을 `409 ENDPOINT_PAUSED`로 거절한다. `/health`는 워커 0개일 때 정상 상태와 구분이 안 되므로 작은 요청 1개로 확인한다(`scripts/measure_trip_plans.py`의 사전 점검).** 운영 활성화에는 `RUNPOD_MODEL_URL`, `RUNPOD_MODEL_API_KEY`만 필요하다(`MODEL_DIR`은 비움, `PLACE_SELECTION_MODE`는 PR #109로 제거). AWS 서버에는 CPU 경로를 배포하지 않는 것으로 가정(폴백 순서 GPU → 규칙). 키 `placemodel-endpoint-1017`(Restricted)은 `.env`에만 있다
- 평가 결과: 최종용은 2026-10-16에 free 조건으로 한 번 열었다(재실행 금지). 모델 13.8%, 인기순 5.9%, 규칙 2.5%, 5개 지역 비열세 통과. 서비스 조건(top2-slots, 개발용) 모델 5.9%, 규칙 2.4%, 인기순 5.5%로 인기순 대비 차이는 없다. 자세한 수치는 ADR-0003·0005·0006
- 취향 반영 한계: 학습 데이터에서 방문이 미션 기반 취향을 거의 따르지 않고(lift 1.06), 모델의 헤더 반응이 일관되지 않다. 취향 반영은 대부분 후보 제한(ADR-0006)이 한다
- 산출물: `$AIHUB_ROOT/derived/semantic_id/v1/m3/`(places.json, visit_map.json, model_a, model_b, bundle_a, eval, runpod). `MANIFEST_m3.json`·`MANIFEST_m3b.json`에 기록. 이후 만든 `eval/dev_pref_sensitivity_model_a.json`, `eval/dev_first_place_variants_*.json`은 어떤 MANIFEST에도 없다(필요하면 `manifest --only-new`로 추가)
- 작업 트리: `TODO.md`는 사용자의 개인 파일이므로 건드리거나 커밋하지 않는다. 로컬 브랜치 정리는 사용자가 한다

### 여러 날 일정과 3개 안 (ADR-0008): 구현과 측정 완료

설계와 결정, 대안, 가정은 ADR-0008과 `.grill/place-selection-multi-day.md`가 기록하고 이 문서에는 복제하지 않는다. 측정 결과는 ADR-0008의 "준수 확인" 항목과 #95 본문이 기록한다.

- 구현 PR 머지(2026-10-19): #91(핸들러·추론 엔진 `days` 요청), #92(여행 전체 슬롯 계획과 규칙·모델 선택), #93(요청 `external_factors`와 9일 상한, 응답 `places[안][일자][장소]`, 시드, 3개 안 조립 `services/trip_orchestrator.py`). 핸드오프·설계 로그 #94 머지
- #95 머지: 측정 스크립트 `scripts/measure_trip_plans.py`와 테스트, ADR-0008 Accepted 전환, ADR-0007 Status 이력
- **응답 계약 변경(ADR-0012, Proposed, 쌓은 PR):** `data.places`는 1안의 `[일자][장소]`(2차원), `data.routes`는 2안 `PREFERENCE`와 3안 `DIVERSITY`이며 안마다 `preference_count`(낮 활동 장소 중 취향 상위 2개 카테고리에 속하는 곳), `discovery_count`(속하지 않는 곳, 보충한 곳 포함), `days`를 갖는다. 백엔드가 새 구조를 읽을 준비가 되기 전에는 배포하지 않는다. 백엔드의 준비 여부는 확인하지 못했다. 이 PR이 머지되기 전까지 아래 "3차원" 서술이 현재 운영 계약이다
- **#93이 머지됐으므로 운영 기본값이 `rule`이던 때에도 place-selection 응답은 이미 3차원이다.** BE가 새 구조를 읽는지는 확인하지 못했다
- 측정 결과 파일(레포 밖, 어떤 MANIFEST에도 없음) `$V/m3/eval/`: 유효한 것은 `trip_plans_gpu.json`(신호 없음), `trip_plans_gpu_gyeongju2.json`, `trip_plans_gpu_signal3.json`, `trip_plans_gpu_cold.json`. **`trip_plans_gpu_gyeongju.json`, `trip_plans_gpu_signal.json`, `trip_plans_gpu_signal2.json`은 엔드포인트가 일시 정지여서 모델이 한 번도 응답하지 못한 규칙 변형 결과라 쓰면 안 된다**
- 결정: 안 사이 중복 제거(ADR-0008 대안 4)는 도입하지 않는다(2026-10-19). 겹침 기준 "모든 일자"에는 엄격하게는 미달이다(`[1]` 대 `[0]` 418/420일, `[2]` 대 `[0]` 419/420일)
- 측정으로 확인하지 못한 것: 콜드 스타트는 n=1(10.3초)이고 요청 직전 워커 0개를 확인하지 못한 추정이다. 두 모델 호출이 워커 하나에서 순서대로 처리되는 것처럼 보였다(`[1]` 약 1.3초, `[2]` 약 2.7초, 신호 없음 측정에서는 둘 다 약 1.2초). 원인은 확인하지 못했다. 표본이 작다(유효 측정 요청 60건, 설문 2~11번, 5일·9일, 날씨 신호는 3·4일째 한 패턴)

남은 일
1. 사용자: RunPod 콘솔에서 Max Workers를 0으로 되돌리고 Idle Timeout 값을 확인한다. BE가 새 응답 구조(`places[안][일자][장소]`)를 읽는지 확인한다
2. 모델 경로 운영 반영(ADR-0010, PR #109). v3 배포는 끝났다. 배포 전에 서버에 `RUNPOD_MODEL_URL`·키가 있는지 확인하고, 배포 뒤 `/trips/place-selection`을 호출해 서버 로그(WARNING)에 "모델 경로 실패, 규칙 변형으로 채움"이 없는지 확인한다. 이 로그는 Sentry 이벤트가 되지 않는다
3. 열린 항목(`.grill/place-selection-multi-day.md`): 규칙 변형의 K(현재 10, 잠정), 모델 다양성 안의 3위·4위 비율(측정값 41.4~49.0%)과 헤더, `reasons.detail`·`closure_risk` 사용 여부
4. 선택: 콜드 스타트를 워커 0 확인을 엄격히 한 방식으로 재측정, 두 모델 호출의 직렬 처리 원인 확인

### precheck (ADR-0009)와 generate 묶음 호출: 머지 완료, 배포 확인 대기

- precheck(#97 머지): `POST /trips/precheck`(본문 없음). 기상청 단기·중기 예보로 5개 지역의 내일부터 9일 치 날씨 위험을 판정하고 응답은 `{status_code, data}` 구조(`data`에 `generated_at`, `forecasts`)다. 요청 때 계산하고 예보를 30분 메모리 캐시한다. 판정 기준 숫자는 운영 경험 없는 제안값이라 `app/trips/services/weather_risk.py`의 상수로 조정한다. 서버에 `KMA_API_KEY`가 없으면 precheck만 503이다. 상세와 결정은 ADR-0009
- **배포 서버 확인 목록은 ADR-0009의 "준수 확인"에 체크박스로 있고 아직 하나도 실행하지 않았다.** 사용자: 서버 환경변수에 `KMA_API_KEY`(Decoding 값) 추가, BE에 응답 구조(`data`를 꺼내 `external_factors`로 전달)·새 `detail` 값(`rain_likely`, `snow_likely`)·빠진 날짜·지역·503 처리 확인
- 확인하지 못한 것: 06시 트리거 시점의 당일 중기 발표분 수신, 경주 중기 구역 코드(`11H10000`)의 구역 이름, 기상청 일일 호출 한도, 서버가 여러 대일 때의 캐시 동작
- generate 묶음 호출(#99 머지, 작성자 `jiyoung720`): 장소를 8곳 이하 묶음으로 나눠 Gemini를 동시에 부르고(프로세스 전체 상한 3), 전체 마감 25초(`GENERATE_DEADLINE_SECONDS`)를 넘으면 504다. 검토에서 승인 가능으로 보았고 막을 문제는 없었다. 다만 동시 요청이 몰리면 상한 3 때문에 504가 늘 수 있다(Gemini 호출을 15초로 가정한 시뮬레이션에서 장소 8곳 요청이 동시에 10개 오면 3개만 성공). 서버 로그의 `generate {...}` 한 줄에서 `final_status` 504 비율과 `chunks[].wait_sec`을 보고 `MAX_CONCURRENT_CALLS`를 조정한다

### 미결·후속 작업 (우선순위 순)

1. 품질 하한을 "영업 종료만 제외"로 풀지 여부: 하한만 풀면 결과가 나빠졌다(제주항 등 교통 거점이 추천됨). 교통 거점·항구·공항 제외 시험을 먼저 하고 결정한다. 풀면 규칙 경로에도 영향이 있어 코드 사전(약 9,957곳) 재패키징, 새 이미지, 7,680건 재검증, README 표 갱신이 필요하다
2. 모델 경로를 쓰지 않기로 하면 엔드포인트 삭제, 키 폐기, Docker Hub 토큰·이미지 정리
3. 재생성 요청: 같은 요청이면 같은 일정이라 재생성이 해결되지 않는다(ADR-0007의 한계). 후속 후보는 요청에 선택 필드 `regenerate_seed` 추가(BE 협의 필요, BE가 재생성 때 무엇을 보내는지는 확인하지 못함). 요청 스키마를 다시 바꾸는 일이라 BE 협의 때 `external_factors`와 함께 정하는 것을 추천
4. 선택: 큰 모델 B를 Recall@10으로 평가(기준을 먼저 정한다: 서비스 조건 개발용 Recall@10 차이의 신뢰구간 하한 > 0), 새 평가 파일의 MANIFEST 추가
5. precheck 확장: 특보 API 연동(ADR-0009 대안 1), 판정 기준 조정, 서버가 여러 대일 때 캐시를 공유 저장소로 옮기기. 지금은 위 "precheck" 절의 배포 서버 확인이 먼저다
6. 최종 발표 자료: A4 보고서(Claude Docs 아티팩트로 만든 적 있음)와 `docs/trips/place-selection-evolution.md`, `docs/trips/semantic-id-glossary.md`를 활용

### 다음 세션 재개 방법

1. 같은 폴더(`KTB4-20th-AI`)에서 `claude`를 실행한다. 이전 대화를 이어 가려면 `claude --continue`(마지막 세션) 또는 `claude --resume`(목록에서 선택)을 쓴다. 대화가 길어 요약본으로 이어지므로, 새 세션이 낫다면 그냥 `claude`로 시작하고 아래 4번을 말한다
2. 세션이 시작되면 `CLAUDE.md`의 순서를 따른다: 이 문서 → ADR-0003~0005 → `.grill/attention-based-place-recommendation-tasks.md`의 해당 절. 여러 날 일정 작업이면 추가로 ADR-0006~0009와 `.grill/place-selection-multi-day.md`를 읽는다. 낯선 용어는 `docs/trips/semantic-id-glossary.md`, 설계 변천은 `docs/trips/place-selection-evolution.md`를 읽는다
3. 상태 점검: `git branch --show-current`, `git status --short`, `git log --oneline -3`, `gh pr list --state open`. 열린 PR이 없는지, 로컬 브랜치가 `dev`가 아니면 `dev`로 옮기고 `git pull`한 뒤 시작한다
4. 시작 때 사용자에게 줄 말 예: "handoff를 읽고 ADR-0009의 배포 서버 확인 결과와 BE 확인 결과를 점검한 뒤 모델 경로 운영 반영 결정(PR #109 머지 여부)부터 시작하자"
5. 규칙: 승인 전에는 파일 생성·브랜치 생성·푸시를 하지 않는다. 푸시와 PR 생성은 사용자가 요청할 때만 한다. 자기 PR은 머지하지 않는다. RunPod 호출은 유료이니 매번 승인을 받는다. 키는 `.env`에만 둔다
6. 환경 주의: PR #109 이후에는 `.env`에 `PLACE_SELECTION_MODE`가 남아 있으면 `Settings`가 알 수 없는 키로 거부해 앱과 테스트가 시작되지 않으니 지운다(OS 환경변수는 영향 없음). 테스트는 `tests/conftest.py`가 모델 경로를 막아 실제 RunPod을 부르지 않는다. 전체 조합 테스트(`-m exhaustive`)는 약 25분, 모델 경로 불변식 검사(`scripts/check_model_invariants.py`)는 약 17분이다. 도커 이미지 푸시, 푸시·PR 같은 공개 작업은 자동 모드 분류기가 막을 수 있으니 그때는 사용자에게 `! <명령>`으로 실행해 달라고 한다. 모델 파일을 쓰는 로컬 명령은 `uv run --with onnxruntime --with numpy python -m ...`로 돌린다. `gh pr create` 직후 리뷰어 지정이 GitHub 일시 오류로 실패한 적이 있으니 `gh pr view --json reviewRequests`로 확인한다

### 시작할 때 상태 점검 명령

```bash
git branch --show-current && git status --short   # 작업 트리와 브랜치
git log --oneline -3                              # dev 최신 커밋
gh pr list --state open                           # 열린 PR
```

로컬 DB가 10,165건인지는 `app.core.config.settings`로 `SELECT COUNT(*) FROM ai_places`를 조회해 확인한다.

## 2. 읽는 순서와 문서 우선순위

1. 이 문서
2. `docs/adr/adr-0003.md`(모델 방식), `adr-0004.md`(서빙), `adr-0005.md`(학습 데이터와 평가). 모두 Accepted다(ADR-0000은 Superseded, ADR-0006~0009도 같이 본다). ADR 상태 변경은 PR로 하고 리뷰어 머지가 승인 시점이다
3. `.grill/attention-based-place-recommendation-tasks.md`의 해당 마일스톤
4. `.grill/attention-based-place-recommendation.md`(결정, 가정, 열린 질문)

내용이 어긋나면 ADR과 `.grill` 문서가 메모리와 기억보다 우선한다. ADR은 불변이고, 결정이 바뀌면 새 ADR을 쓴다.

## 3. 작업 규칙 (사용자 규칙)

- 응답은 한국어, 표와 비유를 쓰지 않는다. 개념은 입력·출력·실제 수치로 설명한다
- 질문은 한 번에 하나씩, 추천 답변과 함께 한다. 사용자가 이미 확인한 것은 다시 묻지 않는다
- 사용자가 "대기하세요"라고 하면 문자 그대로 대기한다.
- 푸시와 PR 생성은 사용자가 명시적으로 요청할 때만 한다. 자기 PR은 머지하지 않고(리뷰어: `jiyoung720`) `gh` 계정은 `whale5500d`다
- 요청하지 않은 범위 확장과 기존 형태 변경을 하지 않는다. 필요하면 먼저 묻는다
- 확인하지 못한 것은 "확인하지 못함"으로 적고 가정으로 분리한다. 직접 잰 수치만 사실로 쓴다
- 커밋 메시지는 한국어 conventional commits(`feat(trips): …`) 형식이다. 푸시 전에 `git status`로 의도한 파일만 있는지 본다

## 4. 환경과 접근 한계

- 로컬 DB: 프로젝트 `.env`로 직접 조회할 수 있다. 계정 권한이 넓으므로 조회 전용으로 쓰고, 검증용 임시 DB는 만들었으면 반드시 지운다
- AWS 서버 DB와 RunPod은 접근할 수 없다. 서버 값이 필요하면 실행할 SQL을 사용자에게 알려 주고 결과를 받는다
- 레포 CI용 덤프(`tests/fixtures/dump.sql`)는 조회에 쓰는 컬럼만 남긴 부분집합이다. 로컬 DB와 컬럼이 다르다
- 쉘은 zsh다. `$VAR:path`는 변수 수정자로 해석되므로 `${VAR}:path`로 쓴다. 글롭이 맞지 않으면 `no matches found`로 중단된다
- macOS의 한글 경로는 유니코드 정규화 차이로 `pathlib.glob`이 못 찾는 경우가 있다. 쉘 글롭이나 `glob.glob`을 쓴다
- 외부에서 받은 파일을 읽는 Python은 `-I` 옵션으로 실행한다

## 5. 데이터와 실행 명령

AI-Hub 국내 여행로그(권역별 Training·Validation)를 `$AIHUB_ROOT`(사용자의 홈 아래 작업 폴더)에 내려받아 두었다. 라벨링 zip만 쓴다.

```
$AIHUB_ROOT/277.국내_여행로그_데이터(수도권)_Training/01-1.정식개방데이터/Training/02.라벨링데이터/TL_csv.zip.part0
$AIHUB_ROOT/277.국내_여행로그_데이터(수도권)_Validation/01-1.정식개방데이터/Validation/02.라벨링데이터/VL_csv.zip.part0
278 동부권, 279 서부권, 280 제주도_및_도서지역도 같은 구조 (폴더 번호가 권역마다 다름)
```

- `.zip.part0`는 단일 조각이어서 zip으로 바로 읽힌다
- 현재 실행 방법: `uv run python -m scripts.measure_travel_sequences --label-zip <8개 경로>`(레포 루트에서). `--aihub`는 다른 폴더 구조(`<aihub>/277_수도권/…/TL_csv.zip`)용이라 이 데이터에는 쓰지 않는다
- Training과 Validation은 같은 여행의 방문을 나눈 것이다. 합쳐야 여행이 온전해진다

### 파생 산출물 보관 위치 (잊지 말 것)

M1·M2가 만든 결과 파일은 레포에 넣지 않고 AI-Hub 원본 옆에 버전 폴더로 보관한다. 여행자 ID와 좌표가 있고 AI-Hub 약관상 재배포 가능 여부를 확인하지 못했기 때문이다.

```
$AIHUB_ROOT/derived/semantic_id/v1/
    MANIFEST.json     만든 날짜, 레포 커밋, 모델 개정, 시드, 파일별 sha256과 크기 (m1, m2, experiments)
    MANIFEST_m3.json  m3/ 파일의 같은 정보. 기존 MANIFEST는 고치지 않고 추가분은 따로 만든다
    m1/               days.jsonl(하루 단위 학습 예제), funnel.json(단계별 건수), split.json(학습/개발/최종 분할)
    m2/               embed_bgem3_nocat/, rqvae_bgem3_nocat/, code_dictionary.json, pairs_nocat.jsonl
    m3/               places.json, visit_map.json, model_a/model_b(학습 결과, ONNX), bundle_a(서빙 묶음), eval/(개발용·최종용 평가 결과), logs/
    experiments/      모델 비교와 카테고리 라벨 포함 실험의 근거(Qwen3, 라벨 포함 변형). 학습과 서비스에는 쓰지 않는다
```

- 폴더 이름의 `nocat`은 카테고리 라벨이 없는 텍스트라는 뜻이고 확정한 설계다. 라벨을 넣은 변형은 `experiments/`에 있다
- 설계를 바꿔 다시 만들 때는 `v2`처럼 새 버전 폴더를 만들고 이전 폴더는 덮어쓰지 않는다
- 폴더에 파일을 더 넣으면 `--name MANIFEST_<이름>.json --include <하위 폴더>`로 추가 MANIFEST를 만든다. `--verify`는 MANIFEST*.json을 모두 읽는다
- 무결성 확인: `uv run python -m scripts.semantic_id.manifest --version-dir "$AIHUB_ROOT/derived/semantic_id/v1" --verify`
- `split.json`은 최종용 잠금의 기준이다. 읽을 때는 `scripts/semantic_id/split.py`의 `load_part`를 쓰고, `final`은 `open_final=True`를 명시한 한 번의 평가에서만 연다
- M3 학습은 `m1/days.jsonl`, `m1/split.json`, `m2/rqvae_bgem3_nocat/`(로그 방문지의 코드), `m2/code_dictionary.json`이 필요하다(합쳐서 약 25MB). RunPod에는 이 파일만 올린다
- 다시 만드는 순서(각 스크립트의 docstring에 명령이 있다): `build_days` → `split` → `embed`(BGE-M3) → `rqvae` → `code_dict`. 정렬 검증은 `pairs`, `eval_retrieval`, `eval_codes`, 투영 측정은 `project_logs`

## 6. 용어 정의 (혼동 방지)

- 학습 문제 수: `scripts/measure_travel_sequences.py`는 길이 n인 하루에서 n−1개(시작 표시 미포함)로 센다. ADR-0005의 45,146개는 시작 표시를 포함해 방문 수와 같게 세고, 여행 메타(미션)가 있고 취향 카테고리가 1개 이상이며 서비스 5개 지역만 센 값이다
- 이름 일치의 엄격/완화: ADR-0003, 0005와 로그에 정의가 있다(엄격: 300m 안 유사도 0.8 이상, 완화: 0.6 이상 또는 한쪽 이름 포함)
- 서비스 5개 지역: 서울, 부산, 경주, 전주, 제주. 로그의 "기타"는 서비스 지역 밖(v1 학습 제외)

## 7. 예상 오류와 방지책

1. 승인 없이 푸시·머지 → 위 규칙 확인
2. 낡은 메모리 신뢰 → 문서 우선순위를 따른다
3. ADR 불변성 위반(ADR-0000 수정, Proposed를 임의로 Accepted로 전환) → 상태 변경은 PR로 한다
4. 데이터 누수: 방문 행동으로 취향 벡터 계산, AI-Hub Validation을 별도 평가셋으로 사용, 최종용을 여러 번 열람 → 모두 금지
5. 범위 이탈: 서비스 지역 밖 데이터 포함, 미매핑 미션 코드(4, 22, 24, 25, 26)나 FOOD 차원을 입력에 사용 → 제외 결정
6. 숫자 정의 혼동(위 6절)
7. 데이터 경로 실수: 폴더 번호를 `277.`으로 반복하면 6개가 없다는 오류가 난다
8. 셸·환경 함정(위 4절)
9. `ai_places.id`를 사전의 키로 사용 → 재수집하면 바뀌므로 `google_place_id`를 쓴다
10. 요청 경로에서 외부 임베딩 API를 호출하거나 VLM과 한 서버리스 워커에 합침 → ADR-0003, 0004에서 금지
11. 확인되지 않은 외부 사실을 사실처럼 씀 → 아래 열린 항목은 미확인이다

## 8. 열린 항목 (미확인 또는 미결)

- 이름 표면 차이(띄어쓰기, 지점 표기)로 정답 쌍의 38%가 같은 코드를 받지 못한다. M3 평가에서 영향이 확인되면 텍스트 정리를 재검토한다
- 여행 미션 코드가 참가자가 고른 것인지 과제로 주어진 것인지(카테고리정의서.xlsx 확보 시 확인, `TRAVEL_STYL` 정의도 같은 파일에 있을 것으로 추정)
- 연속 거리 기준 미달 시 반경 제한 R의 값, 모델 재학습 기준

## 9. 이 문서의 갱신 규칙

마일스톤이 끝날 때마다 1절(변동 상태)만 고쳐 PR에 포함한다. 나머지 절이 바뀌면 근거 ADR이나 로그와 같은 PR에서 고친다.
