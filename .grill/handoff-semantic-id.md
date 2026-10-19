# 핸드오프: Semantic ID 장소 선정 모델 작업

새 Claude 세션이 이 작업을 이어받기 위한 문서다. 사실과 수치는 ADR과 로그를 가리키는 링크로만 두고 복제하지 않는다(복제하면 한쪽이 낡는다).

## 1. 변동 상태 (이 절만 자주 바뀐다)

- 마지막 갱신: 2026-10-19(PR #86 머지 후, 다음 작업 전환 시점). 기준 커밋(`dev`)은 적지 않는다. 아래 점검 명령으로 확인한다
- 마감: 최종 발표·제출·시연은 2026-10-19(월) 빠르면, 20일(화) 늦어도
- 구현 상태: M1~M5 완료. 모델 경로(Semantic ID 생성 + 후보 제한 빔 서치)와 서빙(RunPod GPU 우선, 실패 시 규칙)이 dev에 들어가 있다. 운영 기본값은 아직 규칙(`PLACE_SELECTION_MODE=rule`)이라 사용자가 받는 일정은 규칙 결과다
- ADR 상태: 0000 Superseded, 0002~0007 Accepted(0007 첫 장소 무작위는 PR #86 머지로 확정)
- **PR #86 머지 완료**(2026-10-19): 모델 경로의 첫 장소를 모델 상위 30곳 중 요청 해시 시드로 무작위 선택(ADR-0007). 전체 7,680건 불변식 위반 0, 이미지 v2 GPU 확인
- **#86 리뷰 질문과 답변**(확인함, 2026-10-19 PR 코멘트): 리뷰어 `jiyoung720`의 "같은 그룹이 같은 조건으로 재생성하면 똑같은 일정이 나오나요?"에 `whale5500d`가 "입력으로 들어가는 요청값이 완전히 같으면 똑같은 일정이 생성됩니다."라고 답했고(00:07:30 UTC) 15초 뒤 머지했다. 사실: 시드가 요청 본문 전체의 해시라서 요청이 같으면 일정이 같다. 이 변경이 늘린 것은 서로 다른 요청 사이의 다양성이고 재생성 자체는 해결하지 못한다. 후속 후보: 요청에 선택 필드 `regenerate_seed` 추가(BE 협의 필요, BE가 재생성 때 무엇을 보내는지는 확인하지 못함)
- 서빙: 엔드포인트 `zspx96xu5zs7ge`, 이미지 `whale2200d/place-model-runpod:v2`(digest `sha256:b683…a9c4`; v1은 `sha256:21c0e171…92a6`이고 `first_k`·`seed`를 무시함). Active 0, Idle 5초, FlashBoot 끔, GPU는 Ada/Ampere 16GB 그룹. 확인 후 Max Workers를 0으로 되돌려 일시 정지 상태다(사용자가 콘솔에서 변경, 내가 직접 확인하지는 못함). 다시 쓰려면 Max를 1~2로 올린다. 운영 활성화에는 `PLACE_SELECTION_MODE=model`, `RUNPOD_MODEL_URL`, `RUNPOD_MODEL_API_KEY`만 필요하다(`MODEL_DIR`은 비움). 키 `placemodel-endpoint-1017`(Restricted)은 `.env`에만 있다. AWS 서버에는 CPU 경로를 배포하지 않는 것으로 가정(폴백 순서 GPU → 규칙)
- 평가 결과: 최종용은 2026-10-16에 free 조건으로 한 번 열었다(재실행 금지). 모델 13.8%, 인기순 5.9%, 규칙 2.5%, 5개 지역 비열세 통과. 서비스 조건(top2-slots, 개발용) 모델 5.9%, 규칙 2.4%, 인기순 5.5%로 인기순 대비 차이는 없다(+0.4%p, 구간 −0.3~+1.2). 자세한 수치는 ADR-0003·0005·0006
- 취향 반영 한계: 학습 데이터에서 방문이 미션 기반 취향을 거의 따르지 않고(lift 1.06), 모델의 헤더 반응이 일관되지 않다. 취향 반영은 대부분 후보 제한(ADR-0006)이 한다
- 산출물: `$AIHUB_ROOT/derived/semantic_id/v1/m3/`(places.json, visit_map.json, model_a, model_b, bundle_a, eval, runpod). `MANIFEST_m3.json`·`MANIFEST_m3b.json`에 기록. 이후 만든 `eval/dev_pref_sensitivity_model_a.json`, `eval/dev_first_place_variants_*.json`은 어떤 MANIFEST에도 없다(필요하면 `manifest --only-new`로 추가)
- 작업 트리: `TODO.md`는 사용자의 개인 파일이므로 건드리거나 커밋하지 않는다. 로컬 브랜치 정리는 사용자가 한다

### 다음 작업 후보: 여러 날 일정 지원 (사용자 담당: 서버리스 결과를 AI 서버에서 가공해 BE에 전달)

사용자 설명(그대로): 서버리스로 서빙받은 값을 AI 서버에서 가공해 백엔드 서버로 제공하는 작업이고, 바로 전달하면 5~6개 장소만 제공하며 하루 치 일정이므로 전체 일정이 길어지면 더 많은 장소가 필요하다. 설계부터 시작하며(grill-me) 사용자의 승인 전에는 코드를 쓰지 않는다. 아래는 코드로 확인한 사실과 확인하지 못한 것이다.

코드로 확인한 사실 (2026-10-19)
- 5~6곳은 서버리스 모델의 한계가 아니라 AI 서버의 슬롯 계획이 정한다. `slot_plan.build_slot_plan`이 하루 슬롯을 MORNING, LUNCH, AFTERNOON_FIRST, AFTERNOON_SECOND, DINNER 5개로 만들고, 야간 ACTIVITY 문항 점수가 `EVENING_SLOT_THRESHOLD`를 넘으면 EVENING을 더해 6개로 만든다. 서버리스 핸들러는 받은 슬롯 후보 목록마다 장소 하나를 돌려줄 뿐이다
- 응답 `Place_Selection_Response_Data`는 `places: list[Place]` 하나이고, `Place`에는 일자나 시간대 필드가 없다(`selected_for`, `matched_preferences`와 구글 장소 데이터뿐). 일자를 나타내려면 응답 스키마가 바뀐다
- 요청 `Place_Selection_Request`는 `region`, `start_date`, `end_date`, `members`(1~8명)다. 날짜는 검증(시작일이 과거가 아님, 종료일이 시작일 이전이 아님)에만 쓰이고, `app/trips`의 장소 선정 코드에서는 쓰이지 않는다. 시드 해시에만 요청 본문의 일부로 들어간다
- `app/trips/router.py`의 `place_selection` docstring TODO에 하루 단위 함수를 날짜별로 반복 호출하면 생기는 문제가 적혀 있다: (1) 매일 같은 카테고리 배치 (2) 매일 같은 장소 선택 (3) 첫날·마지막 날 구분 없음 (4) 저녁 이후 슬롯 매일 생성. 필요한 추가 요소: 일자 유형별 템플릿, 일자 간 카테고리 분배 규칙(기획 결정 선행), 여행 전체 장소 중복 제거, 저녁 이후 슬롯 빈도 규칙
- 모델 학습은 하루의 앞 6곳까지만 썼다(`scripts/semantic_id/train_data.py`의 `MAX_VISITS = 6`, 로그의 하루는 최대 23곳). 6곳을 넘는 길이를 한 번에 만든 검증은 없다
- 시드는 요청 본문 전체의 해시라서, 같은 요청으로 날짜별 호출을 반복하면 일자마다 같은 시드가 된다. 일자별로 다른 결과를 내려면 일자 정보를 시드에 넣거나 일자별 시드를 파생해야 한다
- 측정값(ADR-0004): 따뜻한 상태 호출당 약 1.3초(p95 1.43초), 동시 4요청 p95 5.46초(워커 1개), 유휴 후 첫 요청 10~13초, 요청 크기 15MB까지 성공. GPU 호출 제한 20초, 연속 3회 실패 시 60초 차단. 일자별로 호출하면 호출 횟수가 일수만큼 늘어난다(일수가 늘 때의 총 지연은 측정하지 않음)

확인하지 못한 것
- BE가 현재 응답을 어떻게 쓰는지(일자 배열을 받을 수 있는지, 장소 수 5~6에 의존하는지)
- 여행 일수의 상한과 BE가 기대하는 장소 수
- 중복 방지가 여행 전체로 필요한지의 기획 결정, 일자 간 카테고리 분배 규칙
- 한 요청에 여러 날의 슬롯 후보를 담는 방식의 동작(구현·측정한 적 없음)

### 미결·후속 작업 (우선순위 순)

1. 여러 날 일정 지원 설계(위). 재생성 필드(`regenerate_seed`)는 같은 요청 스키마를 다루므로 이 설계와 함께 BE에 협의하는 것을 추천
2. 품질 하한을 "영업 종료만 제외"로 풀지 여부: 하한만 풀면 결과가 나빠졌다(제주항 등 교통 거점이 추천됨). 교통 거점·항구·공항 제외 시험을 먼저 하고 결정한다. 풀면 규칙 경로에도 영향이 있어 코드 사전(약 9,957곳) 재패키징, 새 이미지, 7,680건 재검증, README 표 갱신이 필요하다
3. 모델 경로 운영 활성화 결정과 환경변수 배포. 쓰지 않으면 엔드포인트 삭제, 키 폐기, Docker Hub 토큰·이미지 정리
4. 선택: 큰 모델 B를 Recall@10으로 평가(기준을 먼저 정한다: 서비스 조건 개발용 Recall@10 차이의 신뢰구간 하한 > 0), 새 평가 파일의 MANIFEST 추가
5. 최종 발표 자료: A4 보고서(Claude Docs 아티팩트로 만든 적 있음)와 `docs/trips/place-selection-evolution.md`, `docs/trips/semantic-id-glossary.md`를 활용

### 다음 세션 재개 방법

1. 같은 폴더(`KTB4-20th-AI`)에서 `claude`를 실행한다. 이전 대화를 이어 가려면 `claude --continue`(마지막 세션) 또는 `claude --resume`(목록에서 선택)을 쓴다. 대화가 길어 요약본으로 이어지므로, 새 세션이 낫다면 그냥 `claude`로 시작하고 아래 4번을 말한다
2. 세션이 시작되면 `CLAUDE.md`의 순서를 따른다: 이 문서 → ADR-0003~0005 → `.grill/attention-based-place-recommendation-tasks.md`의 해당 절. 낯선 용어는 `docs/trips/semantic-id-glossary.md`, 설계 변천은 `docs/trips/place-selection-evolution.md`를 읽는다
3. 상태 점검: `git branch --show-current`, `git status --short`, `git log --oneline -3`, `gh pr list --state open`. #86은 머지됐다. 로컬 브랜치가 `dev`가 아니면 `dev`로 옮기고 `git pull`한 뒤 시작한다. `docs/handoff-multi-day` 브랜치가 남아 있으면 PR 머지 여부를 본다
4. 시작 때 사용자에게 줄 말 예: "handoff를 읽고 여러 날 일정 지원 설계부터 시작하자"
5. 규칙: 승인 전에는 파일 생성·브랜치 생성·푸시를 하지 않는다. 푸시와 PR 생성은 사용자가 요청할 때만 한다. 자기 PR은 머지하지 않는다. RunPod 호출은 유료이니 매번 승인을 받는다. 키는 `.env`에만 둔다
6. 환경 주의: `.env`의 `PLACE_SELECTION_MODE`는 테스트 전에 `rule`이어야 한다(`PLACE_SELECTION_MODE=rule uv run pytest -q`). 도커 이미지 푸시, 푸시·PR 같은 공개 작업은 자동 모드 분류기가 막을 수 있으니 그때는 사용자에게 `! <명령>`으로 실행해 달라고 한다. 모델 파일을 쓰는 로컬 명령은 `uv run --with onnxruntime --with numpy python -m ...`로 돌린다


### 시작할 때 상태 점검 명령

```bash
git branch --show-current && git status --short   # 작업 트리와 브랜치
git log --oneline -3                              # dev 최신 커밋이 위 기준 커밋과 맞는지
gh pr list --state open                           # 열린 PR
```

로컬 DB가 10,165건인지는 `app.core.config.settings`로 `SELECT COUNT(*) FROM ai_places`를 조회해 확인한다.

## 2. 읽는 순서와 문서 우선순위

1. 이 문서
2. `docs/adr/adr-0003.md`(모델 방식), `adr-0004.md`(서빙), `adr-0005.md`(학습 데이터와 평가). 모두 Accepted다(ADR-0000은 Superseded, ADR-0006·0007도 같이 본다). 아래는 전환 당시의 조건 기록이다. 전환 조건: ADR-0005는 M1과 개발용 평가 구현 PR이 머지되고 분할·누수 방지 테스트가 통과할 때, ADR-0003은 최종용 평가를 통과할 때(통과 시 ADR-0000을 Superseded로), ADR-0004는 벤치마크로 기본 경로를 정할 때. 상태 변경은 PR로 하고 리뷰어 머지가 승인 시점이다
3. `.grill/attention-based-place-recommendation-tasks.md`의 해당 마일스톤
4. `.grill/attention-based-place-recommendation.md`(결정, 가정, 열린 질문)

내용이 어긋나면 ADR과 `.grill` 문서가 메모리와 기억보다 우선한다. ADR은 불변이고, 결정이 바뀌면 새 ADR을 쓴다.

## 3. 작업 규칙 (사용자 규칙)

- 응답은 한국어, 표와 비유를 쓰지 않는다. 개념은 입력·출력·실제 수치로 설명한다
- 질문은 한 번에 하나씩, 추천 답변과 함께 한다. 사용자가 이미 확인한 것은 다시 묻지 않는다
- 사용자가 "대기하세요"라고 하면 문자 그대로 대기한다. M1은 사용자의 명시적 승인 전에 착수하지 않는다
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

## 6. M1 구현 사양 (확정)

- 코드 위치: `scripts/semantic_id/` 아래에 기능별로 나눈다(로더, 시퀀스, 취향 벡터, 분할). 기존 `scripts/` 관례를 따른다
- 산출물: 레포 밖 경로를 `--out-dir`로 받고 JSONL로 저장한다. 분할 결과 파일은 한 번 만들면 덮어쓰지 않는다(`scripts/split_eval.py` 방식)
- 입력: `--data-root <폴더>`(없으면 환경 변수 `AIHUB_ROOT`)로 폴더를 스캔한다. 시작 전에 8개 zip이 모두 있는지 확인하고 없으면 목록을 출력하고 종료한다. `.zip`과 `.zip.part0` 모두 허용하고 `zipfile.is_zipfile`로 검증한다. `--label-zip`은 부분 실행과 재현용으로 유지한다. 기존 `--aihub`는 건드리지 않는다
- 결정 근거와 범위는 ADR-0005와 과제 목록의 T1-1~T1-8을 따른다

## 7. 용어 정의 (혼동 방지)

- 학습 문제 수: `scripts/measure_travel_sequences.py`는 길이 n인 하루에서 n−1개(시작 표시 미포함)로 센다. ADR-0005의 45,146개는 시작 표시를 포함해 방문 수와 같게 세고, 여행 메타(미션)가 있고 취향 카테고리가 1개 이상이며 서비스 5개 지역만 센 값이다
- 이름 일치의 엄격/완화: ADR-0003, 0005와 로그에 정의가 있다(엄격: 300m 안 유사도 0.8 이상, 완화: 0.6 이상 또는 한쪽 이름 포함)
- 사용자의 번호 체계(과거 대화의 A1, B1…, 안 1~3)와 Claude의 "질문 N" 번호는 다르다. 이름으로 부른다
- 서비스 5개 지역: 서울, 부산, 경주, 전주, 제주. 로그의 "기타"는 서비스 지역 밖(v1 학습 제외)

## 8. 예상 오류와 방지책

1. 승인 없이 M1 착수·푸시·머지 → 위 규칙 확인
2. 낡은 메모리 신뢰 → 문서 우선순위를 따른다
3. ADR 불변성 위반(ADR-0000 수정, Proposed를 임의로 Accepted로 전환) → 전환 조건은 2절을 따르고, 상태 변경은 PR로 한다
4. 데이터 누수: 방문 행동으로 취향 벡터 계산, AI-Hub Validation을 별도 평가셋으로 사용, 최종용을 여러 번 열람 → 모두 금지
5. 범위 이탈: 서비스 지역 밖 데이터 포함, 미매핑 미션 코드(4, 22, 24, 25, 26)나 FOOD 차원을 입력에 사용 → 제외 결정
6. 숫자 정의 혼동(위 7절)
7. 데이터 경로 실수: 폴더 번호를 `277.`으로 반복하면 6개가 없다는 오류가 난다
8. 셸·환경 함정(위 4절)
9. `ai_places.id`를 사전의 키로 사용 → 재수집하면 바뀌므로 `google_place_id`를 쓴다
10. 요청 경로에서 외부 임베딩 API를 호출하거나 VLM과 한 서버리스 워커에 합침 → ADR-0003, 0004에서 금지
11. 표·비유를 쓰거나 한 번에 여러 질문 → 위 규칙
12. 확인되지 않은 외부 사실을 사실처럼 씀 → 아래 열린 항목은 미확인이다

## 9. 열린 항목 (미확인 또는 미결)

- 로그 방문지를 DB에 비슷한 장소가 없다고 보아 정답에서 빼는 임계값: DB 장소끼리 가장 가까운 유사도의 하위 5%(0.607)로 확정(방문의 11.5% 제외). 학습 데이터를 만들 때 적용한다
- 이름 표면 차이(띄어쓰기, 지점 표기)로 정답 쌍의 38%가 같은 코드를 받지 못한다. M3 평가에서 영향이 확인되면 텍스트 정리를 재검토한다
- 여행 미션 코드가 참가자가 고른 것인지 과제로 주어진 것인지(카테고리정의서.xlsx 확보 시 확인, `TRAVEL_STYL` 정의도 같은 파일에 있을 것으로 추정)
- RunPod payload 제한과 호출 방식, 콜드 스타트 시 오류 형태(타임아웃 또는 읽기 시간 초과)
- 연속 거리 기준 미달 시 반경 제한 R의 값, 빔 폭, 서비스 기본 서빙 경로, 모델 재학습 기준

## 10. 이 문서의 갱신 규칙

마일스톤이 끝날 때마다 1절(변동 상태)만 고쳐 PR에 포함한다. 나머지 절이 바뀌면 근거 ADR이나 로그와 같은 PR에서 고친다.
