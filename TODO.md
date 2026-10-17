A. 로컬 준비

- [x] A1. git switch dev && git pull로 dev를 최신으로 맞춘다(serving/runpod/와 app/trips/services/model/이 보여야 함).
- [ ] A2. 모델 묶음 해시를 확인한다. 아래 명령어 실행한다.
      `bash
    V="/Users/imgyeonglak/Documents/01_career/02_dev_project/96_KTB_4_Final_Project/derived/semantic_id/v1"
    uv run python -m scripts.semantic_id.manifest --version-dir "$V" --verify
    `
- [ ] A3. Docker Desktop을 켜 둔다(docker version에 server가 보여야 함).
- [ ] A4. 컨테이너 레지스트리를 정한다. 같은 계정으로 푸시와 RunPod 접근이 가능해야 한다.
  - Docker Hub(무료 계정은 비공개 저장소 1개) 또는 GHCR(GitHub PAT에 write:packages·read:packages 필요).
  - 이미지에는 가중치와 코드 사전이 들어 있다. 코드 사전은 DB 장소 ID와 코드뿐이고 여행자 정보는 없으나, 공개 저장소에 올릴지 비공개로 둘지는 결정한다(비공개를 권장).
- [ ] A5. 레지스트리에 로그인한다(docker login).

B. 이미지 빌드와 푸시

- [ ] B1. 레포 루트에서 빌드한다(로컬 확인은 이미 했으나 푸시용 태그로 다시 빌드):
      docker buildx build --platform linux/amd64 --build-context bundle="$V/m3/bundle_a" -f serving/runpod/Dockerfile -t <레지스트리>/<이름>/place-model-runpod:v1 --load .
      (태그는 버전을 붙인다. latest는 쓰지 않는다. 같은 태그로 다시 푸시하면 RunPod가 캐시한 이전 이미지를 쓸 수 있다.)
- [ ] B2. 푸시 전 로컬 확인: 앞서 쓴 핸들러 호출 테스트가 동작하는지 한 번 더(선택).
- [ ] B3. docker push <태그>. 4.45GB라 업로드 시간이 걸린다(회선 속도에 따라 다름). 푸시 후 레지스트리 웹에서 태그와 크기를 확인한다.
- [ ] B4. 이미지가 비공개면 RunPod에 레지스트리 접근 정보를 등록해야 한다(아래 C3).

C. RunPod 계정과 접근

- [ ] C1. RunPod 계정의 크레딧(또는 결제 수단)을 확인한다. 노트에는 "GPU 비용은 지원"으로 적혀 있으나 실제 잔액은 확인하지 못했다. 벤치마크 중 과금이 시작되므로 예상 한도를 정해 둔다.
- [ ] C2. API 키를 만든다(설정의 API Keys). 엔드포인트 호출 권한이 있는 키여야 한다. 키는 채팅, 코드, 커밋에 쓰지 않고 환경변수나 .env(커밋되지 않음)에만 둔다.
- [ ] C3. 비공개 이미지라면 RunPod에 컨테이너 레지스트리 인증 정보를 등록한다(콘솔의 레지스트리 설정 메뉴).

D. Serverless 엔드포인트 만들기 (VLM 엔드포인트와 별도로)

- [ ] D1. 새 Serverless 엔드포인트를 만든다. 이름은 장소 선정 모델임을 알 수 있게 한다.
- [ ] D2. 컨테이너 이미지에 B1의 태그를 입력하고, 필요하면 D1의 인증 정보를 선택한다.
- [ ] D3. GPU를 고른다. x86 서버의 흔한 GPU(16~24GB급)를 권한다. 가장 최신 세대는 CUDA 12.4 이미지와 onnxruntime-gpu 1.20.2 호환이 확인되지 않았으므로 피한다. 허용 CUDA 버전을 제한하는 옵션이 있으면 12.4 이상으로 둔다.
- [ ] D4. 워커 설정: 상시(Active/Min) 워커 0, 최대(Max) 워커 1~2, 유휴 시간 제한은 짧게(기본값 확인). 상시 워커 0이어야 콜드 스타트를 잴 수 있다.
- [ ] D5. 컨테이너 디스크는 이미지 크기보다 넉넉하게(예: 20GB 이상), 실행 시간 제한은 지나치게 길지 않게 둔다.
- [ ] D6. 빠른 시작(FlashBoot 등)이라는 옵션이 있으면 켠 경우와 끈 경우를 나눠 잴지 정한다(처음에는 기본값으로 시작).
- [ ] D7. 엔드포인트 ID를 적어 둔다. 호출 주소는 https://api.runpod.ai/v2/<엔드포인트 ID>이다.

E. 동작 확인 (스모크 테스트)

- [ ] E1. 샘플 요청을 만든다: bundle_a/codes.json의 장소 ID 몇십 개를 슬롯 후보로 넣은 {"input": {"header": [1,3,5,7,9,11], "slots": [[...],[...]], "width": 10}}(제가 요청 파일을 만들어 드릴 수 있다).
- [ ] E2. curl -X POST "$RUNPOD_MODEL_URL/runsync" -H "Authorization: Bearer $RUNPOD_MODEL_API_KEY" -H "Content-Type: application/json" -d @payload.json을 실행한다.
- [ ] E3. 응답에서 status가 COMPLETED인지, output.places가 슬롯 수와 같은 길이인지 확인한다.
- [ ] E4. 응답의 output.providers에 CUDAExecutionProvider가 있는지 본다. CPUExecutionProvider뿐이면 GPU를 못 쓴 것이므로 콘솔의 워커 로그를 열어 원인(드라이버·CUDA 버전·장치 인식)을 확인하고 로그를 보내 준다.
- [ ] E5. 실패하면(이미지 내려받기 실패, exec format error, CUDA 오류 등) 콘솔의 워커 로그 전문을 복사한다.

F. 벤치마크 측정 (ADR-0004 기본 경로 결정용)

- [ ] F1. 콜드 스타트: 워커가 모두 내려간 상태(유휴 시간 경과 후)에서 요청 1건을 보내 총 소요 시간을 잰다. 각 간격을 두고 5회 반복해 값을 모두 기록한다. 응답에 RunPod가 주는 대기·실행 시간 필드가 있으면(delayTime, executionTime 등으로 알고 있으나 확인 필요) 함께 적는다.
- [ ] F2. 따뜻한 상태: 워커가 떠 있을 때 같은 요청을 30~50회 연속으로 보내 평균, p50, p95 지연을 잰다.
- [ ] F3. 현실적인 페이로드: 실제 서비스처럼 슬롯 5~6개의 후보 목록(한 슬롯에 수백~수천 개 ID)을 보내 페이로드 크기(KB)와 지연을 잰다.
- [ ] F4. 큰 페이로드 한계: 후보를 늘려 가며 요청이 거부되거나 오류가 나는 크기를 찾는다. ADR-0004의 열린 항목("RunPod payload 제한")에 쓸 값이다.
- [ ] F5. 동시 요청: 2~4개를 동시에 보내 워커 증설과 지연 변화를 본다.
- [ ] F6. 비용: 측정 기간의 청구 금액이나 사용 시간을 콘솔에서 기록한다.
- [ ] F7. 기록할 값 정리: 콜드 스타트 5회, 따뜻한 지연 평균/p50/p95, 페이로드 크기, providers, GPU 종류, 워커 설정, 이미지 크기(제가 벤치마크 스크립트를 만들어 이 측정을 자동화해 드릴 수 있다).

G. AI 서버와 연결해 끝에서 끝까지 확인 (로컬에서)

- [ ] G1. .env(커밋되지 않음)에 다음을 넣는다: PLACE_SELECTION_MODE=moNPOD_MODEL_API_KEY, MODEL_DIR(= bundle_a 경로).
- [ ] G2. uv run --extra model uvicorn app.main:app로 서버를 띄우고, /trips/place-selection에 샘플 요청을 보낸다.
- [ ] G3. 서버 로그에서 route=gpu(또는 cpu)와 장소 수를 확인한다.
- [ ] G4. 폴백 시험: 잘못된 URL이나 키를 넣고 요청해 route=cpu로 넘어가는지, 연속 3번 실패 후 60초 동안 GPU 호출이 막히는지(로그에 GPU 호출이 없어짐) 확인한다. 쿨다운
      후 정상 URL로 복구되는지도 본다.
- [ ] G5. 규칙으로 되돌아가는 경우(모델 경로 실패, 규칙 경로로 되돌림 로그)가 정상 시나리오에서 나오지 않는지 확인한다.

H. AWS 서버 CPU 경로 확인 (서버 접근이 가능할 때)

- [ ] H1. 서버의 CPU 아키텍처(x86/arm)를 확인한다(uname -m). 앞서 확인
- [ ] H2. 서버에 모델 묶음(bundle_a)을 올릴 위치를 정하고, AI 서버 컨테이너에 마운트한다(MODEL_DIR).
- [ ] H3. 서버에서 bench_model_path.py 또는 실제 요청으로 평균/p95 지 c 값 480ms/120MB와 비교).
- [ ] H4. AI 서버 이미지가 --extra model로 빌드되어 배포되는지 확인한다(CI 이미지 빌드 로그).

I. 마무리와 정리

- [ ] I1. 측정이 끝나면 엔드포인트의 최대 워커를 0으로 줄이거나 엔드포과금 방지). 쓴 Pod가 있으면 종료한다.
- [ ] I2. API 키를 .env에만 두고 채팅·커밋에 노출되지 않았는지 확인한다. 노출됐다면 즉시 폐기하고 새로 발급한다.
- [ ] I3. F7의 값, 엔드포인트 설정, 로그를 저에게 전달한다.
- [ ] I4. (제가 하는 후속) 값을 받으면 ADR-0004를 갱신한다: 기본 서빙 경로 결정(GPU 우선 또는 CPU 기본), 상시 워커 사용 여부, 타임아웃 수치, payload 제한, 콜드 스타트
      값. 그리고 ADR-0006을 Accepted로 전환하고, 모델 경로를 켜는 방법을

제가 미리 준비해 드릴 수 있는 것 (원하시면 말씀해 주세요)

1. E1용 요청 파일 생성 스크립트와 실제 크기의 페이로드.
2. F 전체를 자동화하는 벤치마크 스크립트 scripts/bench_runpod.py(콜드/ 측정, 결과를 JSON으로 저장, API 키는 환경변수로만 받음).
3. G4 폴백 시험을 자동으로 하는 확인 스크립트.

가장 막힐 가능성이 큰 단계는 A4(레지스트리 선택과 비공개 이미지 접근), D3(GPU·CUDA 호환), E4(GPU 인식 여부)입니다. 막히면 해당 단계의 로그를 그대로 보내 주세요.
