# 장소 선정 모델 RunPod 엔드포인트 (ADR-0004)

VLM(Qwen2.5-VL) 엔드포인트와 별도로 배포한다. 모델 가중치와 코드 사전은 같은 묶음(`model.onnx` + `codes.json`)으로 이미지에 넣는다.

## 묶음 만들기 (로컬)

```
uv run --with torch --with onnx --with onnxruntime --with numpy python -m scripts.semantic_id.export_onnx \
    --model-dir "$V/m3/model_a" --out "$V/m3/model_a/model.onnx"
uv run python -m scripts.semantic_id.package_bundle --model-dir "$V/m3/model_a" --places "$V/m3/places.json" --out-dir "$V/m3/bundle_a"
```

## 이미지 빌드와 배포 (사용자가 직접)

이 단계는 사용자의 컴퓨터(또는 CI)에서 한다. RunPod Pod의 터미널에서 실행하는 것이 아니다. 만든 이미지를 레지스트리에 올리고, RunPod 콘솔의 Serverless 엔드포인트가 그 이미지를 받아 GPU 워커를 띄운다.

```
docker buildx build --platform linux/amd64 --build-context bundle="$V/m3/bundle_a" \
    -f serving/runpod/Dockerfile -t <레지스트리>/place-model-runpod:v1 --load .
docker push <레지스트리>/place-model-runpod:v1
```

- `--build-context`는 BuildKit(Docker 23 이상 기본)이 필요하다. 모델 묶음은 레포 밖에 두고 이 옵션으로만 넘기므로 가중치가 레포에 들어가지 않는다.
- `--platform linux/amd64`는 RunPod GPU가 x86_64라서 필요하다(Apple Silicon에서 기본 빌드는 arm이라 실행되지 않는다).
- RunPod 콘솔에서 Serverless 엔드포인트를 만들고 위 이미지를 지정한다(GPU 하나, 최대 워커 1~2, 상시 워커 0으로 시작해 콜드 스타트를 잰다). 메뉴 이름은 콘솔 버전에 따라 다를 수 있다.
- Docker 없이 GPU 추론만 먼저 확인하려면 Pod를 띄워 터미널에서 `pip install onnxruntime-gpu numpy` 후 `handler`를 직접 실행하면 되지만, Pod는 이미 켜져 있어 콜드 스타트를 잴 수 없다.

## AI 서버 환경변수

```
MODEL_DIR=/models/place_model_v1        # CPU 경로용 묶음(컨테이너에 마운트)
RUNPOD_MODEL_URL=https://api.runpod.ai/v2/<엔드포인트 ID>
RUNPOD_MODEL_API_KEY=<RunPod API 키>
```

`RUNPOD_MODEL_URL`을 비우면 CPU 경로만, `MODEL_DIR`을 비우면 GPU 경로만 쓴다. 모델 경로가 실패하면 기존 규칙 경로로 응답한다.

## 운영 설정과 벤치마크 결과 (2026-10-18, ADR-0004)

- 엔드포인트 `place-model-bench`: Active 워커 0, Max 2, Idle timeout 5초, FlashBoot 꺼짐, 컨테이너 디스크 20GB, 실행 제한 120초.
- **GPU는 Ada·Ampere 16GB 그룹(RTX 2000 Ada, 4000 Ada, A4000, A4500)으로 고정한다.** 24GB 그룹(PRO 6000 MIG 등 최신 세대 포함)이 배정되면 첫 요청이 약 45초 걸린다.
- 측정(16GB 그룹): 따뜻한 상태 총 1.3초, 90초 유휴 후 첫 요청 10~13초, 요청 크기 15MB까지 성공. 자세한 값은 ADR-0004.
- 재현: `scripts/make_runpod_payload.py`로 요청 파일을 만들고 `scripts/bench_runpod.py`로 측정한다(호출은 비용이 나가므로 직접 실행, 키는 `.env`에 둔다).

## 여러 날 요청 (ADR-0008)

기존 `slots` 요청은 그대로 받는다. `days`가 있으면 여러 날 일정으로 처리한다. 후보 풀을 이름 붙여 한 번만 보내고 슬롯은 풀 이름으로 가리키므로, 요청 크기가 일수에 비례하지 않는다.

```
{"input": {"header": [1,3,5,7,9,11],
           "pools": {"daytime": ["<google_place_id>", ...], "food": ["<google_place_id>", ...]},
           "days": [["daytime","food","daytime","daytime","food"], ["daytime","food"]],
           "width": 10, "first_k": 30, "seed": 123}}
```

- 응답: `{"days": [[슬롯마다 고른 google_place_id 또는 null] × 일수], "providers": [...], "model": "..."}`. 입력이 틀리면 `{"error": "..."}`다.
- 일수는 1~9, 하루 슬롯은 1~8개다. 일자마다 문맥을 처음부터 시작하고, 이미 쓴 장소는 여행 전체에서 후보에서 뺀다. 첫 장소 무작위의 시드는 일자마다 `seed + 일자 번호(0부터)`다.
- 이 요청은 이 형식을 아는 이미지부터 처리한다. 이전 이미지(v2)는 `days`를 모르고 오류 응답을 돌려주므로, AI 서버가 이를 실패로 보고 CPU나 규칙으로 되돌린다.

## 동작 확인과 벤치마크

```
curl -s -X POST "$RUNPOD_MODEL_URL/runsync" -H "Authorization: Bearer $RUNPOD_MODEL_API_KEY" -H 'Content-Type: application/json' \
  -d '{"input": {"header": [1,3,5,7,9,11], "slots": [["<google_place_id>", "<google_place_id>"]], "width": 10}}'
```

- 첫 요청(콜드 스타트 포함)과 이후 요청의 지연을 각각 적는다.
- 로컬(Apple Silicon, `--platform linux/amd64` 에뮬레이션)에서 이미지 빌드와 `handler.handler` 호출(CPU provider)까지 확인했다. 이미지 크기는 약 4.45GB(CUDA 베이스 포함)라 콜드 스타트에 이미지 내려받기가 포함될 수 있다.
- GPU 추론(CUDA provider)은 로컬에서 확인하지 못했다. 핸들러는 `/dev/nvidiactl`이 있을 때만 CUDA를 요청한다(장치 없이 요청하면 폴백하지 않고 프로세스가 죽는 것을 로컬에서 확인). 응답의 `providers` 값으로 실제로 GPU를 썼는지 본다.
- 첫 배포에서 오류가 나면 로그를 알려 주면 된다(RunPod 접근 불가).
