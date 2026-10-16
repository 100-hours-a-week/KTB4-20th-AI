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
PLACE_SELECTION_MODE=model
MODEL_DIR=/models/place_model_v1        # CPU 경로용 묶음(컨테이너에 마운트)
RUNPOD_MODEL_URL=https://api.runpod.ai/v2/<엔드포인트 ID>
RUNPOD_MODEL_API_KEY=<RunPod API 키>
```

`RUNPOD_MODEL_URL`을 비우면 CPU 경로만, `MODEL_DIR`을 비우면 GPU 경로만 쓴다. 모델 경로가 실패하면 기존 규칙 경로로 응답한다.

## 동작 확인과 벤치마크

```
curl -s -X POST "$RUNPOD_MODEL_URL/runsync" -H "Authorization: Bearer $RUNPOD_MODEL_API_KEY" -H 'Content-Type: application/json' \
  -d '{"input": {"header": [1,3,5,7,9,11], "slots": [["<google_place_id>", "<google_place_id>"]], "width": 10}}'
```

- 첫 요청(콜드 스타트 포함)과 이후 요청의 지연을 각각 적는다.
- 이 이미지는 레포에서 빌드·실행까지 확인하지 못했다(RunPod 접근 불가). 첫 배포에서 오류가 나면 로그를 알려 주면 된다.
