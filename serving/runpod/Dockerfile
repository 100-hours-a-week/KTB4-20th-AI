# 장소 선정 모델 전용 RunPod Serverless 이미지 (VLM 엔드포인트와 분리). 레포 루트에서 BuildKit으로 빌드한다.
# 모델 묶음(model.onnx + codes.json)은 레포에 두지 않고 --build-context로 넘긴다:
#   docker buildx build --platform linux/amd64 --build-context bundle=<묶음 폴더> \
#       -f serving/runpod/Dockerfile -t <레지스트리>/place-model-runpod:v1 --load .
# RunPod GPU는 x86_64라서 Apple Silicon에서도 --platform linux/amd64가 필요하다. 이 이미지는 RunPod에서 처음 실행해 검증한다.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir runpod==1.7.9 onnxruntime-gpu==1.20.2 numpy==2.1.3

WORKDIR /app
COPY app/trips/services/model/generation.py app/trips/services/model/generation.py
COPY scripts/semantic_id/tokens.py scripts/semantic_id/tokens.py
COPY scripts/semantic_id/decode.py scripts/semantic_id/decode.py
COPY scripts/semantic_id/preference.py scripts/semantic_id/preference.py
COPY serving/runpod/handler.py handler.py
# 묶음은 빌드 컨텍스트 이름 bundle에서 가져온다(레포 밖 폴더)
COPY --from=bundle / /bundle

CMD ["python3", "-u", "handler.py"]
