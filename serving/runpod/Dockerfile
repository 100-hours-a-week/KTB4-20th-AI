# 장소 선정 모델 전용 RunPod Serverless 이미지 (VLM 엔드포인트와 분리). 레포 루트에서 빌드한다:
#   docker build -f serving/runpod/Dockerfile --build-arg BUNDLE=<model.onnx와 codes.json이 있는 폴더> -t place-model-runpod .
# 로컬에서 GPU 없이 확인: docker run --rm place-model-runpod python -c "import handler"  (이 파일은 런타임 확인 전이라 RunPod에서 처음 검증된다)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir runpod==1.7.9 onnxruntime-gpu==1.20.1 numpy==2.1.3

WORKDIR /app
COPY app/trips/services/model/generation.py app/trips/services/model/generation.py
COPY scripts/semantic_id/tokens.py scripts/semantic_id/tokens.py
COPY scripts/semantic_id/decode.py scripts/semantic_id/decode.py
COPY scripts/semantic_id/preference.py scripts/semantic_id/preference.py
COPY serving/runpod/handler.py handler.py
ARG BUNDLE
COPY ${BUNDLE} /bundle

CMD ["python3", "-u", "handler.py"]
