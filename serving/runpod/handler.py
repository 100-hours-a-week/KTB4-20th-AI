"""RunPod Serverless 핸들러: 장소 선정 모델의 하루 일정 생성 (ADR-0004).

요청: {"input": {"header": [토큰...], "slots": [[google_place_id...] × 슬롯 수], "width": 10}}
응답: {"places": [슬롯마다 고른 google_place_id 또는 null]}

모델 가중치(model.onnx)와 코드 사전(codes.json)은 이미지 안 /bundle에 함께 들어 있다. 같은 버전임이 보장되므로
요청에는 후보 목록만 싣는다. 빔 서치와 후보 제한은 AI 서버 CPU 경로와 같은 함수(generate_itinerary)를 쓴다.
VLM(Qwen2.5-VL)과는 별도 엔드포인트로 배포한다.
"""

import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import runpod

from app.trips.services.model.generation import BEAM_WIDTH, generate_itinerary

BUNDLE = Path(os.environ.get("BUNDLE_DIR", "/bundle"))
SESSION = ort.InferenceSession(str(BUNDLE / "model.onnx"), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
BUNDLE_DATA = json.loads((BUNDLE / "codes.json").read_text(encoding="utf-8"))
CODES = BUNDLE_DATA["codes"]


def next_logprobs(prefixes):
    return SESSION.run(None, {"tokens": np.asarray(prefixes, dtype=np.int64)})[0]


def handler(event):
    payload = event.get("input") or {}
    header, slots = payload.get("header"), payload.get("slots")
    if not isinstance(header, list) or not isinstance(slots, list) or not 1 <= len(slots) <= 8:
        return {"error": "header와 slots(1~8개)가 필요함"}
    width = min(int(payload.get("width", BEAM_WIDTH)), 20)
    places = generate_itinerary(header, slots, CODES, next_logprobs, width)
    return {"places": places, "providers": SESSION.get_providers(), "model": BUNDLE_DATA["version"].get("model_sha256", "")[:12]}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
