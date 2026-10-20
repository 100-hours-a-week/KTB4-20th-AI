"""RunPod 엔드포인트 시험용 요청 파일을 만든다 (M5 T5-3, T5-5, ADR-0004).

서비스가 보내는 것과 같은 요청(헤더 토큰 + 슬롯별 후보 google_place_id 목록)을 로컬 DB로 만들어 파일로 저장한다.
엔드포인트는 호출하지 않는다. 파일은 레포 밖 폴더에 둔다.

  smoke.json            후보가 적은 가장 작은 요청 (동작 확인용)
  realistic_seoul.json  서울, 역사·문화 취향. 서비스가 실제로 보내는 크기
  realistic_jeju.json   제주, 자연·액티비티 취향
  pad_0p5mb.json ...    위 요청에 모델이 쓰지 않는 가짜 ID를 채워 요청 크기만 키운 것 (전송 크기 한도 확인용)

  uv run python -m scripts.make_runpod_payload --codes "$V/m3/bundle_a/codes.json" --out-dir "$V/m3/runpod"
"""

import argparse
import json
import pathlib
import sys

from app.trips.schemas.schemas import E_Region
from app.trips.services.model.selection import (
    model_slot_plan_seed,
    select_places_by_model,
)
from app.trips.services.slot_plan import build_slot_plan
from scripts.bench_model_path import PROFILES, member

PAD_SIZES_MB = {"pad_0p5mb": 0.5, "pad_1mb": 1, "pad_5mb": 5, "pad_10mb": 10, "pad_15mb": 15}


class CapturingRouter:
    # 추론은 하지 않고 서비스가 보내려던 요청만 붙잡는다
    def __init__(self):
        self.header, self.slots = None, None

    def generate(self, header, slots, width=10):
        self.header, self.slots = header, slots
        return [None] * len(slots), "capture"


def capture(region: E_Region, profile: str) -> dict:
    members = [member(PROFILES[profile])]
    router = CapturingRouter()
    select_places_by_model(build_slot_plan(members, model_slot_plan_seed()), [], region, members, router)
    return {"input": {"header": router.header, "slots": router.slots, "width": 10}}


def pad_to(payload: dict, megabytes: float) -> dict:
    # 첫 슬롯 끝에 가짜 ID를 붙여 직렬화한 크기를 맞춘다. 가짜 ID는 코드 사전에 없어 모델이 무시한다
    out = json.loads(json.dumps(payload))
    target = int(megabytes * 1024 * 1024)
    i = 0
    while len(json.dumps(out, ensure_ascii=False).encode()) < target:
        out["input"]["slots"][0].extend(f"pad_{i + k:07d}_{'x' * 20}" for k in range(2000))
        i += 2000
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codes", type=pathlib.Path, required=True, help="서빙 묶음의 codes.json")
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = ap.parse_args(argv)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        print(f"이미 있어서 덮어쓰지 않음: {args.out_dir}", file=sys.stderr)
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    codes = sorted(json.loads(args.codes.read_text(encoding="utf-8"))["codes"])
    files = {
        "smoke": {"input": {"header": [1, 3, 5, 7, 9, 11], "slots": [codes[:20], codes[20:40]], "width": 10}},
        "realistic_seoul": capture(E_Region.SEOUL, "역사문화"),
        "realistic_jeju": capture(E_Region.JEJU, "자연액티비티"),
    }
    for name, mb in PAD_SIZES_MB.items():
        files[name] = pad_to(files["realistic_seoul"], mb)
    for name, body in files.items():
        raw = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
        (args.out_dir / f"{name}.json").write_text(raw, encoding="utf-8")
        slots = body["input"]["slots"]
        print(f"{name:16s} {len(raw.encode()) / 1024:9.1f}KB · 슬롯 {len(slots)}개 · 슬롯별 후보 {[len(s) for s in slots][:6]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
