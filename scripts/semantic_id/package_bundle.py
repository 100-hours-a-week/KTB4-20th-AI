"""서빙용 묶음(model.onnx + codes.json)을 만든다 (M5, ADR-0004).

모델 가중치와 코드 사전은 같은 버전이어야 한다. 한 폴더에 함께 넣고 해시를 적어, CPU 경로(컨테이너)와
RunPod 이미지가 같은 묶음을 쓰게 한다.

  uv run python -m scripts.semantic_id.package_bundle --model-dir "$V/m3/model_a" --places "$V/m3/places.json" --out-dir "$V/m3/bundle_a"
"""

import argparse
import hashlib
import json
import pathlib
import shutil
import sys


def sha256_of(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=pathlib.Path, required=True, help="model.onnx와 meta.json이 있는 학습 결과 폴더")
    ap.add_argument("--places", type=pathlib.Path, required=True, help="train_data가 만든 places.json")
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = ap.parse_args(argv)
    if args.out_dir.exists():
        print(f"이미 있어서 덮어쓰지 않음: {args.out_dir}", file=sys.stderr)
        return 1
    places = json.loads(args.places.read_text(encoding="utf-8"))
    codes = {p["google_place_id"]: p["code"] for p in places if p["code"] is not None and p["passes_filter"]}
    meta = json.loads((args.model_dir / "meta.json").read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True)
    shutil.copy2(args.model_dir / "model.onnx", args.out_dir / "model.onnx")
    version = {"model_sha256": sha256_of(args.out_dir / "model.onnx"), "params": meta["params"],
               "best_dev_loss": meta["best_dev_loss"], "places": len(codes), "note": "품질 필터 통과 장소만 포함"}
    (args.out_dir / "codes.json").write_text(json.dumps({"version": version, "codes": codes}, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"묶음 {args.out_dir}: 장소 {len(codes)}개, model.onnx {(args.out_dir / 'model.onnx').stat().st_size / 1e6:.1f}MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
