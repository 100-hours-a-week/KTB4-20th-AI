"""표시한 학습 데이터를 LLaMA-Factory 학습 형식(sharegpt + 이미지)으로 바꾼다.

미션 애매는 뺀다. 사진은 서비스와 같은 리사이즈(normalize_image, 긴 변 1024)를 거쳐 out-dir/images에 저장한다.
사진이 레포 밖(T7)에만 있도록 out-dir도 레포 밖에 둔다. RunPod 볼륨에 올렸다면 학습 후 지운다.

  # 레포 루트에서 실행
  uv run python -m scripts.build_train_dataset --aihub /Volumes/T7/aihub --pilot-dir /Volumes/T7/aihub/train/pilot \\
      --labels /Volumes/T7/aihub/train/pilot/train_pilot_labels.json \\
      --labels /Volumes/T7/aihub/train/pilot/train_pilot2_labels.json \\
      --out-dir /Volumes/T7/aihub/train/dataset_pilot

  결과: train.json, dataset_info.json(LLaMA-Factory에 dataset_dir로 이 폴더를 주면 된다), images/,
        train_index.jsonl(train.json과 같은 순서로 mission_id·출처·등급. 학습에는 쓰지 않음)
"""

import argparse
import json
import pathlib
import zipfile
from collections import Counter

from app.photomissions.pipeline_verify import normalize_image
from scripts.question_format import normalize, rule_grade, to_example

DATASET_NAME = "photomission_pilot"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aihub", type=pathlib.Path, required=True)
    ap.add_argument("--pilot-dir", type=pathlib.Path, required=True)
    ap.add_argument("--labels", type=pathlib.Path, action="append", required=True, help="표시 결과 (여러 번 가능)")
    ap.add_argument("--out-dir", type=pathlib.Path, required=True, help="결과 폴더 (레포 밖에 둘 것)")
    args = ap.parse_args()

    rows = {}
    for name in ("manifest.jsonl", "manifest_pilot2.jsonl"):
        path = args.pilot_dir / name
        if path.exists():
            rows |= {r["photo"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}
    missions = {}
    for line in (args.pilot_dir / "missions.jsonl").read_text(encoding="utf-8").splitlines():
        m = json.loads(line)
        if "error" not in m:  # 다시 만들다 에러 난 줄이 앞의 정상 미션을 덮어쓰지 않게 한다
            missions[m["mission_id"]] = m  # 같은 미션을 다시 만들었으면 마지막 정상 기록을 쓴다
    labels = [x for path in args.labels for x in json.loads(path.read_text(encoding="utf-8"))["items"]]

    (args.out_dir / "images").mkdir(parents=True, exist_ok=True)
    zips: dict[str, zipfile.ZipFile] = {}
    examples, index, skipped, grades, sources = [], [], Counter(), Counter(), Counter()
    for label in labels:
        if not label.get("final_grade"):
            skipped["표시 안 함"] += 1
            continue
        if label["final_grade"] == "unclear":
            skipped["미션 애매"] += 1
            continue
        mission = missions[label["mission_id"]]
        if rule_grade(normalize(label)) != label["final_grade"]:
            # 사람이 규칙과 다르게 정한 사진은 객관식 답만으로 등급이 재현되지 않아 이번에는 뺀다 (파일럿에서는 0건)
            skipped["규칙과 다른 등급"] += 1
            continue
        image_path = args.out_dir / "images" / label["photo"]
        if not image_path.exists():
            row = rows[label["photo"]]
            zf = zips.setdefault(row["zip"], zipfile.ZipFile(args.aihub / row["zip"]))
            image_path.write_bytes(normalize_image(zf.read(row["member"])))
        examples.append(to_example(mission, label, f"images/{label['photo']}"))
        # train.json과 같은 순서. 학습 후 틀린 예시가 어느 미션인지 추적하는 용도
        index.append({"mission_id": label["mission_id"], "photo": label["photo"], "source": mission["source"],
                      "grade": label["final_grade"]})
        grades[label["final_grade"]] += 1
        sources[mission["source"]] += 1

    (args.out_dir / "train.json").write_text(json.dumps(examples, ensure_ascii=False, indent=1), encoding="utf-8")
    (args.out_dir / "train_index.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in index), encoding="utf-8")
    dataset_info = {DATASET_NAME: {
        "file_name": "train.json",
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "system": "system", "images": "images"},
        "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"},
    }}
    (args.out_dir / "dataset_info.json").write_text(json.dumps(dataset_info, ensure_ascii=False, indent=1), encoding="utf-8")
    photos = {ex["images"][0] for ex in examples}
    print(f"학습 예시 {len(examples)}개 · 고유 사진 {len(photos)}장 · 뺀 것 {dict(skipped)}")
    print(f"등급 {dict(grades)} · 출처 {dict(sources)}")
    print(f"저장: {args.out_dir} (train.json, train_index.jsonl, dataset_info.json, images/)")


if __name__ == "__main__":
    main()
