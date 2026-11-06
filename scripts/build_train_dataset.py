"""표시한 학습 데이터를 LLaMA-Factory 학습 형식(sharegpt + 이미지)으로 바꾼다.

미션 애매·보류·거의 같은 사진(near_dup)·규칙과 다른 등급은 뺀다.
사진은 서비스와 같은 리사이즈(normalize_image, 긴 변 1024)를 거쳐 out-dir/images에 저장한다.
사진이 레포 밖(T7)에만 있도록 out-dir도 레포 밖에 둔다. RunPod 볼륨에 올렸다면 학습 후 지운다.

  # 레포 루트에서 실행. 550 세트(1차 학습): run550 표시 + 닮은 장소 + 파일럿 재사용 44개
  R=/Volumes/T7/aihub/train/run550 P=/Volumes/T7/aihub/train/pilot
  uv run python -m scripts.build_train_dataset --aihub /Volumes/T7/aihub --dataset-name photomission_550 \\
      --missions $R/missions.jsonl --missions $R/lookalike_missions.jsonl --missions $R/pilot_reuse.jsonl \\
      --labels $R/train_run550_labels_359_v2.json --labels $P/train_pilot_labels.json --labels $P/train_pilot2_labels.json \\
      --out-dir /Volumes/T7/aihub/train/dataset_550 --dry-run

  --dry-run은 개수만 세고 사진을 쓰지 않는다. 표시 파일에는 학습에 안 쓰는 미션도 있어도 된다
  (파일럿 표시 파일의 나머지는 --missions에 없으면 "미션 파일에 없음"이 아니라 "이번 세트 아님"으로 센다).
  --override는 같은 mission_id의 표시를 통째로 바꾼다 (파일럿 Q2 경계 다시 보기 결과처럼 나중에 고친 표시).

  결과: train.json, dataset_info.json(LLaMA-Factory에 dataset_dir로 이 폴더를 주면 된다), images/,
        train_index.jsonl(train.json과 같은 순서로 mission_id·출처·등급. 학습에는 쓰지 않음)
"""

import argparse
import json
import pathlib
from collections import Counter

from app.photomissions.pipeline_verify import normalize_image
from scripts.aihub_common import DATASETS
from scripts.baseline_verify import index_photos, read_photo
from scripts.question_format import normalize, rule_grade, to_example


def load_missions(lines: list[str]) -> dict[str, dict]:
    missions = {}
    for line in lines:
        if not line.strip():
            continue
        m = json.loads(line)
        if "error" not in m:  # 다시 만들다 에러 난 줄이 앞의 정상 미션을 덮어쓰지 않게 한다
            missions[m["mission_id"]] = m  # 같은 미션을 다시 만들었으면 마지막 정상 기록을 쓴다
    return missions


def merge_missions(files: list[dict[str, dict]]) -> dict[str, dict]:
    # 파일끼리 같은 mission_id가 있으면 어느 미션 글로 배울지 모르니 멈춘다
    merged: dict[str, dict] = {}
    for missions in files:
        dup = sorted(set(merged) & set(missions))
        if dup:
            raise ValueError(f"미션 파일 두 곳에 같은 미션 {len(dup)}개 (예: {dup[:3]})")
        merged |= missions
    return merged


def merge_labels(files: list[list[dict]]) -> list[dict]:
    # 같은 미션이 두 번 표시돼 있으면 학습 데이터에 두 번 들어간다. 어느 쪽이 맞는지는 사람이 정해야 해서 멈춘다
    labels = [x for items in files for x in items]
    dup = sorted(k for k, n in Counter(x["mission_id"] for x in labels).items() if n > 1)
    if dup:
        raise ValueError(f"같은 미션을 두 번 표시함 {len(dup)}개 (예: {dup[:3]}). 표시 파일을 정리할 것")
    return labels


def apply_overrides(labels: list[dict], overrides: list[dict]) -> list[dict]:
    by_id = {x["mission_id"]: x for x in overrides}
    unknown = sorted(set(by_id) - {x["mission_id"] for x in labels})
    if unknown:
        raise ValueError(f"고칠 표시가 원래 표시에 없음 {len(unknown)}개 (예: {unknown[:3]})")
    return [by_id.get(x["mission_id"], x) for x in labels]


def select(labels: list[dict], missions: dict[str, dict],
           allow_extra: bool = False) -> tuple[list[tuple[dict, dict]], Counter]:
    # 학습에 쓸 (표시, 미션) 쌍과 뺀 이유별 개수
    # allow_extra: 표시 파일에 이번 세트가 아닌 미션이 섞여 있어도 된다 (파일럿 표시 중 재사용 44개만 쓸 때)
    missing = [x["mission_id"] for x in labels if x.get("final_grade") and x["mission_id"] not in missions]
    if missing and not allow_extra:
        raise ValueError(f"표시는 있는데 미션 파일에 없는 미션 {len(missing)}개 (예: {missing[:3]}). --missions를 확인할 것")
    kept, skipped = [], Counter()
    for label in labels:
        if label["mission_id"] not in missions:
            skipped["이번 세트 아님"] += 1
        elif not label.get("final_grade"):
            skipped["표시 안 함"] += 1
        elif label["final_grade"] == "unclear":
            skipped["미션 애매"] += 1
        elif label["final_grade"] == "hold":
            skipped["보류"] += 1
        elif label.get("near_dup"):
            # 같은 방문지 사진과 거의 같으면 같은 사진을 두 번 배우는 셈이라 뺀다
            skipped["거의 같은 사진"] += 1
        elif rule_grade(normalize(label)) != label["final_grade"]:
            # 사람이 규칙과 다르게 정한 사진은 객관식 답만으로 등급이 재현되지 않아 이번에는 뺀다
            skipped["규칙과 다른 등급"] += 1
        else:
            kept.append((label, missions[label["mission_id"]]))
    return kept, skipped


def source_of(mission: dict) -> str:
    # 닮은 장소 미션은 source 대신 kind(own·wrong)를 갖는다
    return mission.get("source") or f"lookalike_{mission['kind']}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aihub", type=pathlib.Path, required=True)
    ap.add_argument("--dataset-name", required=True, help="dataset_info.json의 이름 (학습 설정 dataset과 맞출 것)")
    ap.add_argument("--missions", type=pathlib.Path, action="append", required=True, help="미션 jsonl (여러 번 가능)")
    ap.add_argument("--labels", type=pathlib.Path, action="append", required=True, help="표시 결과 (여러 번 가능)")
    ap.add_argument("--override", type=pathlib.Path, help="나중에 고친 표시. 같은 mission_id의 표시를 통째로 바꾼다")
    ap.add_argument("--out-dir", type=pathlib.Path, required=True, help="결과 폴더 (레포 밖에 둘 것)")
    ap.add_argument("--dry-run", action="store_true", help="개수만 세고 사진·파일을 쓰지 않는다")
    args = ap.parse_args()

    try:
        missions = merge_missions([load_missions(p.read_text(encoding="utf-8").splitlines()) for p in args.missions])
        labels = merge_labels([json.loads(p.read_text(encoding="utf-8"))["items"] for p in args.labels])
        if args.override:
            labels = apply_overrides(labels, json.loads(args.override.read_text(encoding="utf-8"))["items"])
        kept, skipped = select(labels, missions, allow_extra=True)
    except ValueError as e:
        ap.error(str(e))
    # 세트에 넣은 미션은 모두 표시돼 있어야 한다. 빠진 게 있으면 표시 파일을 덜 넣은 것이다
    labeled = {x["mission_id"] for x in labels}
    unlabeled = sorted(m for m, x in missions.items() if m not in labeled and x.get("usable", True) and x.get("mission"))
    if unlabeled:
        ap.error(f"미션은 있는데 표시가 없음 {len(unlabeled)}개 (예: {unlabeled[:3]}). --labels를 확인할 것")

    index = index_photos([args.aihub / rel for paths in DATASETS.values() for rel in paths["photos"]])
    no_photo = sorted({label["photo"] for label, _ in kept} - set(index))
    if no_photo:
        ap.error(f"원본 사진을 찾지 못함 {len(no_photo)}장 (예: {no_photo[:3]})")

    examples, rows, grades, sources = [], [], Counter(), Counter()
    for label, mission in kept:
        image_path = args.out_dir / "images" / label["photo"]
        if not args.dry_run and not image_path.exists():
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(normalize_image(read_photo(index[label["photo"]])))
        examples.append(to_example(mission, label, f"images/{label['photo']}"))
        # train.json과 같은 순서. 학습 후 틀린 예시가 어느 미션인지 추적하는 용도
        rows.append({"mission_id": label["mission_id"], "photo": label["photo"], "source": source_of(mission),
                     "grade": label["final_grade"]})
        grades[label["final_grade"]] += 1
        sources[source_of(mission)] += 1

    photos = {ex["images"][0] for ex in examples}
    print(f"학습 예시 {len(examples)}개 · 고유 사진 {len(photos)}장 · 뺀 것 {dict(skipped)}")
    print(f"등급 {dict(grades)} · 출처 {dict(sources)}")
    if args.dry_run:
        print("--dry-run: 파일을 쓰지 않음")
        return

    (args.out_dir / "train.json").write_text(json.dumps(examples, ensure_ascii=False, indent=1), encoding="utf-8")
    (args.out_dir / "train_index.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    dataset_info = {args.dataset_name: {
        "file_name": "train.json",
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "system": "system", "images": "images"},
        "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"},
    }}
    (args.out_dir / "dataset_info.json").write_text(json.dumps(dataset_info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장: {args.out_dir} (train.json, train_index.jsonl, dataset_info.json, images/)")


if __name__ == "__main__":
    main()
