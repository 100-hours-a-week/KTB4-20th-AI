"""파생 산출물 버전 폴더의 MANIFEST.json을 만든다 (M2).

M1·M2 결과 파일(days.jsonl, split.json, 임베딩, RQ-VAE, 코드 사전 등)은 레포 밖에 둔다. 여행자 ID와 좌표가 들어 있고
AI-Hub 약관상 재배포 가능 여부를 확인하지 못했기 때문이다. 대신 어떤 코드와 설정으로 만들었는지, 파일이 그대로인지를
MANIFEST.json에 남긴다. 파일마다 sha256과 크기를 적는다.

보관 위치: $AIHUB_ROOT/derived/semantic_id/<버전>/   (버전을 올릴 때는 새 폴더를 만들고 이전 폴더는 덮어쓰지 않는다)

  uv run python -m scripts.semantic_id.manifest --version-dir "$AIHUB_ROOT/derived/semantic_id/v1" \\
      --note "m1/ m2/는 확정한 설계(카테고리 라벨 제외, BGE-M3), experiments/는 모델 비교 근거"

이미 만든 MANIFEST.json은 고치지 않는다. 폴더에 파일을 더 넣었다면 그 부분만 담은 MANIFEST를 따로 만든다:

  uv run python -m scripts.semantic_id.manifest --version-dir "$V" --name MANIFEST_m3.json --include m3 --note "..."

같은 폴더에 그 뒤로 새 파일만 생겼다면 --only-new로 기존 MANIFEST에 없는 파일만 담는다:

  uv run python -m scripts.semantic_id.manifest --version-dir "$V" --name MANIFEST_m3b.json --only-new --note "..."

--verify는 폴더의 MANIFEST*.json을 모두 읽어 합친 목록과 실제 파일을 대조한다(같은 파일이 두 MANIFEST에 있으면 오류).
"""

import argparse
import datetime
import hashlib
import json
import pathlib
import subprocess
import sys

MANIFEST = "MANIFEST.json"


def sha256_of(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_manifest(path: pathlib.Path) -> bool:
    return path.name.startswith("MANIFEST") and path.suffix == ".json"


def list_files(root: pathlib.Path, include: str | None = None) -> list[pathlib.Path]:
    files = sorted(p for p in root.rglob("*") if p.is_file() and not is_manifest(p) and not p.name.startswith("."))
    if include:
        files = [p for p in files if p.relative_to(root).parts[0] == include]
    return files


def listed_files(root: pathlib.Path) -> set[str]:
    # 폴더의 MANIFEST*.json에 이미 적힌 파일 이름
    names: set[str] = set()
    for path in root.iterdir():
        if is_manifest(path):
            names |= set(json.loads(path.read_text(encoding="utf-8"))["files"])
    return names


def file_table(root: pathlib.Path, include: str | None = None, skip: set[str] | None = None) -> dict[str, dict]:
    return {
        str(p.relative_to(root)): {"sha256": sha256_of(p), "bytes": p.stat().st_size}
        for p in list_files(root, include) if str(p.relative_to(root)) not in (skip or set())
    }


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def verify(root: pathlib.Path) -> list[str]:
    # 폴더의 MANIFEST*.json을 합친 목록과 실제 파일이 다른 곳을 모두 알려 준다
    problems = []
    expected: dict[str, dict] = {}
    for path in sorted(p for p in root.iterdir() if is_manifest(p)):
        for name, entry in json.loads(path.read_text(encoding="utf-8"))["files"].items():
            if name in expected:
                problems.append(f"여러 MANIFEST에 있음: {name} ({path.name})")
            expected[name] = entry
    actual = file_table(root)
    for name in sorted(set(expected) - set(actual)):
        problems.append(f"없는 파일: {name}")
    for name in sorted(set(actual) - set(expected)):
        problems.append(f"MANIFEST에 없는 파일: {name}")
    for name in sorted(set(expected) & set(actual)):
        if expected[name]["sha256"] != actual[name]["sha256"]:
            problems.append(f"내용이 다름: {name}")
    return problems


def build(root: pathlib.Path, note: str, extra: dict | None = None, include: str | None = None, skip: set[str] | None = None) -> dict:
    return {
        "created": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "repo_commit": git_commit(),
        "note": note,
        **(extra or {}),
        "files": file_table(root, include, skip),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version-dir", type=pathlib.Path, required=True)
    ap.add_argument("--note", default="", help="이 버전이 무엇인지 한두 문장")
    ap.add_argument("--info", type=pathlib.Path, help="MANIFEST에 그대로 넣을 설정 JSON (모델, 시드 등)")
    ap.add_argument("--name", default=MANIFEST, help="만들 MANIFEST 파일 이름 (MANIFEST*.json)")
    ap.add_argument("--include", help="이 하위 폴더의 파일만 담는다 (예: m3)")
    ap.add_argument("--only-new", action="store_true", help="기존 MANIFEST*.json에 이미 있는 파일은 빼고 새 파일만 담는다")
    ap.add_argument("--verify", action="store_true", help="만들지 않고 기존 MANIFEST와 파일이 같은지 확인한다")
    args = ap.parse_args(argv)
    root = args.version_dir
    if args.verify:
        problems = verify(root)
        print("\n".join(problems) if problems else f"MANIFEST와 파일 {len(file_table(root))}개가 모두 같음")
        return 1 if problems else 0
    if not (args.name.startswith("MANIFEST") and args.name.endswith(".json")):
        print("--name은 MANIFEST로 시작하고 .json으로 끝나야 함", file=sys.stderr)
        return 1
    if (root / args.name).exists():
        print(f"이미 있어서 덮어쓰지 않음: {root / args.name} (새 이름이나 새 버전 폴더를 만든다)", file=sys.stderr)
        return 1
    extra = json.loads(args.info.read_text(encoding="utf-8")) if args.info else None
    manifest = build(root, args.note, extra, args.include, listed_files(root) if args.only_new else None)
    if not manifest["files"]:
        print("담을 파일이 없음", file=sys.stderr)
        return 1
    (root / args.name).write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    total = sum(f["bytes"] for f in manifest["files"].values())
    print(f"MANIFEST 작성: 파일 {len(manifest['files'])}개, {total / 1e6:.0f}MB, 커밋 {manifest['repo_commit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
