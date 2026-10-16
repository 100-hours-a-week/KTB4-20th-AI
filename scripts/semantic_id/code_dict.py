"""코드 사전: Semantic ID 코드 → 장소를 되찾는 표를 만든다 (M2 T2-5, ADR-0004).

모델이 코드(c1, c2, c3, n)를 생성하면 이 표로 `google_place_id`를 찾는다. 키는 `ai_places.id`가 아니라
`google_place_id`다. 재수집하면 `ai_places.id`가 바뀌기 때문이다(ADR-0001).

사전에는 만든 조건(임베딩 모델과 개정 해시, RQ-VAE 가중치 해시, DB 장소 수, 텍스트 형식)을 함께 적는다.
모델이나 DB가 바뀌면 코드가 달라지므로, 사전과 모델 가중치가 같은 버전인지 확인하는 근거가 된다.

  uv run python -m scripts.semantic_id.code_dict --embed-dir "$OUT/embed_bgem3_nocat" --rqvae-dir "$OUT/rqvae_bgem3_nocat" --out "$OUT/code_dictionary.json"
"""

import argparse
import datetime
import hashlib
import json
import pathlib
import sys

HF_CACHE = pathlib.Path.home() / ".cache" / "huggingface" / "hub"


def code_key(code: list[int] | tuple[int, ...]) -> str:
    return "-".join(str(int(c)) for c in code)


def build_dictionary(semantic_ids: dict[str, list[int]]) -> dict[str, str]:
    """semantic_ids: google_place_id → [c1, c2, c3, n]. 코드 하나가 장소 둘을 가리키면 오류다."""
    out: dict[str, str] = {}
    for place_id, code in sorted(semantic_ids.items()):
        key = code_key(code)
        if key in out:
            raise ValueError(f"같은 코드 {key}가 두 장소에 있음: {out[key]}, {place_id}")
        out[key] = place_id
    return out


def sha256_of(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hf_revision(model: str, cache: pathlib.Path = HF_CACHE) -> str | None:
    # 내려받은 모델의 개정(커밋) 해시. refs/main이 가리키는 개정을 쓰고, 없으면 스냅샷이 하나일 때만 그 값을 쓴다
    root = cache / ("models--" + model.replace("/", "--"))
    ref = root / "refs" / "main"
    if ref.is_file():
        return ref.read_text(encoding="utf-8").strip() or None
    snapshots = root / "snapshots"
    if not snapshots.is_dir():
        return None
    revisions = [p.name for p in snapshots.iterdir() if p.is_dir()]
    return revisions[0] if len(revisions) == 1 else None


DEFAULT_TEXT_FORMAT = "이름 | 시도 시군구"  # 결정 이후 만든 임베딩 산출물에는 meta에 있다. 없으면 이 형식으로 본다(텍스트 일치를 따로 확인함)


def build_payload(
    embed_meta: dict, rqvae_meta: dict, rqvae_sha: str, semantic_ids: dict[str, list[int]], revision: str | None,
    texts_sha: str | None = None,
) -> dict:
    return {
        "version": {
            "embedding_model": embed_meta["model"], "embedding_revision": revision, "embedding_dim": embed_meta["dim"],
            "text_format": embed_meta.get("text_format", DEFAULT_TEXT_FORMAT), "texts_sha256": texts_sha,
            "rqvae_sha256": rqvae_sha, "levels": rqvae_meta["levels"], "codebook": rqvae_meta["codebook"],
            "places": len(semantic_ids), "created": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        },
        "codes": build_dictionary(semantic_ids),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embed-dir", type=pathlib.Path, required=True)
    ap.add_argument("--rqvae-dir", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args(argv)
    if args.out.exists():
        print(f"이미 있어서 덮어쓰지 않음: {args.out}", file=sys.stderr)
        return 1
    embed_meta = json.loads((args.embed_dir / "meta.json").read_text(encoding="utf-8"))
    rqvae_meta = json.loads((args.rqvae_dir / "meta.json").read_text(encoding="utf-8"))
    semantic_ids = json.loads((args.rqvae_dir / "db_semantic_ids.json").read_text(encoding="utf-8"))
    payload = build_payload(
        embed_meta, rqvae_meta, sha256_of(args.rqvae_dir / "rqvae.pt"), semantic_ids, hf_revision(embed_meta["model"]),
        sha256_of(args.embed_dir / "texts.jsonl"),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    v = payload["version"]
    print(f"코드 사전 {len(payload['codes'])}개 · 모델 {v['embedding_model']} ({v['embedding_revision']}) · RQ-VAE {v['rqvae_sha256'][:12]}")
    print(f"저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
