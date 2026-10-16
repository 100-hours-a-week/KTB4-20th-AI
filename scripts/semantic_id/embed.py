"""DB 장소와 로그 방문지의 텍스트를 임베딩해 저장한다 (M2 T2-2, ADR-0003).

임베딩 모델은 오픈 모델이고 오프라인으로 한 번만 계산한다. 무거운 패키지(torch 등)는 프로젝트 의존성에 넣지 않고
`uv run --with`로 임시 환경에서만 쓴다. 모델 후보는 Qwen/Qwen3-Embedding-0.6B(Apache-2.0)와 BAAI/bge-m3(MIT).

  # 레포 루트에서 실행. 로컬 DB(.env)와 build_days의 days.jsonl이 필요하다. 결과는 레포 밖에 둘 것
  uv run --with sentence-transformers python -m scripts.semantic_id.embed \\
      --days "$OUT/days.jsonl" --model Qwen/Qwen3-Embedding-0.6B --out-dir "$OUT/embed_qwen3"

결과 폴더: texts.jsonl(순서가 곧 행 번호), embeddings.npy(float16, 정규화됨), meta.json(모델, 차원, 패키지 버전).
"""

import argparse
import collections
import datetime
import json
import pathlib
import sys
import time

from scripts.semantic_id.place_text import db_place_text, log_visit_text


def load_db_places() -> list[dict]:
    # 로컬 DB의 ai_places와 카테고리. 서버 DB와 같은 상태(10,165건)여야 한다
    import pymysql

    from app.core.config import settings

    conn = pymysql.connect(
        host=settings.db_host, port=settings.db_port, user=settings.db_user,
        password=settings.db_password, database=settings.db_name, charset="utf8mb4",
    )
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, google_place_id, name, formatted_address, latitude, longitude FROM ai_places ORDER BY id")
        rows = cur.fetchall()
        cur.execute("SELECT place_id, category FROM ai_place_categories")
        cats: dict[int, list[str]] = collections.defaultdict(list)
        for pid, cat in cur.fetchall():
            cats[pid].append(cat)
    finally:
        conn.close()
    return [
        {"id": i, "google_place_id": g, "name": n, "address": a or "", "lat": float(la), "lng": float(lo),
         "categories": cats.get(i, [])}
        for i, g, n, a, la, lo in rows
    ]


def load_days(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_texts(places: list[dict], days: list[dict], with_category: bool = True) -> list[dict]:
    # DB 장소는 google_place_id, 로그 방문지는 텍스트가 키다. 같은 텍스트의 로그 방문은 한 번만 임베딩한다
    items = [
        {"source": "db", "key": p["google_place_id"],
         "text": db_place_text(p["name"], p["address"], p["categories"] if with_category else [])}
        for p in places
    ]
    seen: set[str] = set()
    for d in days:
        for v in d["visits"]:
            text = log_visit_text(v if with_category else {**v, "type_cd": ""})
            if text and text not in seen:
                seen.add(text)
                items.append({"source": "log", "key": text, "text": text})
    return items


def pick_device(requested: str | None) -> str:
    if requested:
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=pathlib.Path, required=True, help="build_days가 만든 days.jsonl")
    ap.add_argument("--model", required=True, help="예: Qwen/Qwen3-Embedding-0.6B, BAAI/bge-m3")
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--device", help="cuda, mps, cpu (없으면 자동)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--no-category", action="store_true", help="카테고리 라벨을 뺀 텍스트(안 가)로 계산한다")
    args = ap.parse_args(argv)

    if (args.out_dir / "embeddings.npy").exists():
        print(f"이미 있어서 다시 계산하지 않음: {args.out_dir / 'embeddings.npy'}", file=sys.stderr)
        return 1
    import numpy as np
    import sentence_transformers
    import torch

    items = build_texts(load_db_places(), load_days(args.days), with_category=not args.no_category)
    texts = [i["text"] for i in items]
    device = pick_device(args.device)
    print(f"텍스트 {len(texts)}개 (DB {sum(i['source'] == 'db' for i in items)}, 로그 {sum(i['source'] == 'log' for i in items)}), 장치 {device}")

    model = sentence_transformers.SentenceTransformer(args.model, device=device)
    started = time.perf_counter()
    vectors = model.encode(
        texts, batch_size=args.batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
    )
    elapsed = time.perf_counter() - started

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "embeddings.npy", vectors.astype(np.float16))
    with (args.out_dir / "texts.jsonl").open("w", encoding="utf-8") as f:
        for i in items:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")
    meta = {
        "model": args.model, "dim": int(vectors.shape[1]), "count": len(texts), "device": device,
        "with_category": not args.no_category, "seconds": round(elapsed, 1), "batch_size": args.batch_size,
        "created": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "versions": {"sentence_transformers": sentence_transformers.__version__, "torch": torch.__version__,
                     "numpy": np.__version__},
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"저장: {args.out_dir} (차원 {meta['dim']}, {elapsed:.1f}초)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
