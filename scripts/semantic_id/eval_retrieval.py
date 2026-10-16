"""정렬 검증: 로그 방문지 텍스트로 DB 장소를 검색했을 때 같은 장소가 상위에 오는지 잰다 (M2 T2-3, ADR-0003).

지표 1: 검색 성공률. 로그 방문지의 임베딩과 가장 가까운 DB 장소 k개 안에 정답 쌍의 DB 장소가 있는 비율(k=1, 5, 10)과 MRR.
비교 기준으로 같은 지역의 무작위 순위(우연 수준)를 함께 적는다. 쌍이 이름 유사도로 골라져서 이름을 쓰는 임베딩에 유리하다.

  uv run --with numpy python -m scripts.semantic_id.eval_retrieval --embed-dir "$OUT/embed_qwen3" --pairs "$OUT/pairs.jsonl"
"""

import argparse
import json
import pathlib

KS = (1, 5, 10)


def load_embeddings(embed_dir: pathlib.Path):
    import numpy as np

    vectors = np.load(embed_dir / "embeddings.npy").astype("float32")
    items = [json.loads(line) for line in (embed_dir / "texts.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return vectors, items


def retrieval_metrics(vectors, items: list[dict], pairs: list[dict]) -> dict:
    import numpy as np

    db_index = {i["key"]: n for n, i in enumerate(items) if i["source"] == "db"}
    log_index = {i["key"]: n for n, i in enumerate(items) if i["source"] == "log"}
    db_rows = np.array(sorted(db_index.values()))
    db_matrix = vectors[db_rows]
    row_of_db = {gid: r for r, gid in enumerate(k for k, _ in sorted(db_index.items(), key=lambda kv: kv[1]))}

    ranks = []
    weights = []
    for p in pairs:
        if p["log_text"] not in log_index or p["google_place_id"] not in row_of_db:
            continue
        q = vectors[log_index[p["log_text"]]]
        scores = db_matrix @ q
        target = row_of_db[p["google_place_id"]]
        rank = int((scores > scores[target]).sum()) + 1  # 동점은 유리하게 세지 않는다(위에 있는 것만 센다)
        ranks.append(rank)
        weights.append(p.get("visits", 1))
    if not ranks:
        return {"pairs": 0}
    r = np.array(ranks)
    w = np.array(weights, dtype="float64")
    out = {"pairs": len(r), "db_size": len(db_rows), "mrr": float((1.0 / r).mean()),
           "median_rank": int(np.median(r))}
    for k in KS:
        out[f"top{k}"] = float((r <= k).mean())
        out[f"top{k}_visit_weighted"] = float(((r <= k) * w).sum() / w.sum())
    out["random_top10"] = 10 / len(db_rows)  # 우연 수준
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embed-dir", type=pathlib.Path, required=True)
    ap.add_argument("--pairs", type=pathlib.Path, required=True, help="pairs.jsonl")
    args = ap.parse_args(argv)
    vectors, items = load_embeddings(args.embed_dir)
    pairs = [json.loads(line) for line in args.pairs.read_text(encoding="utf-8").splitlines() if line.strip()]
    meta = json.loads((args.embed_dir / "meta.json").read_text(encoding="utf-8"))
    m = retrieval_metrics(vectors, items, pairs)
    print(f"모델 {meta['model']} (차원 {meta['dim']}, 카테고리 라벨 {'있음' if meta['with_category'] else '없음'})")
    if not m["pairs"]:
        print("평가할 쌍이 없음")
        return 1
    print(f"정답 쌍 {m['pairs']}개, DB 장소 {m['db_size']}개")
    print("  검색 성공률(쌍 기준): " + ", ".join(f"top{k} {m[f'top{k}']:.1%}" for k in KS) + f" · MRR {m['mrr']:.3f} · 순위 중앙값 {m['median_rank']}")
    print("  방문 수 가중: " + ", ".join(f"top{k} {m[f'top{k}_visit_weighted']:.1%}" for k in KS))
    print(f"  우연 수준 top10: {m['random_top10']:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
