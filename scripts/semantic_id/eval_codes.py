"""정렬 검증 2: 같은 장소로 판정된 (로그 방문지, DB 장소) 쌍이 같은 코드를 받는지 잰다 (M2 T2-3, ADR-0003).

앞 d개 코드가 같은 비율(d=1, 2, 3)을 쌍 기준으로 내고, 우연 수준(서로 다른 DB 장소 두 개의 일치율)과 함께 적는다.
쌍이 이름 유사도로 골라져서 이름을 쓰는 임베딩에 유리하다.

  uv run --with numpy python -m scripts.semantic_id.eval_codes --embed-dir "$OUT/embed_qwen3" \\
      --rqvae-dir "$OUT/rqvae_qwen3" --pairs "$OUT/pairs.jsonl"
"""

import argparse
import json
import pathlib

from scripts.semantic_id import codes as codes_mod


def code_agreement(embed_dir: pathlib.Path, rqvae_dir: pathlib.Path, pairs: list[dict]) -> dict:
    import numpy as np

    items = [json.loads(line) for line in (embed_dir / "texts.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    all_codes = np.load(rqvae_dir / "codes.npy")
    row = {(i["source"], i["key"]): n for n, i in enumerate(items)}
    db_codes = {i["key"]: tuple(int(c) for c in all_codes[n]) for n, i in enumerate(items) if i["source"] == "db"}

    def agreement(subset: list[dict]) -> dict:
        got = []
        for p in subset:
            a, b = ("log", p["log_text"]), ("db", p["google_place_id"])
            if a in row and b in row:
                got.append((tuple(int(c) for c in all_codes[row[a]]), tuple(int(c) for c in all_codes[row[b]])))
        return {"pairs": len(got), **{f"first_{d}": v for d, v in codes_mod.prefix_agreement(got).items()}}

    exact = [p for p in pairs if p["similarity"] >= 0.999]
    hard = [p for p in pairs if p["similarity"] < 0.999]
    weighted_hits = {1: 0.0, 2: 0.0, 3: 0.0}
    total_w = 0
    for p in pairs:
        a, b = ("log", p["log_text"]), ("db", p["google_place_id"])
        if a in row and b in row:
            ca, cb = all_codes[row[a]], all_codes[row[b]]
            for d in weighted_hits:
                weighted_hits[d] += p["visits"] * (tuple(ca[:d]) == tuple(cb[:d]))
            total_w += p["visits"]
    return {
        "all": agreement(pairs), "exact_name": agreement(exact), "different_name": agreement(hard),
        "visit_weighted": {f"first_{d}": (weighted_hits[d] / total_w if total_w else 0.0) for d in weighted_hits},
        "random": {f"first_{d}": v for d, v in codes_mod.random_pair_agreement(db_codes).items()},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embed-dir", type=pathlib.Path, required=True)
    ap.add_argument("--rqvae-dir", type=pathlib.Path, required=True)
    ap.add_argument("--pairs", type=pathlib.Path, required=True)
    args = ap.parse_args(argv)
    pairs = [json.loads(line) for line in args.pairs.read_text(encoding="utf-8").splitlines() if line.strip()]
    r = code_agreement(args.embed_dir, args.rqvae_dir, pairs)
    meta = json.loads((args.rqvae_dir / "meta.json").read_text(encoding="utf-8"))
    print(f"{args.embed_dir.name} → {args.rqvae_dir.name}")
    print(f"  DB 코드 충돌: {meta['collisions_db']['colliding_places']}곳 ({meta['collisions_db']['colliding_ratio']:.1%}), 최대 묶음 {meta['collisions_db']['max_group']}")
    for name in ("all", "exact_name", "different_name"):
        v = r[name]
        print(f"  {name:15s} {v['pairs']:5d}쌍 · 첫 코드 {v['first_1']:.1%} · 앞 2개 {v['first_2']:.1%} · 앞 3개 {v['first_3']:.1%}")
    w = r["visit_weighted"]
    print(f"  방문 가중       첫 코드 {w['first_1']:.1%} · 앞 2개 {w['first_2']:.1%} · 앞 3개 {w['first_3']:.1%}")
    rd = r["random"]
    print(f"  우연 수준(서로 다른 DB 장소 두 개)  첫 코드 {rd['first_1']:.2%} · 앞 2개 {rd['first_2']:.3%} · 앞 3개 {rd['first_3']:.4%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
