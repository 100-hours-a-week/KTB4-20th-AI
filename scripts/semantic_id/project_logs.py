"""로그 방문지 중 DB에 비슷한 장소가 없는 것을 가려낸다 (M2 T2-6, ADR-0003).

양자화기는 DB 장소만으로 학습했다. DB에 비슷한 장소가 없는 방문지(집 근처 이름 없는 장소, 서비스에 없는 종류 등)는
가장 가까운 코드를 받더라도 엉뚱한 코드일 수 있다. 그런 방문지는 학습 정답(예측 대상)에서는 빼고 앞 문맥에만 쓰는 방안을 잰다.

기준: 방문지 텍스트의 임베딩과 가장 비슷한 DB 장소의 코사인 유사도(max_sim). DB 장소끼리도 가장 가까운 다른 DB 장소와의
유사도(자기 자신 제외)는 장소마다 다르다. 이 분포의 하위 백분위보다 max_sim이 낮은 방문지는 DB가 다루는 범위 밖에 있다고 본다.
정답 쌍의 유사도(같은 장소끼리, 보통 더 높다)를 경계로 삼으면 "같은 장소가 DB에 있는가"를 묻게 되어 대부분의 방문지가 제외된다.

  uv run --with numpy python -m scripts.semantic_id.project_logs --embed-dir "$OUT/embed_bgem3_nocat" \\
      --days "$OUT/days.jsonl" --pairs "$OUT/pairs_nocat.jsonl"
"""

import argparse
import collections
import json
import pathlib

from scripts.semantic_id.place_text import log_visit_text

PERCENTILES = (1, 5, 10)


def nearest_db(vectors, items: list[dict]):
    """로그 텍스트마다 가장 비슷한 DB 장소와 유사도를 구한다. 반환: {텍스트: (max_sim, google_place_id)}"""
    db_rows = [n for n, i in enumerate(items) if i["source"] == "db"]
    db_keys = [items[n]["key"] for n in db_rows]
    db_matrix = vectors[db_rows]
    out = {}
    log_rows = [n for n, i in enumerate(items) if i["source"] == "log"]
    for start in range(0, len(log_rows), 2048):
        chunk = log_rows[start:start + 2048]
        sims = vectors[chunk] @ db_matrix.T
        best = sims.argmax(1)
        for i, (r, b) in enumerate(zip(chunk, best, strict=True)):
            out[items[r]["key"]] = (float(sims[i, b]), db_keys[b])
    return out


def db_nearest_other(vectors, items: list[dict]) -> list[float]:
    """DB 장소마다 자기 자신을 뺀 가장 비슷한 DB 장소와의 유사도. DB가 다루는 범위의 촘촘함을 보여 준다."""

    db_matrix = vectors[[n for n, i in enumerate(items) if i["source"] == "db"]]
    best: list[float] = []
    for start in range(0, len(db_matrix), 1024):
        sims = db_matrix[start:start + 1024] @ db_matrix.T
        for i in range(sims.shape[0]):
            sims[i, start + i] = -1.0  # 자기 자신 제외
        best.extend(sims.max(1).tolist())
    return best


def pair_similarities(vectors, items: list[dict], pairs: list[dict]) -> list[float]:
    row = {(i["source"], i["key"]): n for n, i in enumerate(items)}
    sims = []
    for p in pairs:
        a, b = ("log", p["log_text"]), ("db", p["google_place_id"])
        if a in row and b in row:
            sims.append(float(vectors[row[a]] @ vectors[row[b]]))
    return sims


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(len(ordered) * q / 100)))]


def excluded_share(nearest: dict[str, tuple[float, str]], visit_count: collections.Counter, threshold: float) -> float:
    total = sum(visit_count.values())
    return sum(c for t, c in visit_count.items() if nearest[t][0] < threshold) / total if total else 0.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embed-dir", type=pathlib.Path, required=True)
    ap.add_argument("--days", type=pathlib.Path, required=True)
    ap.add_argument("--pairs", type=pathlib.Path, required=True)
    args = ap.parse_args(argv)
    import numpy as np

    vectors = np.load(args.embed_dir / "embeddings.npy").astype("float32")
    items = [json.loads(line) for line in (args.embed_dir / "texts.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    days = [json.loads(line) for line in args.days.read_text(encoding="utf-8").splitlines() if line.strip()]
    pairs = [json.loads(line) for line in args.pairs.read_text(encoding="utf-8").splitlines() if line.strip()]

    visits = [v for d in days for v in d["visits"]]
    visit_count = collections.Counter(log_visit_text(v) for v in visits)
    nearest = nearest_db(vectors, items)
    pair_sims = pair_similarities(vectors, items, pairs)
    matched_texts = {p["log_text"] for p in pairs}

    print(f"로그 방문 {len(visits)}건(고유 텍스트 {len(visit_count)}개), 정답 쌍 {len(pair_sims)}개")
    db_nn = db_nearest_other(vectors, items)
    quantiles = {q: percentile(db_nn, q) for q in PERCENTILES}
    print("DB 장소끼리 가장 가까운 다른 장소와의 유사도 하위 백분위: " + ", ".join(f"{q}% = {v:.3f}" for q, v in quantiles.items())
          + f" · 중앙값 {percentile(db_nn, 50):.3f}")
    print(f"참고) 정답 쌍(같은 장소) 유사도 하위 5% = {percentile(pair_sims, 5):.3f}")
    all_sims = sorted(nearest[t][0] for t in visit_count)
    print("모든 방문지의 가장 비슷한 DB 장소 유사도 분포: " + ", ".join(
        f"{q}% = {all_sims[int(len(all_sims) * q / 100)]:.3f}" for q in (1, 5, 10, 25, 50)))
    matched = collections.Counter({t: c for t, c in visit_count.items() if t in matched_texts})
    unmatched = collections.Counter({t: c for t, c in visit_count.items() if t not in matched_texts})
    print("임계값(DB 장소 하위 백분위)별 제외 비율, 방문 기준:")
    for q, th in quantiles.items():
        print(f"  하위 {q}%({th:.3f}) 미만: 전체 {excluded_share(nearest, visit_count, th):.1%}"
              f" · 정답 쌍이 아닌 방문 중 {excluded_share(nearest, unmatched, th):.1%}"
              f" · 정답 쌍 방문 중 {excluded_share(nearest, matched, th):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
