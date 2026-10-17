"""학습·평가용 입력 파일을 만든다 (M3 T3-1, ADR-0003, ADR-0005).

만드는 파일(결과 폴더, 레포 밖):
  places.json     DB 장소마다 지역, 카테고리, 평점, 평가 수, 품질 필터 통과 여부, 규칙 점수, Semantic ID 코드
  visit_map.json  로그 방문지 텍스트 → [가장 비슷한 DB 장소(google_place_id), 유사도]

방문지가 DB 범위 안인지는 유사도 0.607(DB 장소끼리 가장 가까운 유사도의 하위 5%) 이상으로 판정한다(T2-6, 확정).
범위 밖 방문은 정답에서 빼고 가장 가까운 장소의 코드를 앞 문맥에만 쓴다.

  uv run --with numpy python -m scripts.semantic_id.train_data --embed-dir "$V/m2/embed_bgem3_nocat" \\
      --rqvae-dir "$V/m2/rqvae_bgem3_nocat" --days "$V/m1/days.jsonl" --out-dir "$V/m3"
"""

import argparse
import json
import math
import pathlib
import sys
from collections.abc import Iterable

from scripts.semantic_id.place_text import district_of, log_visit_text
from scripts.semantic_id.sequences import region_of
from scripts.semantic_id.tokens import encode_day

DB_RANGE_THRESHOLD = 0.607
MAX_VISITS = 6  # 서비스가 하루에 만드는 장소는 5~6곳이다. 로그의 하루는 최대 23곳이라, 앞 6곳까지만 쓴다
RATINGS_WEIGHT = 0.6  # app/trips/services/place_selection.py의 규칙 점수와 같은 값
USER_RATING_COUNT_WEIGHT = 0.4
MIN_RATING_EXCLUSIVE = 3.0
MIN_USER_RATING_COUNT = 10


def region_of_address(formatted_address: str) -> str | None:
    return region_of(district_of(formatted_address))


def rule_score(rating: float, count: int) -> float:
    return rating * RATINGS_WEIGHT + math.log(count + 1) * USER_RATING_COUNT_WEIGHT


def passes_quality_filter(status: str | None, rating: float | None, count: int | None) -> bool:
    return status == "OPERATIONAL" and rating is not None and count is not None and rating > MIN_RATING_EXCLUSIVE and count >= MIN_USER_RATING_COUNT


def export_places(semantic_ids: dict[str, list[int]]) -> list[dict]:
    """로컬 DB를 조회 전용으로 읽어 장소 표를 만든다."""
    import pymysql

    from app.core.config import settings

    conn = pymysql.connect(
        host=settings.db_host, port=settings.db_port, user=settings.db_user,
        password=settings.db_password, database=settings.db_name, charset="utf8mb4",
    )
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, google_place_id, name, formatted_address, rating, user_rating_count, business_status FROM ai_places ORDER BY id")
        rows = cur.fetchall()
        cur.execute("SELECT place_id, category FROM ai_place_categories ORDER BY place_id, category")
        categories: dict[int, list[str]] = {}
        for pid, cat in cur.fetchall():
            categories.setdefault(pid, []).append(cat)
    finally:
        conn.close()
    out = []
    for pid, gid, name, address, rating, count, status in rows:
        out.append({
            "google_place_id": gid, "name": name, "region": region_of_address(address or ""),
            "categories": categories.get(pid, []),
            "rating": float(rating) if rating is not None else None, "user_rating_count": count,
            "passes_filter": passes_quality_filter(status, rating, count),
            "rule_score": rule_score(float(rating), count) if rating is not None and count is not None else None,
            "code": semantic_ids.get(gid),
        })
    return out


def nearest_places(vectors, items: list[dict], texts: Iterable[str]) -> dict[str, list]:
    """로그 텍스트 → [가장 비슷한 DB 장소 google_place_id, 코사인 유사도]."""
    db_rows = [n for n, i in enumerate(items) if i["source"] == "db"]
    db_keys = [items[n]["key"] for n in db_rows]
    matrix = vectors[db_rows]
    row_of = {i["key"]: n for n, i in enumerate(items) if i["source"] == "log"}
    out: dict[str, list] = {}
    wanted = [t for t in dict.fromkeys(texts) if t in row_of]
    for start in range(0, len(wanted), 2048):
        chunk = wanted[start:start + 2048]
        sims = vectors[[row_of[t] for t in chunk]] @ matrix.T
        best = sims.argmax(1)
        for i, t in enumerate(chunk):
            out[t] = [db_keys[int(best[i])], round(float(sims[i, best[i]]), 5)]
    return out


def build_examples(days: Iterable[dict], visit_map: dict[str, list], codes_by_gid: dict[str, list[int]],
                   threshold: float = DB_RANGE_THRESHOLD, max_visits: int = MAX_VISITS) -> list[dict]:
    """하루마다 토큰열, 손실 마스크, 방문별 장소를 만든다. 앞 max_visits곳까지만 쓴다. 지도에 없는 방문이 있는 하루는 버린다."""
    out = []
    for d in days:
        gids, targets, codes, sims = [], [], [], []
        ok = True
        for v in d["visits"][:max_visits]:
            m = visit_map.get(log_visit_text(v))
            if m is None or m[0] not in codes_by_gid:
                ok = False
                break
            gids.append(m[0])
            sims.append(m[1])
            targets.append(m[1] >= threshold)
            codes.append(codes_by_gid[m[0]])
        if not ok or not gids:
            continue
        tokens, mask = encode_day(d["preference"], d["region"], codes, targets)
        out.append({
            "travel_id": d["travel_id"], "traveler_id": d["traveler_id"], "region": d["region"],
            "preference": d["preference"], "tokens": tokens, "mask": mask, "gids": gids, "targets": targets, "sims": sims,
        })
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embed-dir", type=pathlib.Path, required=True)
    ap.add_argument("--rqvae-dir", type=pathlib.Path, required=True)
    ap.add_argument("--days", type=pathlib.Path, required=True)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = ap.parse_args(argv)
    places_path, map_path = args.out_dir / "places.json", args.out_dir / "visit_map.json"
    if places_path.exists() or map_path.exists():
        print(f"이미 있어서 덮어쓰지 않음: {args.out_dir}", file=sys.stderr)
        return 1
    import numpy as np

    semantic_ids = json.loads((args.rqvae_dir / "db_semantic_ids.json").read_text(encoding="utf-8"))
    vectors = np.load(args.embed_dir / "embeddings.npy").astype("float32")
    items = [json.loads(line) for line in (args.embed_dir / "texts.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    days = [json.loads(line) for line in args.days.read_text(encoding="utf-8").splitlines() if line.strip()]
    texts = [log_visit_text(v) for d in days for v in d["visits"]]
    visit_map = nearest_places(vectors, items, texts)
    places = export_places(semantic_ids)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    places_path.write_text(json.dumps(places, ensure_ascii=False) + "\n", encoding="utf-8")
    map_path.write_text(json.dumps(visit_map, ensure_ascii=False) + "\n", encoding="utf-8")
    mapped = sum(1 for t in texts if t in visit_map and visit_map[t][1] >= DB_RANGE_THRESHOLD)
    print(f"장소 {len(places)}개(필터 통과 {sum(p['passes_filter'] for p in places)}), 방문지 텍스트 {len(visit_map)}개, "
          f"방문 {len(texts)}건 중 대응 {mapped}건({mapped / len(texts):.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
