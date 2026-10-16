import collections

import pytest

from scripts.semantic_id import code_dict, project_logs


def test_dictionary_maps_code_to_place_and_rejects_duplicates():
    ids = {"g2": [1, 2, 3, 1], "g1": [1, 2, 3, 0], "g3": [9, 9, 9, 0]}
    assert code_dict.build_dictionary(ids) == {"1-2-3-0": "g1", "1-2-3-1": "g2", "9-9-9-0": "g3"}
    with pytest.raises(ValueError, match="두 장소"):
        code_dict.build_dictionary({"a": [1, 2, 3, 0], "b": [1, 2, 3, 0]})


def test_code_key_format():
    assert code_dict.code_key((12, 87, 3, 0)) == "12-87-3-0"


def test_sha256_and_revision(tmp_path):
    f = tmp_path / "w.bin"
    f.write_bytes(b"abc")
    assert code_dict.sha256_of(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    snap = tmp_path / "models--BAAI--bge-m3" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    assert code_dict.hf_revision("BAAI/bge-m3", tmp_path) == "abc123"
    (tmp_path / "models--BAAI--bge-m3" / "snapshots" / "def456").mkdir()
    assert code_dict.hf_revision("BAAI/bge-m3", tmp_path) is None  # 개정이 둘이고 refs/main이 없으면 모호하다
    refs = tmp_path / "models--BAAI--bge-m3" / "refs"
    refs.mkdir()
    (refs / "main").write_text("abc123\n", encoding="utf-8")
    assert code_dict.hf_revision("BAAI/bge-m3", tmp_path) == "abc123"  # refs/main이 있으면 그 개정
    assert code_dict.hf_revision("none/none", tmp_path) is None


def test_payload_records_the_conditions():
    payload = code_dict.build_payload(
        {"model": "BAAI/bge-m3", "dim": 1024, "text_format": "이름 | 시도 시군구"},
        {"levels": 3, "codebook": 256}, "sha", {"g1": [1, 2, 3, 0]}, "rev",
    )
    v = payload["version"]
    assert v["embedding_model"] == "BAAI/bge-m3" and v["embedding_revision"] == "rev" and v["places"] == 1
    assert v["text_format"] == "이름 | 시도 시군구"  # meta에 없으면 기본 형식
    assert payload["codes"] == {"1-2-3-0": "g1"}


def test_nearest_db_and_excluded_share():
    np = pytest.importorskip("numpy")
    items = [
        {"source": "db", "key": "g1"}, {"source": "db", "key": "g2"},
        {"source": "log", "key": "close"}, {"source": "log", "key": "far"},
    ]
    vectors = np.array([[1, 0], [0, 1], [0.99, 0.14], [0.6, 0.6]], dtype="float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    near = project_logs.nearest_db(vectors, items)
    assert near["close"][1] == "g1" and near["close"][0] > 0.95
    assert near["far"][0] < 0.8
    counts = collections.Counter({"close": 3, "far": 1})
    assert project_logs.excluded_share(near, counts, 0.9) == 0.25
    assert project_logs.percentile([0.1, 0.2, 0.3, 0.4], 50) == 0.3


def test_db_nearest_other_excludes_self():
    np = pytest.importorskip("numpy")
    items = [{"source": "db", "key": "a"}, {"source": "db", "key": "b"}, {"source": "db", "key": "c"}, {"source": "log", "key": "x"}]
    vectors = np.array([[1, 0], [0.9, 0.436], [0, 1], [1, 0]], dtype="float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    got = project_logs.db_nearest_other(vectors, items)
    assert len(got) == 3
    assert abs(got[0] - 0.9) < 0.01 and got[0] < 0.999  # 자기 자신(1.0)이 아니라 b와의 유사도
    assert got[2] < 0.5  # c는 모두와 멀다
