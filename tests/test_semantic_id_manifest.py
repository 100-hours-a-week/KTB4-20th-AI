import json

from scripts.semantic_id import manifest


def _make(root):
    (root / "m1").mkdir(parents=True)
    (root / "m1" / "days.jsonl").write_text("a\n", encoding="utf-8")
    (root / "m2").mkdir()
    (root / "m2" / "codes.npy").write_bytes(b"123")
    (root / ".hidden").write_text("x", encoding="utf-8")


def test_manifest_lists_files_with_hash_and_size_and_skips_hidden(tmp_path):
    _make(tmp_path)
    table = manifest.file_table(tmp_path)
    assert sorted(table) == ["m1/days.jsonl", "m2/codes.npy"]
    assert table["m2/codes.npy"]["bytes"] == 3
    assert table["m2/codes.npy"]["sha256"] == "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3"


def test_main_writes_once_and_refuses_to_overwrite(tmp_path, capsys):
    _make(tmp_path)
    assert manifest.main(["--version-dir", str(tmp_path), "--note", "테스트"]) == 0
    saved = json.loads((tmp_path / "MANIFEST.json").read_text(encoding="utf-8"))
    assert saved["note"] == "테스트" and set(saved["files"]) == {"m1/days.jsonl", "m2/codes.npy"}
    assert manifest.main(["--version-dir", str(tmp_path)]) == 1
    assert "덮어쓰지 않음" in capsys.readouterr().err


def test_verify_detects_changed_missing_and_extra_files(tmp_path, capsys):
    _make(tmp_path)
    manifest.main(["--version-dir", str(tmp_path)])
    assert manifest.verify(tmp_path) == []
    (tmp_path / "m1" / "days.jsonl").write_text("changed\n", encoding="utf-8")
    (tmp_path / "m2" / "codes.npy").unlink()
    (tmp_path / "m2" / "new.json").write_text("{}", encoding="utf-8")
    problems = manifest.verify(tmp_path)
    assert "내용이 다름: m1/days.jsonl" in problems
    assert "없는 파일: m2/codes.npy" in problems
    assert "MANIFEST에 없는 파일: m2/new.json" in problems
    assert manifest.main(["--version-dir", str(tmp_path), "--verify"]) == 1


def test_second_manifest_covers_added_folder_and_verify_reads_both(tmp_path):
    _make(tmp_path)
    manifest.main(["--version-dir", str(tmp_path)])
    first = (tmp_path / "MANIFEST.json").read_text(encoding="utf-8")
    (tmp_path / "m3").mkdir()
    (tmp_path / "m3" / "model.onnx").write_bytes(b"onnx")
    assert manifest.verify(tmp_path) == ["MANIFEST에 없는 파일: m3/model.onnx"]
    assert manifest.main(["--version-dir", str(tmp_path), "--name", "MANIFEST_m3.json", "--include", "m3"]) == 0
    assert (tmp_path / "MANIFEST.json").read_text(encoding="utf-8") == first  # 기존 MANIFEST는 그대로
    assert set(json.loads((tmp_path / "MANIFEST_m3.json").read_text(encoding="utf-8"))["files"]) == {"m3/model.onnx"}
    assert manifest.verify(tmp_path) == []
    (tmp_path / "m3" / "model.onnx").write_bytes(b"changed")
    assert manifest.verify(tmp_path) == ["내용이 다름: m3/model.onnx"]


def test_main_rejects_bad_name_and_empty_include(tmp_path, capsys):
    _make(tmp_path)
    assert manifest.main(["--version-dir", str(tmp_path), "--name", "other.json"]) == 1
    assert manifest.main(["--version-dir", str(tmp_path), "--name", "MANIFEST_x.json", "--include", "nothing"]) == 1


def test_only_new_skips_files_listed_in_existing_manifests_and_keeps_old_ones_untouched(tmp_path):
    _make(tmp_path)
    manifest.main(["--version-dir", str(tmp_path)])
    first = (tmp_path / "MANIFEST.json").read_text(encoding="utf-8")
    (tmp_path / "m2" / "late.json").write_text("{}", encoding="utf-8")
    assert manifest.main(["--version-dir", str(tmp_path), "--name", "MANIFEST_late.json", "--only-new"]) == 0
    assert (tmp_path / "MANIFEST.json").read_text(encoding="utf-8") == first
    assert set(json.loads((tmp_path / "MANIFEST_late.json").read_text(encoding="utf-8"))["files"]) == {"m2/late.json"}
    assert manifest.verify(tmp_path) == []
    assert manifest.main(["--version-dir", str(tmp_path), "--name", "MANIFEST_none.json", "--only-new"]) == 1  # 새 파일이 없으면 만들지 않는다
