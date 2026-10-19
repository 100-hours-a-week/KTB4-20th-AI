import asyncio
import json

import pytest
from fastapi import HTTPException
from google.genai import errors

from app.photomissions import pipeline_generate
from app.photomissions.prompts import MISSION_SYSTEM_PROMPT
from app.photomissions.schemas import MissionBatch, MissionDescription
from scripts.generate_eval_missions import (
    PROMPTS,
    describe_cause,
    main,
    make_batches,
    plan_sha,
    prompt_sha,
    read_done,
    sample_places,
    to_request,
    use_prompt,
    write_meta,
)
from scripts.mission_prompt_v1 import MISSION_SYSTEM_PROMPT_V1


def _row(photo, region, category="FOOD", place="식당"):
    return {"photo": photo, "region": region, "category": category, "place_name": place}


def test_batches_keep_one_region_and_cover_every_photo():
    # 실제 동선처럼 한 묶음에는 같은 지역 장소만 들어가고, 모든 사진이 한 번씩 들어간다
    rows = [_row(f"s{i}.jpg", "서울") for i in range(7)] + [_row(f"j{i}.jpg", "제주") for i in range(3)]
    batches = make_batches(rows, batch_size=5, seed=1)
    assert [len(b) for b in batches] == [5, 2, 3]
    assert all(len({r["region"] for r in b}) == 1 for b in batches)
    assert sorted(r["photo"] for b in batches for r in b) == sorted(r["photo"] for r in rows)


def test_batches_are_deterministic():
    # 다시 실행해도 같은 묶음이어야 이미 만든 묶음을 건너뛸 수 있다
    rows = [_row(f"p{i}.jpg", "부산") for i in range(12)]
    first = [[r["photo"] for r in b] for b in make_batches(rows, batch_size=5, seed=3)]
    second = [[r["photo"] for r in b] for b in make_batches(list(reversed(rows)), batch_size=5, seed=3)]
    assert first == second


def test_request_uses_photo_as_place_id_and_category_as_preference():
    request = to_request([_row("a.jpg", "경주", "ACTIVITY", "경주엑스포대공원")], "eval_b_000")
    place = request.places[0]
    assert request.itinerary_id == "eval_b_000"
    assert place.id == "a.jpg"
    assert place.displayName.text == "경주엑스포대공원"
    assert place.matched_preferences == ["ACTIVITY"]


def test_sample_places_drops_repeated_place_and_spreads_cells():
    # 같은 방문지 사진은 한 곳으로 치고, 지역×카테고리 칸을 돌아가며 고른다
    rows = (
        [_row(f"f{i}.jpg", "서울", "FOOD", f"식당{i}") for i in range(5)]
        + [_row("dup.jpg", "서울", "FOOD", "식당0")]
        + [_row(f"n{i}.jpg", "서울", "NATURE_HEALING", f"공원{i}") for i in range(1)]
        + [_row(f"j{i}.jpg", "제주", "FOOD", f"제주식당{i}") for i in range(2)]
    )
    picked = sample_places(rows, 4, seed=7)
    assert len(picked) == 4
    assert len({(r["region"], r["category"], r["place_name"]) for r in picked}) == 4
    assert {(r["region"], r["category"]) for r in picked} == {("서울", "FOOD"), ("서울", "NATURE_HEALING"), ("제주", "FOOD")}
    assert [r["photo"] for r in picked] == [r["photo"] for r in sample_places(list(reversed(rows)), 4, seed=7)]
    assert len(sample_places(rows, 100, seed=7)) == 8  # 고유 방문지 8곳보다 많이 달라고 해도 8곳


def test_prompt_versions_are_distinct_and_v1_is_frozen():
    assert PROMPTS == {"current": MISSION_SYSTEM_PROMPT, "v1": MISSION_SYSTEM_PROMPT_V1}
    assert prompt_sha(MISSION_SYSTEM_PROMPT) != prompt_sha(MISSION_SYSTEM_PROMPT_V1)
    # 비교 기준이 몰래 바뀌지 않게 이전 프롬프트 내용을 고정한다
    assert prompt_sha(MISSION_SYSTEM_PROMPT_V1) == "b300ef18be34"


def test_use_prompt_changes_what_generate_sends(monkeypatch):
    monkeypatch.setattr(pipeline_generate, "MISSION_SYSTEM_PROMPT", pipeline_generate.MISSION_SYSTEM_PROMPT)
    sent = []

    async def fake_generate_structured(system, user, response_schema, is_valid=None, trace=None):
        sent.append(system)
        return MissionBatch(missions=[MissionDescription(number=1, description="기와지붕 크게 찍기", scope="GROUP")])

    monkeypatch.setattr(pipeline_generate, "generate_structured", fake_generate_structured)
    places = to_request([_row("a.jpg", "경주")], "b0").places
    assert use_prompt("v1") == prompt_sha(MISSION_SYSTEM_PROMPT_V1)
    asyncio.run(pipeline_generate.write_descriptions(places))
    use_prompt("current")
    asyncio.run(pipeline_generate.write_descriptions(places))
    assert sent == [MISSION_SYSTEM_PROMPT_V1, MISSION_SYSTEM_PROMPT]


EXPECT = {"prompt": "v1", "prompt_sha": "aaa", "plan_sha": "ppp"}


def _write(path, records):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")


@pytest.mark.parametrize(
    "record",
    [
        {"photo": "a.jpg", "mission_description": "x"},  # 지문 없는 예전 결과
        {"photo": "a.jpg", "mission_description": "x", **EXPECT, "prompt_sha": "bbb"},  # 다른 프롬프트
        {"photo": "a.jpg", "mission_description": "x", **EXPECT, "plan_sha": "qqq"},  # 다른 표본·묶음
    ],
)
def test_compare_run_refuses_to_resume_different_results(tmp_path, record):
    out = tmp_path / "run.jsonl"
    _write(out, [record])
    with pytest.raises(SystemExit):
        read_done(out, EXPECT)


def test_compare_run_resumes_only_its_own_successes(tmp_path):
    out = tmp_path / "run.jsonl"
    _write(out, [{"photo": "a.jpg", "mission_description": "x", **EXPECT}, {"photo": "b.jpg", "error": "502", **EXPECT}])
    assert read_done(out, EXPECT) == {"a.jpg"}


def test_legacy_run_keeps_old_resume_behavior(tmp_path):
    out = tmp_path / "missions.jsonl"
    _write(out, [{"photo": "a.jpg", "mission_description": "x"}, {"photo": "b.jpg", "error": "502"}])
    assert read_done(out) == {"a.jpg"}


def test_plan_sha_changes_with_seed_size_or_order():
    rows = [_row(f"p{i}.jpg", "부산") for i in range(7)]
    base = plan_sha(make_batches(rows, 5, 1), 5, 1)
    assert base == plan_sha(make_batches(list(reversed(rows)), 5, 1), 5, 1)
    assert base != plan_sha(make_batches(rows, 5, 2), 5, 2)
    assert base != plan_sha(make_batches(rows, 3, 1), 3, 1)


@pytest.mark.parametrize(("field", "value"), [("place_name", "고친 장소명"), ("category", "ACTIVITY")])
def test_plan_sha_changes_when_gemini_input_changes(field, value):
    # 사진 이름이 같아도 Gemini에 가는 장소명·카테고리가 바뀌면 다른 실험이다
    rows = [_row(f"p{i}.jpg", "부산") for i in range(3)]
    edited = [{**rows[0], field: value}, *rows[1:]]
    assert plan_sha(make_batches(rows, 5, 1), 5, 1) != plan_sha(make_batches(edited, 5, 1), 5, 1)


def test_meta_appends_runs_and_refuses_other_experiment(tmp_path):
    meta = tmp_path / "run.jsonl.meta.json"
    write_meta(meta, EXPECT)
    write_meta(meta, EXPECT)
    saved = json.loads(meta.read_text(encoding="utf-8"))
    assert saved["prompt_sha"] == "aaa" and len(saved["runs"]) == 2
    assert {"started_at", "model", "git_commit", "git_dirty", "argv"} <= set(saved["runs"][0])
    with pytest.raises(SystemExit):
        write_meta(meta, {**EXPECT, "plan_sha": "qqq"})


@pytest.fixture
def manifest_and_split(tmp_path):
    rows = [_row(f"d{i}.jpg", "서울", "FOOD", f"식당{i}") for i in range(3)] + [_row("f0.jpg", "서울", "FOOD", "final식당")]
    manifest = tmp_path / "manifest.jsonl"
    _write(manifest, rows)
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"photos": {"dev": ["d0.jpg", "d1.jpg", "d2.jpg"], "final": ["f0.jpg"]}}), encoding="utf-8")
    return manifest, split


@pytest.mark.parametrize(
    "extra",
    [
        ["--prompt", "v1"],  # 비교 실행인데 split 파일이 없음
        ["--sample-places", "2"],  # 비교 실행이 아닌데 표본 옵션
        ["--prompt", "v1", "--split-file", "SPLIT", "--sample-places", "-1"],
        ["--batch-size", "0"],
    ],
)
def test_bad_arguments_stop_before_any_call(manifest_and_split, tmp_path, extra):
    manifest, split = manifest_and_split
    extra = [str(split) if a == "SPLIT" else a for a in extra]
    with pytest.raises(SystemExit):
        asyncio.run(main(["--manifest", str(manifest), "--out", str(tmp_path / "o.jsonl"), "--dry-run", *extra]))


def test_compare_dry_run_uses_dev_only(manifest_and_split, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(pipeline_generate, "MISSION_SYSTEM_PROMPT", pipeline_generate.MISSION_SYSTEM_PROMPT)
    manifest, split = manifest_and_split
    out = tmp_path / "o.jsonl"
    asyncio.run(main(["--manifest", str(manifest), "--split-file", str(split), "--prompt", "v1", "--out", str(out), "--dry-run"]))
    printed = capsys.readouterr().out
    assert "사진 3장" in printed and "final식당" not in printed
    assert not out.exists() and not out.with_name(out.name + ".meta.json").exists()  # dry-run은 아무것도 쓰지 않는다


def _http_error_from(cause):
    try:
        raise HTTPException(status_code=502, detail="미션 문구 생성 실패") from cause
    except HTTPException as e:
        return e


def test_failed_line_keeps_original_cause():
    # 서비스는 원인을 숨기므로, 측정 결과에는 Gemini 에러 코드나 검증 실패 이유를 남긴다
    api = errors.ServerError(503, {"error": {"code": 503, "status": "UNAVAILABLE", "message": "overloaded"}})
    assert describe_cause(_http_error_from(api)) == "ServerError 503 UNAVAILABLE: overloaded"
    assert describe_cause(_http_error_from(ValueError("Gemini 응답이 요청 내용과 맞지 않음"))) == (
        "ValueError: Gemini 응답이 요청 내용과 맞지 않음"
    )
    assert describe_cause(HTTPException(status_code=502)) == ""


def test_max_calls_does_not_count_skipped_batches_as_done(manifest_and_split, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(pipeline_generate, "MISSION_SYSTEM_PROMPT", pipeline_generate.MISSION_SYSTEM_PROMPT)
    manifest, split = manifest_and_split
    argv = ["--manifest", str(manifest), "--split-file", str(split), "--prompt", "v1",
            "--out", str(tmp_path / "o.jsonl"), "--batch-size", "1", "--max-calls", "1", "--dry-run"]
    asyncio.run(main(argv))
    assert "이미 완료 0개) → 이번에 호출할 것 1개" in capsys.readouterr().out
