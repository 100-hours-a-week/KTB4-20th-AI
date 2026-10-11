from scripts.baseline_verify import auth_headers, is_openai_compat, user_text
from scripts.measure_eval_b import build_cases, case_key, out_conflict, record_grade


def _row(photo, place="황리단길"):
    return {"photo": photo, "zip": "z.zip", "member": f"m/{photo}", "place_name": place}


def _label(photo, label="fail"):
    return {"photo": photo, "region": "경주", "category": "쇼핑·편의", "label": label}


def test_cases_take_truth_from_human_labels_and_skip_unlabeled_or_failed():
    # 사람 표시가 없는 사진(빠진 1장)과 미션 생성에 실패한 사진은 정답이 없어 측정에서 뺀다
    manifest = [_row("a.jpg"), _row("b.jpg"), _row("c.jpg")]
    missions = [
        {"photo": "a.jpg", "mission_description": "한옥 거리 찍기"},
        {"photo": "b.jpg", "mission_description": "간판 찍기"},
        {"photo": "c.jpg", "error": "502: 실패"},
    ]
    cases = build_cases(manifest, missions, [_label("a.jpg"), _label("c.jpg")], [])
    assert [(c["photo"], c["truth"]) for c in cases] == [("a.jpg", "fail")]


def test_augmented_set_uses_its_own_mission_and_truth():
    # 같은 사진이 B와 B_aug에 모두 있어도 미션·정답이 달라 별개 케이스로 센다
    manifest = [_row("a.jpg")]
    missions = [{"photo": "a.jpg", "mission_description": "원래 미션"}]
    aug = [{"photo": "a.jpg", "mission": "사진 보고 쓴 미션", "label": "success", "region": "경주", "category": "쇼핑·편의"}]
    cases = build_cases(manifest, missions, [_label("a.jpg")], aug)
    assert [(c["eval_set"], c["mission"], c["truth"]) for c in cases] == [
        ("B", "원래 미션", "fail"), ("B_aug", "사진 보고 쓴 미션", "success"),
    ]
    assert len({case_key(c) for c in cases}) == 2


def test_prompt_uses_case_mission_in_same_format_as_service():
    # 서비스(score_photo)와 같은 "목표 장소 / 미션 내용" 형식이어야 측정 결과를 서비스 성능으로 볼 수 있다
    case = {"target": "황리단길", "mission": "한옥 거리 찍기"}
    assert user_text(case) == "목표 장소: 황리단길\n미션 내용: 한옥 거리 찍기"


def test_gemini_names_go_to_gemini_and_others_to_openai_compatible_server():
    # RunPod vLLM 모델 이름(Qwen/...)처럼 목록에 없던 이름도 받을 수 있어야 한다
    assert not is_openai_compat("gemini-3.1-pro-preview")
    assert is_openai_compat("qwen2.5vl:7b")
    assert is_openai_compat("Qwen/Qwen2.5-VL-7B-Instruct")


def test_api_key_header_only_when_key_is_set():
    # 로컬 Ollama는 키 없이, RunPod은 Bearer 키로 부른다
    assert auth_headers("") == {}
    assert auth_headers("k") == {"Authorization": "Bearer k"}


def _record(model="qwen", fmt=None):
    r = {"model": model}
    if fmt:
        r["format"] = fmt
    return r


def test_out_file_with_other_model_or_format_is_refused():
    # 이미 잰 사진은 건너뛰므로, 다른 모델·형식 파일에 이어 쓰면 두 결과가 조용히 섞인다
    assert out_conflict([], "qwen", "question") is None
    assert out_conflict([_record(fmt="question")], "qwen", "question") is None
    assert out_conflict([_record()], "qwen", "question") is not None  # format 칸 없는 예전 기록은 점수 형식
    assert out_conflict([_record("gemini-3.1-pro-preview", "question")], "qwen", "question") is not None


def test_summary_grade_comes_from_answer_for_question_format():
    answer = {"judgeable": "yes", "target": "short", "condition": "na", "place": "yes"}
    assert record_grade({"answer": answer, "grade": "success"}) == "retry"
    assert record_grade({"match_score": 95}) == "success"
