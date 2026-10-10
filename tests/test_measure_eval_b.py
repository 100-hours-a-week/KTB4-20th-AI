from scripts.baseline_verify import user_text
from scripts.measure_eval_b import build_cases, case_key


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
