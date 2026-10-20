from scripts import aihub_common, build_eval_category


def test_old_import_path_points_to_the_same_objects():
    # 공용 모듈로 옮긴 뒤에도 build_eval_category에서 가져가는 코드가 같은 객체를 쓴다
    for name in ("DATASETS", "read_csv_in_zip", "region_of"):
        assert getattr(build_eval_category, name) is getattr(aihub_common, name)


def test_trips_scripts_use_the_shared_module():
    # trips 추천 모델 스크립트가 포토미션 평가셋 스크립트에 기대지 않는다
    from scripts import measure_travel_sequences
    from scripts.semantic_id import sequences

    assert measure_travel_sequences.region_of is aihub_common.region_of
    assert sequences.region_of is aihub_common.region_of
