import re

import pytest

from app.core.config import Settings
from app.photomissions import pipeline_generate
from app.photomissions.pipeline_generate import (
    assemble_missions,
    build_mission,
    covers_every_place,
    new_mission_id,
    pick_primary_category,
    split_places,
)
from app.photomissions.schemas import (
    DisplayName,
    MissionBatch,
    MissionDescription,
    MissionPlace,
)


def _place(matched_preferences: list[str]) -> MissionPlace:
    return MissionPlace(
        id="places/ChIJ123",
        displayName=DisplayName(text="첨성대", languageCode="ko"),
        selected_for=[],
        matched_preferences=matched_preferences,
    )


def test_mission_id_format():
    assert re.fullmatch(r"ms_[0-9a-f]{12}", new_mission_id())


def test_primary_category_is_first_preference():
    assert pick_primary_category(["HISTORY_CULTURE", "FOOD"]) == "HISTORY_CULTURE"


def test_primary_category_none_when_empty():
    assert pick_primary_category([]) is None


def test_build_mission_maps_fields():
    place = _place(["HISTORY_CULTURE"])
    result = MissionDescription(number=1, description="첨성대 정면이 보이게 찍기", scope="GROUP")

    mission = build_mission(place, result)

    assert mission.place_id == place.id
    assert mission.description == result.description
    assert mission.scope == "GROUP"
    assert mission.primary_category == "HISTORY_CULTURE"
    assert mission.mission_id.startswith("ms_")


def test_build_mission_without_preferences():
    result = MissionDescription(number=1, description="첨성대 정면이 보이게 찍기", scope="PERSONAL")
    assert build_mission(_place([]), result).primary_category is None


def _batch(*numbers: int) -> MissionBatch:
    return MissionBatch(missions=[
        MissionDescription(number=n, description="첨성대 정면이 보이게 찍기", scope="GROUP")
        for n in numbers
    ])


def test_covers_every_place_in_any_order():
    assert covers_every_place(_batch(3, 1, 2), place_count=3) is True


@pytest.mark.parametrize(
    "numbers",
    [
        (1, 2),  # 장소 하나 빠짐
        (1, 2, 2),  # 개수는 맞지만 한 장소가 두 번
        (1, 2, 2, 3),  # 모든 장소가 있지만 한 장소가 두 번
        (0, 1, 2),  # 없는 번호
        (1, 2, 3, 4),  # 장소보다 많음
    ],
)
def test_covers_every_place_rejects_mismatch(numbers):
    assert covers_every_place(_batch(*numbers), place_count=3) is False


def _generate_settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_generate_deadline_default_is_shorter_than_backend_timeout():
    # 백엔드 읽기 타임아웃(30초) 안에 우리가 먼저 끝내야 원인별 코드(504 등)가 백엔드에 남는다
    s = _generate_settings()
    assert pipeline_generate.settings_errors(s) == []
    assert s.generate_deadline_seconds < 30


@pytest.mark.parametrize("deadline", [0, -1, float("nan"), float("inf")])
def test_generate_deadline_must_be_positive_finite(deadline):
    s = _generate_settings(generate_deadline_seconds=deadline)
    assert pipeline_generate.settings_errors(s)
    with pytest.raises(RuntimeError, match="GENERATE_DEADLINE_SECONDS"):
        pipeline_generate.check_settings(s)


def _places(n: int) -> list[MissionPlace]:
    return [
        MissionPlace(id=f"p{i}", displayName=DisplayName(text=f"장소{i}", languageCode="ko"),
                     selected_for=[], matched_preferences=[])
        for i in range(n)
    ]


@pytest.mark.parametrize(
    ("count", "sizes"),
    [(1, [1]), (8, [8]), (9, [5, 4]), (15, [8, 7]), (16, [8, 8]), (20, [7, 7, 6])],
)
def test_split_places_evenly_within_limit(count, sizes):
    places = _places(count)
    chunks = split_places(places)
    assert [len(c) for c in chunks] == sizes
    assert [p for c in chunks for p in c] == places  # 순서를 지키고 빠짐·중복 없이 이어 붙는다


def test_assemble_restores_request_order_across_chunks():
    # 묶음마다 번호가 1부터 다시 시작하고, 모델이 번호 순서를 섞어 돌려줘도 요청 순서대로 짝지어야 한다
    places = _places(5)
    chunks = [places[:3], places[3:]]

    def batch(*pairs: tuple[int, str]) -> MissionBatch:
        return MissionBatch(missions=[MissionDescription(number=n, description=d, scope="PERSONAL") for n, d in pairs])

    missions = assemble_missions(chunks, [batch((3, "c"), (1, "a"), (2, "b")), batch((2, "e"), (1, "d"))])
    assert [(m.place_id, m.description) for m in missions] == [
        ("p0", "a"), ("p1", "b"), ("p2", "c"), ("p3", "d"), ("p4", "e")]
