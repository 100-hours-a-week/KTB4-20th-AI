import re

import pytest

from app.photomissions.pipeline_generate import (
    build_mission,
    covers_every_place,
    new_mission_id,
    pick_primary_category,
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
