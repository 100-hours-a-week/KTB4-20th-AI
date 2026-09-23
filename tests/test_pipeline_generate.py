import re

from app.photomissions.pipeline_generate import (
    build_mission,
    new_mission_id,
    pick_primary_category,
)
from app.photomissions.schemas import DisplayName, MissionDescription, MissionPlace


def _place(matched_preferences: list[str]) -> MissionPlace:
    return MissionPlace(
        id="places/ChIJ123",
        displayName=DisplayName(text="첨성대", languageCode="ko"),
        selected_for=[],
        matched_preferences=matched_preferences,
    )


def test_mission_id_format():
    assert re.fullmatch(r"ms_[0-9a-f]{12}", new_mission_id())


def test_mission_id_is_unique():
    assert len({new_mission_id() for _ in range(1000)}) == 1000


def test_primary_category_is_first_preference():
    assert pick_primary_category(["HISTORY_CULTURE", "FOOD"]) == "HISTORY_CULTURE"


def test_primary_category_none_when_empty():
    assert pick_primary_category([]) is None


def test_build_mission_maps_fields():
    place = _place(["HISTORY_CULTURE"])
    result = MissionDescription(description="첨성대 정면이 보이게 찍기", scope="GROUP")

    mission = build_mission(place, result)

    assert mission.place_id == place.id
    assert mission.description == result.description
    assert mission.scope == "GROUP"
    assert mission.primary_category == "HISTORY_CULTURE"
    assert mission.mission_id.startswith("ms_")


def test_build_mission_without_preferences():
    result = MissionDescription(description="첨성대 정면이 보이게 찍기", scope="PERSONAL")
    assert build_mission(_place([]), result).primary_category is None
