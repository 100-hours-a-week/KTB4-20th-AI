from app.trips.schemas.schemas import E_Google_Place_Type, E_Preference
from app.trips.services.place_selection import NIGHT_ACTIVITY_TYPE_VALUES
from app.trips.services.taste_share import count_taste_and_discovery, is_evening_place
from tests.test_trip_selection import make_place

NIGHT_TYPE = E_Google_Place_Type(min(NIGHT_ACTIVITY_TYPE_VALUES))
TASTE = {E_Preference.HISTORY_CULTURE, E_Preference.NATURE_HEALING}


def _place(place_id, category, night=False):
    place = make_place(place_id)
    place.matched_preferences = [category]
    if night:
        place.types = [NIGHT_TYPE]
    return place


def test_counts_taste_and_discovery_for_daytime_places_only():
    day = [
        _place("a", E_Preference.HISTORY_CULTURE),
        _place("lunch", E_Preference.FOOD),
        _place("b", E_Preference.NATURE_HEALING),
        _place("c", E_Preference.CONVENIENCE_SHOPPING),  # 취향 상위 2개 밖(3·4위 또는 보충)
        _place("dinner", E_Preference.FOOD),
        _place("night", E_Preference.ACTIVITY, night=True),  # 저녁 이후
    ]
    assert count_taste_and_discovery([day], TASTE) == (2, 1)


def test_counts_sum_over_all_days_and_an_empty_trip_is_zero():
    days = [[_place("a", E_Preference.HISTORY_CULTURE)], [_place("b", E_Preference.ACTIVITY), _place("c", E_Preference.ACTIVITY)]]
    assert count_taste_and_discovery(days, TASTE) == (1, 2)
    assert count_taste_and_discovery([], TASTE) == (0, 0)
    assert count_taste_and_discovery([[]], TASTE) == (0, 0)


def test_a_daytime_activity_place_is_counted_but_an_evening_one_is_not():
    daytime = _place("day", E_Preference.ACTIVITY)
    evening = _place("night", E_Preference.ACTIVITY, night=True)
    assert not is_evening_place(daytime) and is_evening_place(evening)
    assert count_taste_and_discovery([[daytime, evening]], {E_Preference.ACTIVITY}) == (1, 0)


def test_places_without_a_matched_category_are_not_counted():
    assert count_taste_and_discovery([[make_place("x")]], TASTE) == (0, 0)
