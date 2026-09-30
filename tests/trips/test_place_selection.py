import pytest

from app.trips.schemas.schemas import E_Breaker, E_Region
from tests.constants import (
    TODAY,
    TOMORROW,
    VALID_MEMBERS,
    YESTERDAY,
    resolve_date_value,
)

PLACE_SELECTION_URL = "/trips/place-selection"

"""
    Request Body 구분
    - 지역
    - 여행 기간
    - 내부 요인
      1. 사용자별 설문 답변 결과 목록
      2. 사용자별 회피 조건 선택 목록
    - 외부 요인
"""

# ---------------- 1. 지역 ----------------
class TestRegion:
    # 1. 각 상황별 Request Body 조합
    def make_region_request_body(self, region_value, omit_field=None):
        request_body = {
            "region": region_value,
            "start_date": TODAY.isoformat(),
            "end_date": TODAY.isoformat(),
            "members": VALID_MEMBERS,
        }
        if omit_field:
            request_body.pop(omit_field)
        return request_body

    # 2. 성공 케이스
    @pytest.mark.parametrize("region_value", [
        pytest.param(region.value, id=region.name) for region in E_Region
    ])
    def test_region_valid(self, test_client, request_headers, region_value):
        response = test_client.post(
            PLACE_SELECTION_URL,
            json=self.make_region_request_body(region_value),
            headers=request_headers,
        )
        assert response.status_code == 200, response.json()

    # 3. 실패 케이스
    @pytest.mark.parametrize('region_value, omit_field', [
        # 비즈니스 규칙 위반 (미지원 지역)
        pytest.param("강원도", None, id="region_not_supported"),

        # 구문 위반
        pytest.param("", None, id="region_empty_string"),
        pytest.param(E_Region.SEOUL.name, None, id="region_enum_name_instead_of_value"),

        # 타입 위반
        pytest.param(123, None, id="region_int"),
        pytest.param(12.3, None, id="region_float"),
        pytest.param([E_Region.SEOUL.value], None, id="region_list"),
        pytest.param({"region": E_Region.SEOUL.value}, None, id="region_dict"),

        # None / 누락
        pytest.param(None, None, id="region_none"),
        pytest.param(E_Region.SEOUL.value, "region", id="region_missing"),
    ])
    def test_region_invalid(self, test_client, request_headers, region_value, omit_field):
        response = test_client.post(
            PLACE_SELECTION_URL,
            json=self.make_region_request_body(region_value, omit_field),
            headers=request_headers,
        )
        assert response.status_code == 422, response.json()
        error_locations = [error['loc'][-1] for error in response.json()['detail']]
        assert "region" in error_locations, response.json()

# ---------------- 2. 여행 기간 (v1에는 하루용) ----------------
class TestPeriod:
    # 1. 고정 파라미터 작성
    REGION = E_Region.SEOUL.value

    # 2. 각 상황별 Request Body(dict) 조합
    def make_period_request_body(self, start_date_value, end_date_value, omit_field=None):
        request_body = {
            "region": self.REGION,
            "start_date": resolve_date_value(start_date_value),
            "end_date": resolve_date_value(end_date_value),
            "members": VALID_MEMBERS,
        }
        if omit_field:
            request_body.pop(omit_field)
        return request_body

    # 3. 성공 케이스
    @pytest.mark.parametrize("start_date_value, end_date_value", [
        pytest.param(TODAY, TODAY, id="today_single_day"),
        pytest.param(TOMORROW, TOMORROW, id="future_single_day"),
    ])
    def test_period_valid(self, test_client, request_headers, start_date_value, end_date_value):
        response = test_client.post(
            PLACE_SELECTION_URL,
            json=self.make_period_request_body(start_date_value, end_date_value),
            headers=request_headers
        )
        assert response.status_code == 200, response.json()

    # 4. 실패 케이스
    @pytest.mark.parametrize("start_date_value, end_date_value, omit_field, expected_error_field", [
        # 비즈니스 규칙 위반
        pytest.param(YESTERDAY, YESTERDAY, None, "start_date", id="start_in_past"),
        pytest.param(TOMORROW, TODAY, None, "end_date", id="end_before_start"),

        # 구문 위반
        pytest.param("wrong_format", TODAY, None, "start_date", id="start_wrong_format"),
        pytest.param("2026/09/27", TODAY, None, "start_date", id="start_wrong_separator"),
        pytest.param(TODAY, "2026-02-30", None, "end_date", id="end_invalid_calendar_date"),

        # 타입 위반
        pytest.param(["2026-09-27"], TODAY, None, "start_date", id="start_list"),
        pytest.param({"start": "2026-09-27"}, TODAY, None, "start_date", id="start_dict"),

        # None / 누락
        pytest.param(None, TODAY, None, "start_date", id="start_none"),
        pytest.param(TODAY, TODAY, "end_date", "end_date", id="end_missing"),
    ])
    def test_period_invalid(self, test_client, request_headers, start_date_value, end_date_value, omit_field, expected_error_field):
        response = test_client.post(
            PLACE_SELECTION_URL,
            json=self.make_period_request_body(start_date_value, end_date_value, omit_field),
            headers=request_headers
        )
        assert response.status_code == 422, response.json()
        error_locations = [error["loc"][-1] for error in response.json()["detail"]]
        assert expected_error_field in error_locations, response.json()


# ---------------- 3. 내부 요인 (사용자별 설문 답변 결과 목록, 사용자별 회피 조건 선택 목록) ----------------
class TestInternalFactors:
    # 1. 고정 파라미터 작성
    REGION = E_Region.SEOUL.value

    # 2. 각 상황별 Request Body 조합
    def make_member(self, user_id, override_fields=None, omit_member_field=None):
        member = {"user": {"user_id": user_id}, "survey_result": [3] * 15, "deal_breakers": []}
        if override_fields:
            member.update(override_fields)
        if omit_member_field:
            member.pop(omit_member_field)
        return member

    def make_members_with_first_member_changed(self, override_fields=None, omit_member_field=None):
        # members[0]만 변경하고 members[1]은 정상값으로 고정
        return [
            self.make_member("user_1", override_fields, omit_member_field),
            self.make_member("user_2"),
        ]

    def make_internal_factors_request_body(self, members_value, omit_field=None):
        request_body = {
            "region": self.REGION,
            "start_date": TODAY.isoformat(),
            "end_date": TODAY.isoformat(),
            "members": members_value,
        }
        if omit_field:
            request_body.pop(omit_field)
        return request_body

    def post_place_selection(self, test_client, request_headers, request_body):
        return test_client.post(PLACE_SELECTION_URL, json=request_body, headers=request_headers)

    def assert_validation_error(self, response, expected_error_field):
        assert response.status_code == 422, response.json()
        error_locations = [error["loc"] for error in response.json()["detail"]]
        assert any(expected_error_field in location for location in error_locations), response.json()

    # 3. survey_result
    @pytest.mark.parametrize("override_fields", [
        pytest.param({"survey_result": [1] * 15}, id="survey_result_all_minimum"),
        pytest.param({"survey_result": [5] * 15}, id="survey_result_all_maximum"),
        pytest.param({"survey_result": [1, 2, 3, 4, 5] * 3}, id="survey_result_mixed"),
    ])
    def test_survey_result_valid(self, test_client, request_headers, override_fields):
        members_value = self.make_members_with_first_member_changed(override_fields)
        response = self.post_place_selection(
            test_client, request_headers, self.make_internal_factors_request_body(members_value)
        )
        assert response.status_code == 200, response.json()

    @pytest.mark.parametrize("override_fields, omit_member_field", [
        # 비즈니스 규칙 위반 (1~5 범위)
        pytest.param({"survey_result": [0] + [3] * 14}, None, id="survey_result_below_minimum"),
        pytest.param({"survey_result": [6] + [3] * 14}, None, id="survey_result_above_maximum"),

        # 구문 위반 (문항 수)
        pytest.param({"survey_result": [3] * 14}, None, id="survey_result_14_items"),
        pytest.param({"survey_result": [3] * 16}, None, id="survey_result_16_items"),
        pytest.param({"survey_result": []}, None, id="survey_result_empty"),

        # 타입 위반
        pytest.param({"survey_result": [3.5] + [3] * 14}, None, id="survey_result_float_item"),
        pytest.param({"survey_result": ["a"] + [3] * 14}, None, id="survey_result_string_item"),
        pytest.param({"survey_result": {"question_1": 3}}, None, id="survey_result_dict"),
        pytest.param({"survey_result": ["3"] + [3] * 14}, None, id="survey_result_numeric_string_item"),

        # None / 누락
        pytest.param({"survey_result": None}, None, id="survey_result_none"),
        pytest.param(None, "survey_result", id="survey_result_missing"),
    ])
    def test_survey_result_invalid(self, test_client, request_headers, override_fields, omit_member_field):
        members_value = self.make_members_with_first_member_changed(override_fields, omit_member_field)
        response = self.post_place_selection(
            test_client, request_headers, self.make_internal_factors_request_body(members_value)
        )
        self.assert_validation_error(response, "survey_result")

    # 4. deal_breakers
    @pytest.mark.parametrize("override_fields, omit_member_field", [
        pytest.param({"deal_breakers": [breaker.value]}, None, id=breaker.name) for breaker in E_Breaker
    ] + [
        pytest.param({"deal_breakers": [breaker.value for breaker in E_Breaker]}, None, id="deal_breakers_all"),
        pytest.param({"deal_breakers": []}, None, id="deal_breakers_empty"),
        pytest.param(None, "deal_breakers", id="deal_breakers_missing_defaults_to_empty"),
    ])
    def test_deal_breakers_valid(self, test_client, request_headers, override_fields, omit_member_field):
        members_value = self.make_members_with_first_member_changed(override_fields, omit_member_field)
        response = self.post_place_selection(
            test_client, request_headers, self.make_internal_factors_request_body(members_value)
        )
        assert response.status_code == 200, response.json()

    @pytest.mark.parametrize("override_fields", [
        # 구문 위반
        pytest.param({"deal_breakers": [E_Breaker.NOISY_PLACE.name]}, id="deal_breakers_enum_name_instead_of_value"),
        pytest.param({"deal_breakers": ["소음"]}, id="deal_breakers_not_in_enum"),

        # 타입 위반
        pytest.param({"deal_breakers": E_Breaker.SEAFOOD.value}, id="deal_breakers_string_not_list"),
        pytest.param({"deal_breakers": [None]}, id="deal_breakers_none_item"),

        # None
        pytest.param({"deal_breakers": None}, id="deal_breakers_none"),
    ])
    def test_deal_breakers_invalid(self, test_client, request_headers, override_fields):
        members_value = self.make_members_with_first_member_changed(override_fields)
        response = self.post_place_selection(
            test_client, request_headers, self.make_internal_factors_request_body(members_value)
        )
        self.assert_validation_error(response, "deal_breakers")

    # 5. user
    @pytest.mark.parametrize("override_fields, omit_member_field, expected_error_field", [
        # 타입 위반
        pytest.param({"user": {"user_id": 123}}, None, "user_id", id="user_id_int"),
        pytest.param({"user": "user_1"}, None, "user", id="user_string_not_object"),

        # None / 누락
        pytest.param({"user": {"user_id": None}}, None, "user_id", id="user_id_none"),
        pytest.param({"user": {}}, None, "user_id", id="user_id_missing"),
        pytest.param(None, "user", "user", id="user_missing"),
    ])
    def test_user_invalid(self, test_client, request_headers, override_fields, omit_member_field, expected_error_field):
        members_value = self.make_members_with_first_member_changed(override_fields, omit_member_field)
        response = self.post_place_selection(
            test_client, request_headers, self.make_internal_factors_request_body(members_value)
        )
        self.assert_validation_error(response, expected_error_field)

    # 6. members
    @pytest.mark.parametrize("user_ids", [
        pytest.param(["user_1"], id="members_minimum_one"),
        pytest.param(["user_1", "user_2"], id="members_two"),
        pytest.param(["user_1", "user_2", "user_3"], id="members_three"),
        pytest.param([f"user_{index}" for index in range(1, 9)], id="members_maximum_eight"),
    ])
    def test_members_valid(self, test_client, request_headers, user_ids):
        members_value = [self.make_member(user_id) for user_id in user_ids]
        response = self.post_place_selection(
            test_client, request_headers, self.make_internal_factors_request_body(members_value)
        )
        assert response.status_code == 200, response.json()

    @pytest.mark.parametrize("user_ids", [
        # 비즈니스 규칙 위반 (최소 인원)
        pytest.param([], id="members_empty"),
        
        # 비즈니스 규칙 위반 (최대 인원)
        pytest.param([f"user_{index}" for index in range(1, 10)], id="members_nine"),
        
        # 비즈니스 규칙 위반 (user_id 중복)
        pytest.param(["user_1", "user_1"], id="members_duplicate_user_id"),
        pytest.param(["user_1", "user_2", "user_1"], id="members_duplicate_user_id_non_adjacent"),
    ])
    def test_members_rule_invalid(self, test_client, request_headers, user_ids):
        members_value = [self.make_member(user_id) for user_id in user_ids]
        response = self.post_place_selection(
            test_client, request_headers, self.make_internal_factors_request_body(members_value)
        )
        self.assert_validation_error(response, "members")

    @pytest.mark.parametrize("members_value, omit_field", [
        # 타입 위반
        pytest.param({"user": {"user_id": "user_1"}}, None, id="members_dict_not_list"),
        pytest.param(["user_1", "user_2"], None, id="members_string_item"),

        # None / 누락
        pytest.param(None, None, id="members_none"),
        pytest.param([], "members", id="members_missing"),
    ])
    def test_members_invalid(self, test_client, request_headers, members_value, omit_field):
        response = self.post_place_selection(
            test_client, request_headers, self.make_internal_factors_request_body(members_value, omit_field)
        )
        self.assert_validation_error(response, "members")

# ---------------- 4. 외부 요인 (보류, v1에는 없음) ----------------
class TestExternalFactors:
    pass
    
# ---------------- 5. 조합 테스트 ----------------
import copy
import json
from itertools import combinations
from pathlib import Path

from app.trips.schemas.schemas import E_Preference
from app.trips.services.place_selection import DIRECT_EXCLUDE_MAP

REPRESENTATIVE_SURVEYS_PATH = Path(__file__).resolve().parent / "data" / "representative_surveys.json"
REPRESENTATIVE_SURVEYS = json.loads(REPRESENTATIVE_SURVEYS_PATH.read_text(encoding="utf-8"))

# 회피 조건 부분집합 전체 (공집합 포함, 2^n개)
DEAL_BREAKER_SUBSETS = [
    list(subset)
    for subset_size in range(len(E_Breaker) + 1)
    for subset in combinations(E_Breaker, subset_size)
]

@pytest.mark.exhaustive
class TestCombination:
    @pytest.mark.parametrize("region_value", [
        pytest.param(region.value, id=region.name) for region in E_Region
    ])
    @pytest.mark.parametrize("deal_breakers_value", [
        pytest.param(
            [breaker.value for breaker in subset],
            id="+".join(breaker.name for breaker in subset) or "NONE",
        )
        for subset in DEAL_BREAKER_SUBSETS
    ])
    @pytest.mark.parametrize("representative_survey", [
        pytest.param(case, id=case["case_id"]) for case in REPRESENTATIVE_SURVEYS
    ])
    def test_combination(self, test_client, request_headers, region_value, deal_breakers_value, representative_survey):
        # 1. 요청 body 구성 (회피 조건은 합집합으로 처리되므로 첫 번째 멤버에만 지정)
        members_value = copy.deepcopy(representative_survey["members"])
        members_value[0]["deal_breakers"] = deal_breakers_value
        request_body = {
            "region": region_value,
            "start_date": TODAY.isoformat(),
            "end_date": TODAY.isoformat(),
            "members": members_value,
        }

        response = test_client.post(PLACE_SELECTION_URL, json=request_body, headers=request_headers)
        assert response.status_code == 200, response.json()
        places = response.json()["data"]["places"]

        # 2. invariant 1: 반환 장소 수는 5 또는 6
        assert len(places) in (5, 6), [place["matched_preferences"] for place in places]

        # 3. invariant 2: 점심(2번째), 저녁(5번째)은 FOOD
        assert E_Preference.FOOD.value in places[1]["matched_preferences"], places[1]
        assert E_Preference.FOOD.value in places[4]["matched_preferences"], places[4]

        # 4. invariant 3: 회피 조건 type을 가진 장소 없음
        excluded_types: set[str] = set()
        for breaker_value in deal_breakers_value:
            excluded_types |= DIRECT_EXCLUDE_MAP.get(E_Breaker(breaker_value), set())
        for place in places:
            assert not excluded_types & set(place["types"]), place

        # 5. invariant 4: 장소 중복 없음
        place_ids = [place["id"] for place in places]
        assert len(place_ids) == len(set(place_ids)), place_ids