"""
테스트 구현 목록
1. place_selection (진행 중)
2. course_recommendation (예정)
3. precheck (예정)
"""
import pytest

from app.trips.schemas.schemas import E_Region
from tests.constants import YESTERDAY, TODAY, TOMORROW

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
class TestArea:
    pass

class TestPeriod:
    # 1. 고정 파라미터 작성
    MEMBERS = [
        {'user': { 'user_id': "user_1" }, "survey_result": [3]*15, "deal_breakers": []},
        {'user': { 'user_id': "user_2" }, "survey_result": [3]*15, "deal_breakers": []},
    ]

    # 2. 각 상황별 Request Body(dict) 조합
    def make_period_request_body(self, start_date_value, end_date_value, omit_field=None):
        request_body = {
            "region": E_Region.SEOUL.value,
            "start_date": start_date_value,
            "end_date": end_date_value,
            "members": self.MEMBERS,
        }
        if omit_field:
            request_body.pop(omit_field)
        return request_body

    # 3. 성공 케이스
    @pytest.mark.parametrize("start_date_value, end_date_value", [
        pytest.param(TODAY.isoformat(), TODAY.isoformat(), id="today_single_day"),
        pytest.param(TOMORROW.isoformat(), TOMORROW.isoformat(), id="future_single_day"),
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
        pytest.param(YESTERDAY.isoformat(), YESTERDAY.isoformat(), None, "start_date", id="start_in_past"),
        pytest.param(TOMORROW.isoformat(), TODAY.isoformat(), None, "end_date", id="end_before_start"),
        
        # 구문 위반
        pytest.param("wrong_format", TODAY.isoformat(), None, "start_date", id="start_wrong_format"),
        pytest.param("2026/09/27", TODAY.isoformat(), None, "start_date", id="start_wrong_separator"),
        pytest.param(TODAY.isoformat(), "2026-02-30", None, "end_date", id="end_invalid_calendar_date"),
        
        # 타입 위반
        pytest.param(["2026-09-27"], TODAY.isoformat(), None, "start_date", id="start_list"),
        pytest.param({"start": "2026-09-27"}, TODAY.isoformat(), None, "start_date", id="start_dict"),
        
        # None / 누락
        pytest.param(None, TODAY.isoformat(), None, "start_date", id="start_none"),
        pytest.param(TODAY.isoformat(), TODAY.isoformat(), "end_date", "end_date", id="end_missing"),
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

class TestInternalFactors:
    pass

class TestExternalFactors:
    pass



# 1. place_selection
# def test_place_selction():

#     # ---------------- 1. 지역 ----------------
#     # 1. 변동 파라미터 작성
#     # 정상 케이스 - SEOUL, BUSAN, GYEONGJU, JEJU, JEONJU
#     success = [E_Region.SEOUL.value, E_Region.BUSAN.value, E_Region.GYEONGJU.value, E_Region.JEJU.value, E_Region.JEONJU.value]
#     # 실패 케이스 (그 외 모든 값)
#     #   - 타입은 같은데, 존재하지 않는 값("", "다른 지역")
#     #   - 타입이 다른 값(int, float, list, dict)
#     #   - none/필드 자체 누락
#     failure_1 = ["", "강원도"]
#     failure_2 = [123, 12.3, [E_Region.SEOUL], {"area": E_Region.SEOUL}]
#     failure_3 = [None]

#     # 2. 고정 파라미터 작성(변동 파라미터 외 파라미터는 모두 고정, "시작일, 종료일, 내부 요인, 외부 요인")
#     start_date = "2026-09-27" # 시작일
#     end_date = "2026-09-27" # 종료일
#     members = [
#         {'user': { 'user_id': "user_1" }, "survey_result": [3]*15, "deal_breakers": []},
#         {'user': { 'user_id': "user_2" }, "survey_result": [3]*15, "deal_breakers": []},
#     ]

#     # 각 상황별 Request Body(dict) 조합
#     def make_request_body_1(region_value):
#         return {
#             "region": region_value,
#             "start_date": start_date,
#             "end_date": end_date,
#             "members": members,
#         }

#     def success_region():
#         for region_value in success:
#             response = client.post(PLACE_SELECTION_URL, json=make_request_body_1(region_value), headers=HEADERS)
#             assert response.status_code == 200, f"실패: region={region_value}"

#     def failure_region():
#         for region_value in failure_1 + failure_2 + failure_3:
#             response = client.post(PLACE_SELECTION_URL, json=make_request_body_1(region_value), headers=HEADERS)
#             assert response.status_code == 422, f"실패: region={region_value}"

#     success_region()
#     failure_region()
    


#     # ---------------- 2. 여행 기간 (v1에는 하루용) ----------------
    

    

    

    
    

#     # ---------------- 3. 내부 요인 (사용자별 설문 답변 결과 목록, 사용자별 회피 조건 선택 목록) ----------------
#     # 1. 확인 파라미터 작성
#     # 정상 케이스
#     #   - (user) 사용자 ID가 올바른 형태여야 함. (규칙이 무엇인지 알아야 함)
#     #   - (survey_result) 설문조사 문항이 정확하게 15개가 맞아야 함.
#     #   - (survey_result) 모든 문항이 1~5점 사이여야 함.
#     #   - (deal_breakers) 배열 내 값이 있을 경우, "NOISY_PLACE, OUTDOOR_ACTIVITY, SEAFOOD, RELIGIOUS_FACILITY, ANIMAL_FACILITY, HEIGHT_AVERSION" 중 하나여야 함.

#     # 실패 케이스

#     # 2. 고정 파라미터 작성

#     # ---------------- 5. 외부 요인 (보류, v1에는 없음) ----------------
#     # 1. 확인 파라미터 작성
        
#     # 2. 고정 파라미터 작성
    

# 2. course_recommendation

# 3. precheck