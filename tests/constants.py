from datetime import timedelta

from app.core.time import get_service_today

TODAY = get_service_today()
YESTERDAY = TODAY - timedelta(days=1)
TOMORROW = TODAY + timedelta(days=1)

VALID_MEMBERS = [
    {"user": {"user_id": "user_1"}, "survey_result": [3] * 15, "deal_breakers": []},
    {"user": {"user_id": "user_2"}, "survey_result": [3] * 15, "deal_breakers": []},
]