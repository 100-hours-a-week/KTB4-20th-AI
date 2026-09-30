from dataclasses import dataclass
from datetime import timedelta

from app.core.time import get_service_today


@dataclass(frozen=True)
class Service_Date_Offset:
    """서비스 기준 오늘로부터 며칠 뒤인지를 나타냄. isoformat() 호출 시점에 날짜를 계산함.
    모듈 import 시점에 날짜를 고정하면, 테스트 실행 중 자정이 지날 때 서버 기준 오늘과 어긋나기 때문."""
    days: int

    def isoformat(self) -> str:
        return (get_service_today() + timedelta(days=self.days)).isoformat()


TODAY = Service_Date_Offset(0)
YESTERDAY = Service_Date_Offset(-1)
TOMORROW = Service_Date_Offset(1)


def resolve_date_value(date_value):
    """parametrize 인자용: Service_Date_Offset이면 실행 시점 날짜 문자열로 변환하고, 그 외(잘못된 형식 검증용 값)는 그대로 반환."""
    return date_value.isoformat() if isinstance(date_value, Service_Date_Offset) else date_value


VALID_MEMBERS = [
    {"user": {"user_id": "user_1"}, "survey_result": [3] * 15, "deal_breakers": []},
    {"user": {"user_id": "user_2"}, "survey_result": [3] * 15, "deal_breakers": []},
]