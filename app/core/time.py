from datetime import date, datetime
from zoneinfo import ZoneInfo

SERVICE_TIMEZONE = ZoneInfo("Asia/Seoul")


def get_service_today() -> date:
    return datetime.now(tz=SERVICE_TIMEZONE).date()