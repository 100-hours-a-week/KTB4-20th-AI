import pymysql

from app.core.config import settings


def get_connection():
    # 부하 테스트 baseline: 커넥션 풀 없이 호출마다 새 연결을 연다. 닫는 책임은 호출한 쪽의 close()에 있다
    return pymysql.connect(
        host=settings.db_host,
        port=settings.db_port,
        user=settings.db_user,
        password=settings.db_password,
        database=settings.db_name,
        charset="utf8mb4",
    )
