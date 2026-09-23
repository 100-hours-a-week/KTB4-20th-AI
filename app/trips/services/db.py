import pymysql
from dbutils.pooled_db import PooledDB

from app.core.config import settings

pool = PooledDB(
    creator=pymysql,
    maxconnections=5,
    host=settings.db_host,
    port=settings.db_port,
    user=settings.db_user,
    password=settings.db_password,
    database=settings.db_name,
    charset="utf8mb4",
)


def get_connection():
    return pool.connection()