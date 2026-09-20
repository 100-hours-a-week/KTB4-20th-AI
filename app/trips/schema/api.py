from enum import Enum
from typing import Generic, TypeVar
from pydantic import BaseModel

T = TypeVar("T")

class E_Response_Status_Code(int, Enum):
    OK = 200
    INVALID_INPUT = 422
    SERVER_ERROR = 500

class Response(BaseModel, Generic[T]):
    status_code: E_Response_Status_Code
    data: T