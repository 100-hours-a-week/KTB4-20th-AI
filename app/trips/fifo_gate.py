"""요청을 도착한 순서대로 한 번에 하나씩 처리하는 줄(FIFO)과 입장 한도.

RunPod 워커 1개가 호출을 하나씩 처리하므로, 동시에 온 요청을 줄 세워 앞 요청이 응답을 반환한 뒤에 다음 요청을 처리한다.
줄(처리 중 1건 + 대기 중)이 capacity에 차면 기다리지 않고 거절하고, 줄에서 기다린 시간이 max_wait_seconds를 넘으면 줄에서 빼고 거절한다.
"""

import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager


class QueueFull(Exception):
    """줄(처리 중 + 대기 중)이 가득 차서 기다리지 않고 거절함."""


class QueueWaitTimeout(Exception):
    """줄에서 기다린 시간이 상한을 넘어 줄에서 뺌."""


class FifoGate:
    def __init__(self, capacity: int, max_wait_seconds: float):
        self.capacity = capacity
        self.max_wait_seconds = max_wait_seconds
        self._cond = threading.Condition()
        self._waiting: deque[object] = deque()  # 도착한 순서대로 줄에 서 있는 대기 요청
        self._busy = False  # 처리 중인 요청이 있는지

    @property
    def size(self) -> int:
        """줄에 있는 요청 수(처리 중 1건 + 대기 중)."""
        with self._cond:
            return len(self._waiting) + (1 if self._busy else 0)

    @contextmanager
    def turn(self) -> Iterator[None]:
        """차례가 오면 들어가고, 블록이 끝나면(예외로 끝나도) 다음 요청에 차례를 넘긴다."""
        ticket = object()
        deadline = time.monotonic() + self.max_wait_seconds
        with self._cond:
            if len(self._waiting) + (1 if self._busy else 0) >= self.capacity:
                raise QueueFull
            self._waiting.append(ticket)
            try:
                # 줄의 맨 앞이고 처리 중인 요청이 없을 때만 들어간다
                while self._busy or self._waiting[0] is not ticket:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise QueueWaitTimeout
                    self._cond.wait(timeout=remaining)
            except BaseException:
                self._waiting.remove(ticket)  # 대기를 그만두면 줄에서 뺀다
                self._cond.notify_all()
                raise
            self._waiting.popleft()
            self._busy = True
        try:
            yield
        finally:
            with self._cond:
                self._busy = False
                self._cond.notify_all()
