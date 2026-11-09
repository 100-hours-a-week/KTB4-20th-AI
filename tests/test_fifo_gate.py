import threading
import time

import pytest

from app.trips.fifo_gate import FifoGate, QueueFull, QueueWaitTimeout


def _wait_until(condition, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "시간 안에 조건이 만족되지 않음"
        time.sleep(0.005)


def _hold_turn(gate):
    """줄의 처리 칸을 잡고 놓지 않는 스레드를 띄운다. release.set()으로 놓는다."""
    entered, release = threading.Event(), threading.Event()

    def hold():
        with gate.turn():
            entered.set()
            release.wait(5)

    thread = threading.Thread(target=hold)
    thread.start()
    assert entered.wait(2)
    return release, thread


def _wait_in_turn(gate):
    with gate.turn():
        pass


def test_requests_are_served_one_at_a_time_in_arrival_order_and_the_kth_finishes_after_k_services():
    service = 0.03
    count = 6
    gate = FifoGate(capacity=count, max_wait_seconds=5)
    hold = threading.Event()  # 모두 줄을 설 때까지 첫 요청이 처리 칸을 놓지 않는다
    lock = threading.Lock()
    inside = 0
    max_inside = 0
    served, finished_at = [], {}

    def request(index):
        nonlocal inside, max_inside
        with gate.turn():
            with lock:
                inside += 1
                max_inside = max(max_inside, inside)
                served.append(index)
            if index == 0:
                hold.wait(5)
            time.sleep(service)
            with lock:
                inside -= 1
                finished_at[index] = time.monotonic()

    threads = []
    for index in range(count):
        thread = threading.Thread(target=request, args=(index,))
        thread.start()
        threads.append(thread)
        _wait_until(lambda expected=index + 1: gate.size == expected)  # 이 요청이 줄에 선 뒤에 다음 요청을 시작한다
    released_at = time.monotonic()
    hold.set()
    for thread in threads:
        thread.join()

    assert served == list(range(count))  # 도착한 순서대로 처리
    assert max_inside == 1  # 한 번에 하나만 처리
    for index in range(count):
        assert finished_at[index] - released_at >= (index + 1) * service * 0.9  # k번째는 서비스 시간의 k배 이후에 끝난다
    assert gate.size == 0


def test_a_full_queue_rejects_the_next_request_without_waiting():
    gate = FifoGate(capacity=2, max_wait_seconds=5)
    release, holder = _hold_turn(gate)  # 처리 중 1건
    waiter = threading.Thread(target=_wait_in_turn, args=(gate,))
    waiter.start()
    _wait_until(lambda: gate.size == 2)  # 대기 1건, 줄이 가득 참

    started = time.monotonic()
    with pytest.raises(QueueFull), gate.turn():
        pass
    assert time.monotonic() - started < 0.5  # 기다리지 않고 바로 거절

    release.set()
    holder.join()
    waiter.join()
    assert gate.size == 0


def test_a_request_that_waits_longer_than_the_limit_is_removed_from_the_queue():
    gate = FifoGate(capacity=3, max_wait_seconds=0.05)
    release, holder = _hold_turn(gate)

    with pytest.raises(QueueWaitTimeout), gate.turn():
        pass
    assert gate.size == 1  # 처리 중인 1건만 남고 기다리던 요청은 줄에서 빠졌다

    release.set()
    holder.join()
    assert gate.size == 0
    with gate.turn():  # 줄이 정상으로 돌아왔다
        pass


def test_the_turn_is_passed_on_when_the_request_fails():
    gate = FifoGate(capacity=1, max_wait_seconds=1)
    with pytest.raises(RuntimeError), gate.turn():
        raise RuntimeError("처리 중 실패")
    assert gate.size == 0
    with gate.turn():  # 실패한 요청이 차례를 넘겼다
        pass
