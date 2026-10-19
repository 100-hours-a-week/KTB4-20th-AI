import asyncio
import json
import re

import pytest
from fastapi import HTTPException

from app.core.config import Settings, settings
from app.photomissions import pipeline_generate
from app.photomissions.pipeline_generate import (
    assemble_missions,
    build_mission,
    covers_every_place,
    new_mission_id,
    pick_failure,
    pick_primary_category,
    split_places,
)
from app.photomissions.schemas import (
    DisplayName,
    MissionBatch,
    MissionDescription,
    MissionPlace,
    PhotoMissionGenerateRequest,
)

# 앱 없이 generate_missions를 직접 부르므로 테스트마다 동시 호출 상한을 새로 만든다 (conftest의 generate_started)
pytestmark = pytest.mark.usefixtures("generate_started")


def _place(matched_preferences: list[str]) -> MissionPlace:
    return MissionPlace(
        id="places/ChIJ123",
        displayName=DisplayName(text="첨성대", languageCode="ko"),
        selected_for=[],
        matched_preferences=matched_preferences,
    )


def test_mission_id_format():
    assert re.fullmatch(r"ms_[0-9a-f]{12}", new_mission_id())


def test_primary_category_is_first_preference():
    assert pick_primary_category(["HISTORY_CULTURE", "FOOD"]) == "HISTORY_CULTURE"


def test_primary_category_none_when_empty():
    assert pick_primary_category([]) is None


def test_build_mission_maps_fields():
    place = _place(["HISTORY_CULTURE"])
    result = MissionDescription(number=1, description="첨성대 정면이 보이게 찍기", scope="GROUP")

    mission = build_mission(place, result)

    assert mission.place_id == place.id
    assert mission.description == result.description
    assert mission.scope == "GROUP"
    assert mission.primary_category == "HISTORY_CULTURE"
    assert mission.mission_id.startswith("ms_")


def test_build_mission_without_preferences():
    result = MissionDescription(number=1, description="첨성대 정면이 보이게 찍기", scope="PERSONAL")
    assert build_mission(_place([]), result).primary_category is None


def _batch(*numbers: int) -> MissionBatch:
    return MissionBatch(missions=[
        MissionDescription(number=n, description="첨성대 정면이 보이게 찍기", scope="GROUP")
        for n in numbers
    ])


def test_covers_every_place_in_any_order():
    assert covers_every_place(_batch(3, 1, 2), place_count=3) is True


@pytest.mark.parametrize(
    "numbers",
    [
        (1, 2),  # 장소 하나 빠짐
        (1, 2, 2),  # 개수는 맞지만 한 장소가 두 번
        (1, 2, 2, 3),  # 모든 장소가 있지만 한 장소가 두 번
        (0, 1, 2),  # 없는 번호
        (1, 2, 3, 4),  # 장소보다 많음
    ],
)
def test_covers_every_place_rejects_mismatch(numbers):
    assert covers_every_place(_batch(*numbers), place_count=3) is False


def _generate_settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_generate_deadline_default_is_shorter_than_backend_timeout():
    # 백엔드 읽기 타임아웃(30초) 안에 우리가 먼저 끝내야 원인별 코드(504 등)가 백엔드에 남는다
    s = _generate_settings()
    assert pipeline_generate.settings_errors(s) == []
    assert s.generate_deadline_seconds < 30


@pytest.mark.parametrize("deadline", [0, -1, float("nan"), float("inf")])
def test_generate_deadline_must_be_positive_finite(deadline):
    s = _generate_settings(generate_deadline_seconds=deadline)
    assert pipeline_generate.settings_errors(s)
    with pytest.raises(RuntimeError, match="GENERATE_DEADLINE_SECONDS"):
        pipeline_generate.check_settings(s)


def _places(n: int) -> list[MissionPlace]:
    return [
        MissionPlace(id=f"p{i}", displayName=DisplayName(text=f"장소{i}", languageCode="ko"),
                     selected_for=[], matched_preferences=[])
        for i in range(n)
    ]


@pytest.mark.parametrize(
    ("count", "sizes"),
    [(1, [1]), (8, [8]), (9, [5, 4]), (15, [8, 7]), (16, [8, 8]), (20, [7, 7, 6])],
)
def test_split_places_evenly_within_limit(count, sizes):
    places = _places(count)
    chunks = split_places(places)
    assert [len(c) for c in chunks] == sizes
    assert [p for c in chunks for p in c] == places  # 순서를 지키고 빠짐·중복 없이 이어 붙는다


def test_assemble_restores_request_order_across_chunks():
    # 묶음마다 번호가 1부터 다시 시작하고, 모델이 번호 순서를 섞어 돌려줘도 요청 순서대로 짝지어야 한다
    places = _places(5)
    chunks = [places[:3], places[3:]]

    def batch(*pairs: tuple[int, str]) -> MissionBatch:
        return MissionBatch(missions=[MissionDescription(number=n, description=d, scope="PERSONAL") for n, d in pairs])

    missions = assemble_missions(chunks, [batch((3, "c"), (1, "a"), (2, "b")), batch((2, "e"), (1, "d"))])
    assert [(m.place_id, m.description) for m in missions] == [
        ("p0", "a"), ("p1", "b"), ("p2", "c"), ("p3", "d"), ("p4", "e")]


# generate_missions — 묶음을 나눠 동시에 부르고, 실패·마감·취소를 정한 규칙대로 처리하는지 (Gemini 대신 가짜)

def _request(n: int) -> PhotoMissionGenerateRequest:
    return PhotoMissionGenerateRequest(itinerary_id="it_1", places=_places(n))


def _echo(places: list[MissionPlace]) -> MissionBatch:
    # 번호를 거꾸로 섞어 돌려준다. 문구는 장소 이름이라 짝짓기가 맞았는지 바로 보인다
    return MissionBatch(missions=[
        MissionDescription(number=n, description=places[n - 1].displayName.text, scope="PERSONAL")
        for n in range(len(places), 0, -1)
    ])


class FakeWriter:
    # write_descriptions 대신 쓴다. 묶음별 동작을 첫 장소 id로 고르고, 동시에 몇 개가 돌았는지 센다
    # 순서는 sleep 시간이 아니라 Event로 정한다: all_running(묶음 n개가 호출 중) / release(테스트가 풀어 줌)
    def __init__(self, outcomes: dict | None = None, hold: bool = False, expect_running: int = 3):
        # outcomes: 첫 장소 id → Exception(바로 실패) / ("all_running", e)·("release", e)(그 신호 뒤에 실패)
        # hold: True면 성공 묶음도 취소될 때까지 끝나지 않는다
        self.outcomes = outcomes or {}
        self.hold = hold
        self.expect_running = expect_running
        self.all_running = asyncio.Event()
        self.release = asyncio.Event()
        self.calls: list[int] = []
        self.active = self.max_active = 0
        self.cancelled = 0
        self.max_tasks = 0

    async def __call__(self, places: list[MissionPlace], trace: dict | None = None) -> MissionBatch:
        self.calls.append(len(places))
        if trace is not None:
            trace["attempts"] = 1
        outcome = self.outcomes.get(places[0].id)
        if isinstance(outcome, BaseException):
            raise outcome
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.max_tasks = max(self.max_tasks, len(asyncio.all_tasks()))
        if self.active == self.expect_running:
            self.all_running.set()
        try:
            if isinstance(outcome, tuple):
                signal, error = outcome
                await (self.all_running if signal == "all_running" else self.release).wait()
                raise error
            if self.hold:
                await asyncio.Event().wait()  # 취소될 때까지
            await asyncio.sleep(0)  # 다른 묶음에 차례를 넘겨 동시에 도는 상태를 만든다
            return _echo(places)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.active -= 1


@pytest.fixture
def writer(monkeypatch):
    def install(**kwargs) -> FakeWriter:
        fake = FakeWriter(**kwargs)
        monkeypatch.setattr(pipeline_generate, "write_descriptions", fake)
        return fake
    return install


@pytest.fixture
def deadline_spy(monkeypatch):
    # generate_missions가 만드는 실제 asyncio.timeout을 붙잡아, 테스트가 마감 시각을 원하는 순간으로 당긴다
    made = []
    real = asyncio.timeout

    def spy(delay):
        made.append(real(delay))
        return made[-1]

    monkeypatch.setattr(asyncio, "timeout", spy)
    return made


class SignalSlots(asyncio.Semaphore):
    # 누가 자리를 기다리기 시작하면 알려 주는 동시 호출 상한
    def __init__(self, value: int):
        super().__init__(value)
        self.someone_waiting = asyncio.Event()

    async def acquire(self):
        if self.locked():
            self.someone_waiting.set()
        return await super().acquire()


@pytest.mark.anyio
async def test_many_places_are_split_and_returned_in_request_order(writer):
    fake = writer()
    res = await pipeline_generate.generate_missions(_request(20))
    assert sorted(fake.calls) == [6, 7, 7]
    assert [m.description for m in res.missions] == [f"장소{i}" for i in range(20)]
    assert [m.place_id for m in res.missions] == [f"p{i}" for i in range(20)]


@pytest.mark.anyio
async def test_eight_places_or_fewer_is_one_call(writer):
    fake = writer()
    res = await pipeline_generate.generate_missions(_request(8))
    assert fake.calls == [8]
    assert [m.place_id for m in res.missions] == [f"p{i}" for i in range(8)]


@pytest.mark.anyio
async def test_concurrent_requests_share_one_call_limit(writer):
    # 상한은 요청마다가 아니라 서버 프로세스 전체에 하나다. 20곳 요청 3개(묶음 9개)가 겹쳐도 동시에 3개까지만
    fake = writer()
    await asyncio.gather(*(pipeline_generate.generate_missions(_request(20)) for _ in range(3)))
    assert len(fake.calls) == 9
    assert fake.max_active == pipeline_generate.MAX_CONCURRENT_CALLS


@pytest.mark.anyio
async def test_large_request_does_not_create_a_task_per_chunk(writer):
    # 장소 100곳(묶음 13개)이어도 작업자는 3개만 만든다
    fake = writer()
    before = len(asyncio.all_tasks())
    res = await pipeline_generate.generate_missions(_request(100))
    assert len(fake.calls) == 13 and len(res.missions) == 100
    assert fake.max_tasks <= before + pipeline_generate.MAX_CONCURRENT_CALLS


def _http(code: int, **kwargs) -> HTTPException:
    return HTTPException(status_code=code, detail=f"fake {code}", **kwargs)


def test_pick_failure_prefers_429_then_smallest_chunk():
    first_429, later_429 = _http(429), _http(429)
    assert pick_failure({2: first_429, 0: _http(502)}) is first_429
    assert pick_failure({3: later_429, 1: first_429}) is first_429  # 429가 여럿이면 묶음 번호가 작은 쪽
    smallest = _http(504)
    assert pick_failure({1: _http(502), 0: smallest}) is smallest


@pytest.mark.anyio
async def test_one_chunk_failure_fails_whole_request_and_cancels_the_rest(writer):
    # 백엔드는 장소마다 미션이 다 있어야 받으므로 일부만 돌려주지 않는다. 20곳 → 묶음 [p0~p6], [p7~p13], [p14~p19]
    fake = writer(hold=True, outcomes={"p7": ("all_running", _http(502))})
    with pytest.raises(HTTPException) as exc:
        await pipeline_generate.generate_missions(_request(20))
    assert exc.value.status_code == 502
    assert fake.cancelled == 2


@pytest.mark.anyio
async def test_observed_429_wins_and_keeps_retry_after(writer):
    # 두 묶음이 기다리지 않고 바로 실패하면 둘 다 관측된다. 429와 Retry-After를 그대로 돌려준다
    rate_limited = _http(429, headers={"Retry-After": "30"})
    writer(outcomes={"p0": _http(502), "p7": rate_limited})
    with pytest.raises(HTTPException) as exc:
        await pipeline_generate.generate_missions(_request(20))
    assert exc.value is rate_limited
    assert exc.value.headers == {"Retry-After": "30"}


@pytest.mark.anyio
async def test_program_error_is_not_hidden_as_http_failure(writer):
    # 프로그램 오류와 HTTP 실패가 함께 나면 502·429로 바꾸지 않고 프로그램 오류를 그대로 올린다
    writer(outcomes={"p0": _http(429), "p7": KeyError("bug")})
    with pytest.raises(ExceptionGroup) as exc:
        await pipeline_generate.generate_missions(_request(20))
    assert [type(e) for e in exc.value.exceptions] == [KeyError]


@pytest.mark.anyio
async def test_deadline_while_waiting_for_a_slot_is_504_without_calling(writer, monkeypatch):
    fake = writer()
    monkeypatch.setattr(settings, "generate_deadline_seconds", 0.05)
    slots = pipeline_generate._call_slots
    for _ in range(pipeline_generate.MAX_CONCURRENT_CALLS):  # 다른 요청들이 자리를 다 잡고 있는 상황
        await slots.acquire()
    with pytest.raises(HTTPException) as exc:
        async with asyncio.timeout(2):  # 마감이 동작하지 않으면 영영 기다리므로, 멈추지 않고 실패하게 한다
            await pipeline_generate.generate_missions(_request(3))
    assert exc.value.status_code == 504
    assert fake.calls == []


@pytest.mark.anyio
async def test_deadline_while_calling_is_504_and_cancels(writer, deadline_spy):
    fake = writer(hold=True)
    task = asyncio.create_task(pipeline_generate.generate_missions(_request(20)))
    await fake.all_running.wait()
    deadline_spy[0].reschedule(asyncio.get_running_loop().time())  # 세 묶음이 호출 중일 때 마감
    with pytest.raises(HTTPException) as exc:
        await task
    assert exc.value.status_code == 504
    assert fake.cancelled == 3


@pytest.mark.anyio
async def test_outside_cancel_cleans_up_running_chunks(writer):
    # 바깥에서 취소되면(예: 서버 종료) 그대로 취소로 전달하고, 진행 중인 묶음도 남기지 않는다
    fake = writer(hold=True)
    before = len(asyncio.all_tasks())
    task = asyncio.create_task(pipeline_generate.generate_missions(_request(100)))
    await fake.all_running.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert fake.cancelled == 3
    assert len(fake.calls) == 3  # 차례가 오지 않은 묶음은 시작하지 않았다
    assert len(asyncio.all_tasks()) == before


@pytest.mark.anyio
async def test_outside_cancel_while_waiting_for_a_slot(writer, caplog):
    # 다른 요청들이 자리를 다 잡고 있을 때 취소되면, 호출 없이 정리되고 기다린 시간이 남는다
    fake = writer()
    slots = SignalSlots(pipeline_generate.MAX_CONCURRENT_CALLS)
    pipeline_generate._call_slots = slots
    for _ in range(pipeline_generate.MAX_CONCURRENT_CALLS):
        await slots.acquire()
    with caplog.at_level("INFO", logger="uvicorn.error"):
        task = asyncio.create_task(pipeline_generate.generate_missions(_request(3)))
        await slots.someone_waiting.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert fake.calls == []
    [line] = _generate_logs(caplog)
    [chunk] = line["chunks"]
    assert (chunk["status"], chunk["cancelled_while"], chunk["attempts"], chunk["call_sec"]) == (
        "cancelled", "waiting", 0, None)
    assert chunk["wait_sec"] is not None
    assert (line["final_status"], line["outcome"]) == (None, "cancelled")


@pytest.mark.anyio
async def test_outside_cancel_racing_a_chunk_failure_stays_a_cancel(writer):
    # 묶음이 429로 실패하는 순간 바깥 취소가 겹쳐도, 취소를 429로 바꾸지 않는다 (Codex 리뷰에서 재현한 경우)
    fake = writer(outcomes={"p0": ("release", _http(429))}, expect_running=1)
    task = asyncio.create_task(pipeline_generate.generate_missions(_request(3)))
    await fake.all_running.wait()
    fake.release.set()  # 묶음이 먼저 실패하고
    task.cancel()  # 같은 순간 바깥에서 취소한다
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.anyio
async def test_deadline_racing_a_chunk_failure_is_504(writer, deadline_spy):
    # 묶음이 502로 실패하는 순간 마감이 겹치면 마감(504)으로 끝낸다 (Codex 리뷰에서 재현한 경우)
    fake = writer(outcomes={"p0": ("release", _http(502))}, expect_running=1)
    task = asyncio.create_task(pipeline_generate.generate_missions(_request(3)))
    await fake.all_running.wait()
    fake.release.set()
    deadline_spy[0].reschedule(asyncio.get_running_loop().time())
    with pytest.raises(HTTPException) as exc:
        await task
    assert exc.value.status_code == 504


@pytest.mark.anyio
async def test_generate_needs_start(writer):
    writer()
    pipeline_generate.close()
    with pytest.raises(RuntimeError, match="start"):
        await pipeline_generate.generate_missions(_request(3))


def _generate_logs(caplog) -> list[dict]:
    return [json.loads(r.getMessage().removeprefix("generate ")) for r in caplog.records
            if r.getMessage().startswith("generate {")]


@pytest.mark.anyio
async def test_log_for_success_has_each_chunk(writer, caplog):
    writer()
    with caplog.at_level("INFO", logger="uvicorn.error"):
        await pipeline_generate.generate_missions(_request(20))
    [line] = _generate_logs(caplog)
    assert (line["places"], line["final_status"], line["outcome"], line["error_kind"]) == (20, 200, "ok", None)
    assert [c["places"] for c in line["chunks"]] == [7, 7, 6]
    assert {c["status"] for c in line["chunks"]} == {"ok"}
    assert all(c["attempts"] == 1 and c["wait_sec"] is not None and c["call_sec"] is not None for c in line["chunks"])
    assert line["deadline_sec"] == settings.generate_deadline_seconds and line["total_sec"] >= 0
    assert "장소" not in json.dumps(line, ensure_ascii=False)  # 장소 이름은 남기지 않는다


@pytest.mark.anyio
async def test_log_marks_failed_cancelled_and_not_started_chunks(writer, caplog):
    # 100곳 = 묶음 13개. 묶음 1이 실패하면 0·2는 취소되고, 나머지 10개는 차례가 오지 않는다
    writer(hold=True, outcomes={"p8": ("all_running", _http(502))})
    with caplog.at_level("INFO", logger="uvicorn.error"), pytest.raises(HTTPException):
        await pipeline_generate.generate_missions(_request(100))
    [line] = _generate_logs(caplog)
    statuses = [c["status"] for c in line["chunks"]]
    assert statuses[:3] == ["cancelled", "failed", "cancelled"]
    assert [c["cancelled_while"] for c in line["chunks"][:3]] == ["running", None, "running"]
    assert set(statuses[3:]) == {"not_started"}
    assert line["chunks"][1]["error_status"] == 502
    assert (line["final_status"], line["outcome"], line["error_kind"]) == (502, "error", "fake 502")


@pytest.mark.anyio
async def test_log_for_deadline(writer, caplog, deadline_spy):
    fake = writer(hold=True, expect_running=1)
    with caplog.at_level("INFO", logger="uvicorn.error"):
        task = asyncio.create_task(pipeline_generate.generate_missions(_request(3)))
        await fake.all_running.wait()
        deadline_spy[0].reschedule(asyncio.get_running_loop().time())
        with pytest.raises(HTTPException):
            await task
    [line] = _generate_logs(caplog)
    assert (line["final_status"], line["error_kind"]) == (504, "미션 문구 생성 시간 초과")
    assert (line["chunks"][0]["status"], line["chunks"][0]["cancelled_while"]) == ("cancelled", "running")
