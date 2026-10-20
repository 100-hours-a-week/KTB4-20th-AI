import asyncio
import json

import httpx

from scripts import bench_runpod


def test_summarize_and_stats_ignore_failed_requests():
    s = bench_runpod.summarize([1.0, 2.0, 3.0, 4.0])
    assert s["n"] == 4 and s["mean"] == 2.5 and s["max"] == 4.0
    st = bench_runpod.stats([
        {"status": "COMPLETED", "total": 1.0, "delayTime": 0.6, "executionTime": 0.07, "worker": "w1", "providers": ["CUDAExecutionProvider"]},
        {"status": "FAILED", "total": 9.0, "error": "x"},
    ])
    assert st["completed"] == 1 and st["total"]["n"] == 1 and st["workers"] == ["w1"] and st["failures"] == ["x"]


def test_call_polls_status_until_completed_and_never_returns_the_key():
    calls = []

    def handler(request):
        calls.append((request.method, request.url.path))
        if request.url.path.endswith("/runsync"):
            return httpx.Response(200, json={"id": "j1", "status": "IN_QUEUE"})
        return httpx.Response(200, json={"id": "j1", "status": "COMPLETED", "delayTime": 5000, "executionTime": 300,
                                         "workerId": "w", "output": {"places": ["a", None], "providers": ["CUDAExecutionProvider"]}})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await bench_runpod.call(client, "https://x/v2/e", "SECRET", b"{}")

    result = asyncio.run(run())
    assert result["status"] == "COMPLETED" and result["polls"] >= 1 and result["places"] == 1
    assert result["delayTime"] == 5.0 and result["executionTime"] == 0.3
    assert "SECRET" not in json.dumps(result)
    assert calls[0] == ("POST", "/v2/e/runsync") and calls[1][1] == "/v2/e/status/j1"
