"""RunPod 모델 엔드포인트의 콜드 스타트, 따뜻한 지연, 동시 요청, 요청 크기 한도를 잰다 (M5 T5-5, ADR-0004).

요청은 `make_runpod_payload`가 만든 파일을 쓴다. 주소와 키는 환경변수 또는 .env의 RUNPOD_MODEL_URL, RUNPOD_MODEL_API_KEY에서 읽고 출력하지 않는다.
호출은 비용이 나가므로 사용자가 직접 실행한다. 결과는 JSON으로 저장한다(레포 밖 폴더 권장).

  uv run python -m scripts.bench_runpod --payload-dir "$V/m3/runpod" --out "$V/m3/runpod/results_$(date +%m%d_%H%M).json" \\
      --cold 3 --cold-gap 120 --warm 20 --concurrency 4 --sizes

단계:
  cold         워커가 내려갔을 때 첫 요청. cold-gap초 쉰 뒤 보낸다(유휴 시간보다 길어야 한다)
  warm         워커가 떠 있는 상태에서 연속 요청 (서울, 제주 요청 번갈아)
  concurrency  동시에 보낸 요청의 지연과 서로 다른 워커 수
  sizes        요청 크기를 키워 가며 한도 확인 (pad_*.json). 처음 실패한 크기에서 멈춘다
"""

import argparse
import asyncio
import json
import os
import pathlib
import statistics
import sys
import time

import httpx


def load_env() -> tuple[str, str]:
    url, key = os.environ.get("RUNPOD_MODEL_URL", ""), os.environ.get("RUNPOD_MODEL_API_KEY", "")
    if not (url and key) and pathlib.Path(".env").exists():
        for line in pathlib.Path(".env").read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name == "RUNPOD_MODEL_URL" and not url:
                url = value.strip()
            if name == "RUNPOD_MODEL_API_KEY" and not key:
                key = value.strip()
    if not (url and key):
        raise SystemExit("RUNPOD_MODEL_URL, RUNPOD_MODEL_API_KEY가 필요함 (환경변수 또는 .env)")
    return url.rstrip("/"), key


def summarize(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(ordered), "mean": round(statistics.mean(ordered), 3), "p50": round(ordered[len(ordered) // 2], 3),
        "p95": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 3), "max": round(ordered[-1], 3),
    }


async def call(client: httpx.AsyncClient, url: str, key: str, body: bytes, poll_limit: float = 300.0) -> dict:
    """/runsync로 보내고, 완료 전에 돌아오면 /status로 조회한다. 지연은 초 단위."""
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    started = time.perf_counter()
    result: dict = {"bytes": len(body)}
    try:
        response = await client.post(f"{url}/runsync", content=body, headers=headers)
        result["http"] = response.status_code
        data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        polls = 0
        while data.get("status") in ("IN_QUEUE", "IN_PROGRESS") and time.perf_counter() - started < poll_limit:
            polls += 1
            await asyncio.sleep(2.0)
            status = await client.get(f"{url}/status/{data['id']}", headers=headers)
            data = status.json()
        result["polls"] = polls
        result["status"] = data.get("status")
        for field in ("delayTime", "executionTime"):
            if data.get(field) is not None:
                result[field] = data[field] / 1000
        result["worker"] = data.get("workerId")
        output = data.get("output") or {}
        result["providers"] = output.get("providers")
        result["places"] = len([p for p in output.get("places", []) if p]) if isinstance(output.get("places"), list) else None
        if data.get("status") != "COMPLETED":
            result["error"] = str(data.get("error") or data)[:300]
    except httpx.HTTPError as error:
        result["error"] = f"{type(error).__name__}"
    result["total"] = round(time.perf_counter() - started, 3)
    return result


def stats(results: list[dict]) -> dict:
    ok = [r for r in results if r.get("status") == "COMPLETED"]
    return {
        "requests": len(results), "completed": len(ok),
        "total": summarize([r["total"] for r in ok]),
        "delay": summarize([r["delayTime"] for r in ok if "delayTime" in r]),
        "execution": summarize([r["executionTime"] for r in ok if "executionTime" in r]),
        "workers": sorted({r["worker"] for r in ok if r.get("worker")}),
        "providers": sorted({",".join(r["providers"]) for r in ok if r.get("providers")}),
        "failures": [r.get("error") or r.get("http") for r in results if r.get("status") != "COMPLETED"][:5],
    }


async def main_async(args) -> dict:
    url, key = load_env()
    payloads = {p.stem: p.read_bytes() for p in pathlib.Path(args.payload_dir).glob("*.json")}
    for needed in ("realistic_seoul", "realistic_jeju"):
        if needed not in payloads:
            raise SystemExit(f"{needed}.json이 {args.payload_dir}에 없음")
    report: dict = {"url_tail": url[-8:], "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    timeout = httpx.Timeout(300.0, connect=15.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        if args.cold:
            cold = []
            for i in range(args.cold):
                print(f"[cold {i + 1}/{args.cold}] {args.cold_gap}초 쉰 뒤 요청 (워커가 내려가기를 기다림)", flush=True)
                await asyncio.sleep(args.cold_gap)
                cold.append(await call(client, url, key, payloads["realistic_seoul"]))
                print("   ", {k: cold[-1].get(k) for k in ("status", "total", "delayTime", "executionTime", "worker")}, flush=True)
            report["cold"] = {"stats": stats(cold), "runs": cold}
        if args.warm:
            await call(client, url, key, payloads["realistic_seoul"])  # 워커를 먼저 깨운다. 결과는 버린다
            warm = []
            for i in range(args.warm):
                warm.append(await call(client, url, key, payloads["realistic_seoul" if i % 2 == 0 else "realistic_jeju"]))
            report["warm"] = {"stats": stats(warm), "runs": warm}
            print("[warm]", json.dumps(report["warm"]["stats"], ensure_ascii=False), flush=True)
        if args.concurrency:
            rounds = []
            for _ in range(3):
                rounds += await asyncio.gather(*[call(client, url, key, payloads["realistic_seoul"]) for _ in range(args.concurrency)])
            report["concurrency"] = {"width": args.concurrency, "stats": stats(rounds), "runs": rounds}
            print("[concurrency]", json.dumps(report["concurrency"]["stats"], ensure_ascii=False), flush=True)
        if args.sizes:
            sizes = []
            for name in sorted((n for n in payloads if n.startswith("pad_")), key=lambda n: len(payloads[n])):
                result = await call(client, url, key, payloads[name])
                result["name"] = name
                sizes.append(result)
                print("[size]", name, {k: result.get(k) for k in ("bytes", "http", "status", "total", "error")}, flush=True)
                if result.get("status") != "COMPLETED":
                    break
            report["sizes"] = sizes
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--payload-dir", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--cold", type=int, default=0, help="콜드 스타트 시도 횟수")
    ap.add_argument("--cold-gap", type=float, default=120.0, help="콜드 시도 전 쉬는 시간(초). 유휴 시간보다 길게")
    ap.add_argument("--warm", type=int, default=0, help="따뜻한 상태 연속 요청 수")
    ap.add_argument("--concurrency", type=int, default=0, help="동시 요청 수(3라운드)")
    ap.add_argument("--sizes", action="store_true", help="요청 크기 한도 시험")
    args = ap.parse_args(argv)
    if args.out.exists():
        print(f"이미 있어서 덮어쓰지 않음: {args.out}", file=sys.stderr)
        return 1
    report = asyncio.run(main_async(args))
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
