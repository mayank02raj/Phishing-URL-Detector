"""
scripts/benchmark.py
Throughput and latency benchmark for the API. Spawns N concurrent workers
that hammer /predict for a fixed duration and reports p50/p95/p99 + RPS.

Usage:
    python scripts/benchmark.py --url http://localhost:8000 --workers 8 \
        --duration 30
"""

from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import time

import httpx

SAMPLE_URLS = [
    "http://paypa1-secure.tk/login",
    "https://www.google.com/",
    "http://192.168.1.1:8080/admin/wp-login.php",
    "https://github.com/anthropics/anthropic-sdk-python",
    "http://login-microsoft-update.cf/auth?token=xyz",
    "https://docs.python.org/3/library/asyncio.html",
    "http://amazon-account-verify.gq/signin",
    "https://en.wikipedia.org/wiki/Phishing",
    "http://bit.ly/3xEvIl",
    "https://stackoverflow.com/questions/tagged/python",
]


async def worker(client: httpx.AsyncClient, url: str, deadline: float,
                 latencies: list[float], counts: dict[str, int]):
    while time.perf_counter() < deadline:
        target = random.choice(SAMPLE_URLS)
        t0 = time.perf_counter()
        try:
            r = await client.post(f"{url}/predict",
                                  json={"url": target}, timeout=5.0)
            latency = (time.perf_counter() - t0) * 1000
            latencies.append(latency)
            if r.status_code == 200:
                counts["ok"] += 1
            else:
                counts["error"] += 1
        except Exception:
            counts["error"] += 1


async def run(url: str, workers: int, duration: float):
    latencies: list[float] = []
    counts = {"ok": 0, "error": 0}
    deadline = time.perf_counter() + duration

    async with httpx.AsyncClient() as client:
        # Warmup
        await client.get(f"{url}/health")

        tasks = [
            worker(client, url, deadline, latencies, counts)
            for _ in range(workers)
        ]
        t0 = time.perf_counter()
        await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - t0

    if not latencies:
        print("No successful requests")
        return

    latencies.sort()
    n = len(latencies)
    p50 = latencies[n // 2]
    p95 = latencies[int(n * 0.95)]
    p99 = latencies[int(n * 0.99)]

    print(f"\n=== Benchmark Results ===")
    print(f"  Target          {url}")
    print(f"  Workers         {workers}")
    print(f"  Duration        {elapsed:.1f}s")
    print(f"  Successful      {counts['ok']}")
    print(f"  Errors          {counts['error']}")
    print(f"  Throughput      {counts['ok'] / elapsed:.1f} req/s")
    print(f"\n  Latency (ms)")
    print(f"    mean          {statistics.mean(latencies):.2f}")
    print(f"    p50           {p50:.2f}")
    print(f"    p95           {p95:.2f}")
    print(f"    p99           {p99:.2f}")
    print(f"    max           {max(latencies):.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--duration", type=float, default=30.0)
    args = p.parse_args()
    asyncio.run(run(args.url, args.workers, args.duration))


if __name__ == "__main__":
    main()
