"""Asynchronous high-concurrency load testing suite for LLM inference servers.
Measures TTFT, TPOT, E2E Latency percentiles, throughput (tokens/s, req/s),
and concurrent GPU utilization under variable workloads.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import httpx

logger = logging.getLogger("load_test")


@dataclass
class LoadTestConfig:
    target_url: str = "http://127.0.0.1:8000/generate"
    concurrency: int = 4
    total_requests: int = 20
    max_new_tokens: int = 24
    stream: bool = False
    timeout_s: float = 60.0
    prompts_file: Optional[str] = None


@dataclass
class RequestMetric:
    request_id: str
    status_code: int
    ttft_ms: float
    tpot_ms: float
    e2e_ms: float
    prompt_tokens: int
    completion_tokens: int
    success: bool
    error: Optional[str] = None


def calc_percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_d = sorted(data)
    idx = (len(sorted_d) - 1) * (p / 100.0)
    floor_idx = math.floor(idx)
    ceil_idx = math.ceil(idx)
    if floor_idx == ceil_idx:
        return sorted_d[int(idx)]
    d0 = sorted_d[int(floor_idx)] * (ceil_idx - idx)
    d1 = sorted_d[int(ceil_idx)] * (idx - floor_idx)
    return d0 + d1


@dataclass
class LoadTestResult:
    config: LoadTestConfig
    total_duration_s: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    failure_rate_pct: float
    system_throughput_rps: float
    generation_throughput_tps: float
    total_tokens_throughput_tps: float
    ttft_p50_ms: float
    ttft_p90_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    ttft_avg_ms: float
    tpot_p50_ms: float
    tpot_p90_ms: float
    tpot_p95_ms: float
    tpot_p99_ms: float
    tpot_avg_ms: float
    e2e_p50_ms: float
    e2e_p90_ms: float
    e2e_p95_ms: float
    e2e_p99_ms: float
    e2e_avg_ms: float
    peak_gpu_util_pct: float = 0.0
    peak_gpu_mem_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concurrency": self.config.concurrency,
            "total_requests": self.total_requests,
            "duration_seconds": round(self.total_duration_s, 2),
            "throughput": {
                "system_rps": round(self.system_throughput_rps, 2),
                "generation_tokens_per_sec": round(self.generation_throughput_tps, 2),
                "total_tokens_per_sec": round(self.total_tokens_throughput_tps, 2),
            },
            "latency_ms": {
                "ttft": {
                    "avg": round(self.ttft_avg_ms, 2),
                    "p50": round(self.ttft_p50_ms, 2),
                    "p90": round(self.ttft_p90_ms, 2),
                    "p95": round(self.ttft_p95_ms, 2),
                    "p99": round(self.ttft_p99_ms, 2),
                },
                "tpot": {
                    "avg": round(self.tpot_avg_ms, 2),
                    "p50": round(self.tpot_p50_ms, 2),
                    "p90": round(self.tpot_p90_ms, 2),
                    "p95": round(self.tpot_p95_ms, 2),
                    "p99": round(self.tpot_p99_ms, 2),
                },
                "e2e": {
                    "avg": round(self.e2e_avg_ms, 2),
                    "p50": round(self.e2e_p50_ms, 2),
                    "p90": round(self.e2e_p90_ms, 2),
                    "p95": round(self.e2e_p95_ms, 2),
                    "p99": round(self.e2e_p99_ms, 2),
                },
            },
            "reliability": {
                "success": self.successful_requests,
                "failure": self.failed_requests,
                "failure_rate_pct": round(self.failure_rate_pct, 2),
            },
            "hardware": {
                "peak_gpu_util_pct": round(self.peak_gpu_util_pct, 1),
                "peak_gpu_mem_mb": round(self.peak_gpu_mem_mb, 1),
            },
        }

    def print_summary(self) -> None:
        """Pretty-print benchmark results to console."""
        print("\n" + "=" * 65)
        print(f"  LOAD TEST RESULTS (Concurrency={self.config.concurrency}, Total Requests={self.total_requests})")
        print("=" * 65)
        print(f" Duration:            {self.total_duration_s:.2f} s")
        print(f" Successful Requests: {self.successful_requests} / {self.total_requests} (Failures: {self.failed_requests})")
        print(f" Request Throughput:  {self.system_throughput_rps:.2f} req/s")
        print(f" Token Throughput:    {self.generation_throughput_tps:.2f} gen tokens/s ({self.total_tokens_throughput_tps:.2f} total tok/s)")
        print(f" Peak GPU Util:       {self.peak_gpu_util_pct:.1f}% (Peak VRAM: {self.peak_gpu_mem_mb:.1f} MB)")
        print("-" * 65)
        print(f" Latency Metric       Average     P50        P90        P99")
        print(f" TTFT (Time to 1st)   {self.ttft_avg_ms:6.1f} ms  {self.ttft_p50_ms:6.1f} ms  {self.ttft_p90_ms:6.1f} ms  {self.ttft_p99_ms:6.1f} ms")
        print(f" TPOT (Inter-Token)   {self.tpot_avg_ms:6.1f} ms  {self.tpot_p50_ms:6.1f} ms  {self.tpot_p90_ms:6.1f} ms  {self.tpot_p99_ms:6.1f} ms")
        print(f" E2E Turnaround       {self.e2e_avg_ms:6.1f} ms  {self.e2e_p50_ms:6.1f} ms  {self.e2e_p90_ms:6.1f} ms  {self.e2e_p99_ms:6.1f} ms")
        print("=" * 65 + "\n")


class AsyncLoadTester:
    """Orchestrates concurrent load generation against an LLM inference endpoint."""

    def __init__(self, config: LoadTestConfig) -> None:
        self.config = config
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> List[str]:
        if self.config.prompts_file and os.path.exists(self.config.prompts_file):
            with open(self.config.prompts_file, "r") as f:
                return json.load(f)
        # Fallback to local default file
        cur_dir = os.path.dirname(__file__)
        def_file = os.path.join(cur_dir, "prompts.json")
        if os.path.exists(def_file):
            with open(def_file, "r") as f:
                return json.load(f)
        return [
            "Explain continuous batching in LLM inference.",
            "Compare FP16 and INT4 group quantization trade-offs.",
            "How does memory bandwidth govern token generation speed?",
        ]

    async def _send_single_request(
        self, client: httpx.AsyncClient, req_id: str, prompt: str
    ) -> RequestMetric:
        t_start = time.time()
        ttft_ms = 0.0
        tpot_ms = 0.0
        e2e_ms = 0.0
        p_tokens = 0
        c_tokens = 0

        payload = {
            "prompt": prompt,
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": 0.7,
            "stream": self.config.stream,
        }

        try:
            if self.config.stream:
                # Streaming mode SSE
                first_tok_t: Optional[float] = None
                tokens_count = 0
                async with client.stream(
                    "POST", self.config.target_url, json=payload, timeout=self.config.timeout_s
                ) as resp:
                    if resp.status_code != 200:
                        return RequestMetric(
                            request_id=req_id,
                            status_code=resp.status_code,
                            ttft_ms=0,
                            tpot_ms=0,
                            e2e_ms=(time.time() - t_start) * 1000.0,
                            prompt_tokens=0,
                            completion_tokens=0,
                            success=False,
                            error=f"HTTP {resp.status_code}",
                        )

                    async for line in resp.aiter_lines():
                        if line.startswith("data: ") and line.strip() != "data: [DONE]":
                            now = time.time()
                            if first_tok_t is None:
                                first_tok_t = now
                            tokens_count += 1

                t_finish = time.time()
                e2e_ms = (t_finish - t_start) * 1000.0
                ttft_ms = (first_tok_t - t_start) * 1000.0 if first_tok_t else e2e_ms
                tpot_ms = (
                    ((t_finish - first_tok_t) / (tokens_count - 1)) * 1000.0
                    if (first_tok_t and tokens_count > 1)
                    else 0.0
                )
                c_tokens = tokens_count
                return RequestMetric(
                    request_id=req_id,
                    status_code=200,
                    ttft_ms=ttft_ms,
                    tpot_ms=tpot_ms,
                    e2e_ms=e2e_ms,
                    prompt_tokens=p_tokens,
                    completion_tokens=c_tokens,
                    success=True,
                )
            else:
                # Standard JSON request
                resp = await client.post(
                    self.config.target_url, json=payload, timeout=self.config.timeout_s
                )
                t_finish = time.time()
                e2e_ms = (t_finish - t_start) * 1000.0

                if resp.status_code == 200:
                    data = resp.json()
                    metrics = data.get("metrics", {})
                    usage = data.get("usage", {})
                    ttft_ms = metrics.get("ttft_ms", e2e_ms)
                    tpot_ms = metrics.get("tpot_ms", 0.0)
                    p_tokens = usage.get("prompt_tokens", 0)
                    c_tokens = usage.get("completion_tokens", self.config.max_new_tokens)
                    return RequestMetric(
                        request_id=req_id,
                        status_code=200,
                        ttft_ms=ttft_ms,
                        tpot_ms=tpot_ms,
                        e2e_ms=e2e_ms,
                        prompt_tokens=p_tokens,
                        completion_tokens=c_tokens,
                        success=True,
                    )
                else:
                    return RequestMetric(
                        request_id=req_id,
                        status_code=resp.status_code,
                        ttft_ms=0,
                        tpot_ms=0,
                        e2e_ms=e2e_ms,
                        prompt_tokens=0,
                        completion_tokens=0,
                        success=False,
                        error=f"HTTP {resp.status_code}",
                    )
        except Exception as e:
            return RequestMetric(
                request_id=req_id,
                status_code=0,
                ttft_ms=0,
                tpot_ms=0,
                e2e_ms=(time.time() - t_start) * 1000.0,
                prompt_tokens=0,
                completion_tokens=0,
                success=False,
                error=str(e),
            )

    async def run(self) -> LoadTestResult:
        """Execute concurrent load test run."""
        limits = httpx.Limits(
            max_connections=self.config.concurrency * 2,
            max_keepalive_connections=self.config.concurrency,
        )
        async with httpx.AsyncClient(limits=limits) as client:
            # Query initial GPU status if available
            base_url = "/".join(self.config.target_url.split("/")[:3])
            peak_gpu_util = 0.0
            peak_vram = 0.0
            try:
                hw_resp = await client.get(f"{base_url}/gpu/status", timeout=2.0)
                if hw_resp.status_code == 200:
                    hw_data = hw_resp.json()
                    peak_gpu_util = hw_data.get("latest", {}).get("gpu_utilization_pct", 0.0)
                    peak_vram = hw_data.get("latest", {}).get("gpu_memory_used_mb", 0.0)
            except Exception:
                pass

            queue: asyncio.Queue[Tuple[int, str]] = asyncio.Queue()
            for i in range(self.config.total_requests):
                p = self.prompts[i % len(self.prompts)]
                queue.put_nowait((i, p))

            results: List[RequestMetric] = []
            results_lock = asyncio.Lock()

            async def worker(worker_id: int):
                while not queue.empty():
                    try:
                        req_idx, prompt = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    req_id = f"w{worker_id}-r{req_idx}"
                    metric = await self._send_single_request(client, req_id, prompt)
                    async with results_lock:
                        results.append(metric)
                    queue.task_done()

            t0 = time.time()
            workers = [asyncio.create_task(worker(w)) for w in range(self.config.concurrency)]
            await asyncio.gather(*workers)
            t_total = max(0.001, time.time() - t0)

            # Query peak GPU status post-run
            try:
                hw_resp = await client.get(f"{base_url}/gpu/status", timeout=2.0)
                if hw_resp.status_code == 200:
                    hw_data = hw_resp.json()
                    peak_gpu_util = max(peak_gpu_util, hw_data.get("latest", {}).get("gpu_utilization_pct", 0.0))
                    peak_vram = max(peak_vram, hw_data.get("peak_gpu_memory_mb", 0.0))
            except Exception:
                pass

        # Aggregate metrics
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        tot_gen_tokens = sum(r.completion_tokens for r in successful)
        tot_prompt_tokens = sum(r.prompt_tokens for r in successful)

        ttft_vals = [r.ttft_ms for r in successful if r.ttft_ms > 0]
        tpot_vals = [r.tpot_ms for r in successful if r.tpot_ms > 0]
        e2e_vals = [r.e2e_ms for r in successful]

        def avg(l: List[float]) -> float:
            return sum(l) / len(l) if l else 0.0

        return LoadTestResult(
            config=self.config,
            total_duration_s=t_total,
            total_requests=len(results),
            successful_requests=len(successful),
            failed_requests=len(failed),
            failure_rate_pct=(len(failed) / len(results) * 100.0) if results else 0.0,
            system_throughput_rps=len(successful) / t_total,
            generation_throughput_tps=tot_gen_tokens / t_total,
            total_tokens_throughput_tps=(tot_gen_tokens + tot_prompt_tokens) / t_total,
            ttft_avg_ms=avg(ttft_vals),
            ttft_p50_ms=calc_percentile(ttft_vals, 50),
            ttft_p90_ms=calc_percentile(ttft_vals, 90),
            ttft_p95_ms=calc_percentile(ttft_vals, 95),
            ttft_p99_ms=calc_percentile(ttft_vals, 99),
            tpot_avg_ms=avg(tpot_vals),
            tpot_p50_ms=calc_percentile(tpot_vals, 50),
            tpot_p90_ms=calc_percentile(tpot_vals, 90),
            tpot_p95_ms=calc_percentile(tpot_vals, 95),
            tpot_p99_ms=calc_percentile(tpot_vals, 99),
            e2e_avg_ms=avg(e2e_vals),
            e2e_p50_ms=calc_percentile(e2e_vals, 50),
            e2e_p90_ms=calc_percentile(e2e_vals, 90),
            e2e_p95_ms=calc_percentile(e2e_vals, 95),
            e2e_p99_ms=calc_percentile(e2e_vals, 99),
            peak_gpu_util_pct=peak_gpu_util,
            peak_gpu_mem_mb=peak_vram,
        )
