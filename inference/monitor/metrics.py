"""Inference metrics tracking system.
Tracks TTFT (Time To First Token), TPOT (Time Per Output Token), E2E Latency,
Throughput (tokens/s and req/s), Queue wait times, and percentiles (P50, P90, P95, P99).
Supports Prometheus metrics export and real-time dashboard snapshots.
"""
from __future__ import annotations

import collections
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class RequestRecord:
    request_id: str
    prompt_tokens: int
    generated_tokens: int
    arrival_time: float
    start_schedule_time: float
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None
    inter_token_latencies: List[float] = field(default_factory=list)
    success: bool = True
    error_msg: Optional[str] = None

    @property
    def queue_latency_ms(self) -> float:
        return max(0.0, (self.start_schedule_time - self.arrival_time) * 1000.0)

    @property
    def ttft_ms(self) -> Optional[float]:
        if self.first_token_time is not None:
            return max(0.0, (self.first_token_time - self.arrival_time) * 1000.0)
        return None

    @property
    def prefill_latency_ms(self) -> Optional[float]:
        if self.first_token_time is not None:
            return max(0.0, (self.first_token_time - self.start_schedule_time) * 1000.0)
        return None

    @property
    def e2e_latency_ms(self) -> Optional[float]:
        if self.finish_time is not None:
            return max(0.0, (self.finish_time - self.arrival_time) * 1000.0)
        return None

    @property
    def tpot_ms(self) -> Optional[float]:
        if self.inter_token_latencies:
            return (sum(self.inter_token_latencies) / len(self.inter_token_latencies)) * 1000.0
        elif (
            self.first_token_time is not None
            and self.finish_time is not None
            and self.generated_tokens > 1
        ):
            decode_time = (self.finish_time - self.first_token_time) * 1000.0
            return decode_time / (self.generated_tokens - 1)
        return None


def calculate_percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    k = (len(sorted_v) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_v[int(k)]
    d0 = sorted_v[int(f)] * (c - k)
    d1 = sorted_v[int(c)] * (k - f)
    return d0 + d1


class MetricsTracker:
    """Thread-safe latency and throughput tracker for the LLM inference engine."""

    def __init__(self, window_size: int = 1000) -> None:
        self.window_size = window_size
        self._lock = threading.Lock()
        self.start_time = time.time()

        # Cumulative counters
        self.total_requests = 0
        self.completed_requests = 0
        self.failed_requests = 0
        self.total_prompt_tokens = 0
        self.total_generated_tokens = 0

        # Concurrency counters
        self.active_requests = 0
        self.queued_requests = 0

        # Rolling history of recent completed requests
        self._completed_records: collections.deque[RequestRecord] = collections.deque(
            maxlen=window_size
        )

    def record_request_arrival(self) -> None:
        with self._lock:
            self.total_requests += 1
            self.queued_requests += 1

    def record_schedule_start(self) -> None:
        with self._lock:
            if self.queued_requests > 0:
                self.queued_requests -= 1
            self.active_requests += 1

    def record_completion(self, record: RequestRecord) -> None:
        with self._lock:
            if self.active_requests > 0:
                self.active_requests -= 1

            if record.success:
                self.completed_requests += 1
                self.total_prompt_tokens += record.prompt_tokens
                self.total_generated_tokens += record.generated_tokens
                self._completed_records.append(record)
            else:
                self.failed_requests += 1

    def get_summary(self) -> Dict[str, Any]:
        """Compute latency percentiles, throughput, and engine load."""
        with self._lock:
            records = list(self._completed_records)
            tot_req = self.total_requests
            comp_req = self.completed_requests
            fail_req = self.failed_requests
            tot_prompt_tok = self.total_prompt_tokens
            tot_gen_tok = self.total_generated_tokens
            active_req = self.active_requests
            queue_req = self.queued_requests
            start_t = self.start_time

        now = time.time()
        elapsed_s = max(0.001, now - start_t)

        ttft_list = [r.ttft_ms for r in records if r.ttft_ms is not None]
        tpot_list = [r.tpot_ms for r in records if r.tpot_ms is not None]
        e2e_list = [r.e2e_latency_ms for r in records if r.e2e_latency_ms is not None]
        queue_list = [r.queue_latency_ms for r in records]

        # Throughput
        gen_tokens_sec = tot_gen_tok / elapsed_s
        total_tokens_sec = (tot_prompt_tok + tot_gen_tok) / elapsed_s
        req_sec = comp_req / elapsed_s

        # Rolling window stats for instant throughput (last 30s)
        recent_window_s = 10.0
        recent_records = [r for r in records if r.finish_time and (now - r.finish_time) <= recent_window_s]
        if recent_records:
            recent_gen_tok = sum(r.generated_tokens for r in recent_records)
            recent_gen_tps = recent_gen_tok / recent_window_s
        else:
            recent_gen_tps = gen_tokens_sec

        def stats(vals: List[float]) -> Dict[str, float]:
            if not vals:
                return {"avg": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
            return {
                "avg": round(sum(vals) / len(vals), 2),
                "p50": round(calculate_percentile(vals, 50), 2),
                "p90": round(calculate_percentile(vals, 90), 2),
                "p95": round(calculate_percentile(vals, 95), 2),
                "p99": round(calculate_percentile(vals, 99), 2),
                "min": round(min(vals), 2),
                "max": round(max(vals), 2),
            }

        return {
            "uptime_seconds": round(elapsed_s, 2),
            "counters": {
                "total_requests": tot_req,
                "completed_requests": comp_req,
                "failed_requests": fail_req,
                "active_requests": active_req,
                "queued_requests": queue_req,
                "total_prompt_tokens": tot_prompt_tok,
                "total_generated_tokens": tot_gen_tok,
            },
            "throughput": {
                "gen_tokens_per_sec": round(gen_tokens_sec, 2),
                "total_tokens_per_sec": round(total_tokens_sec, 2),
                "requests_per_sec": round(req_sec, 2),
                "recent_gen_tokens_per_sec": round(recent_gen_tps, 2),
            },
            "latency_ms": {
                "ttft": stats(ttft_list),
                "tpot": stats(tpot_list),
                "e2e": stats(e2e_list),
                "queue": stats(queue_list),
            },
            "samples_in_window": len(records),
        }

    def generate_prometheus_metrics(self) -> str:
        """Render Prometheus metrics in standard exposition format."""
        s = self.get_summary()
        lines = [
            "# HELP llm_requests_total Total number of inference requests",
            "# TYPE llm_requests_total counter",
            f'llm_requests_total{{status="all"}} {s["counters"]["total_requests"]}',
            f'llm_requests_total{{status="completed"}} {s["counters"]["completed_requests"]}',
            f'llm_requests_total{{status="failed"}} {s["counters"]["failed_requests"]}',
            "",
            "# HELP llm_in_flight_requests Current active or queued requests",
            "# TYPE llm_in_flight_requests gauge",
            f'llm_in_flight_requests{{state="active"}} {s["counters"]["active_requests"]}',
            f'llm_in_flight_requests{{state="queued"}} {s["counters"]["queued_requests"]}',
            "",
            "# HELP llm_tokens_total Total tokens processed",
            "# TYPE llm_tokens_total counter",
            f'llm_tokens_total{{type="prompt"}} {s["counters"]["total_prompt_tokens"]}',
            f'llm_tokens_total{{type="generated"}} {s["counters"]["total_generated_tokens"]}',
            "",
            "# HELP llm_generation_throughput_tokens_per_sec Generated tokens per second",
            "# TYPE llm_generation_throughput_tokens_per_sec gauge",
            f'llm_generation_throughput_tokens_per_sec {s["throughput"]["gen_tokens_per_sec"]}',
            "",
            "# HELP llm_ttft_ms Time to first token in milliseconds",
            "# TYPE llm_ttft_ms gauge",
            f'llm_ttft_ms{{quantile="0.5"}} {s["latency_ms"]["ttft"]["p50"]}',
            f'llm_ttft_ms{{quantile="0.9"}} {s["latency_ms"]["ttft"]["p90"]}',
            f'llm_ttft_ms{{quantile="0.99"}} {s["latency_ms"]["ttft"]["p99"]}',
            "",
            "# HELP llm_tpot_ms Time per output token in milliseconds",
            "# TYPE llm_tpot_ms gauge",
            f'llm_tpot_ms{{quantile="0.5"}} {s["latency_ms"]["tpot"]["p50"]}',
            f'llm_tpot_ms{{quantile="0.9"}} {s["latency_ms"]["tpot"]["p90"]}',
            f'llm_tpot_ms{{quantile="0.99"}} {s["latency_ms"]["tpot"]["p99"]}',
            "",
            "# HELP llm_e2e_latency_ms End to end request latency in milliseconds",
            "# TYPE llm_e2e_latency_ms gauge",
            f'llm_e2e_latency_ms{{quantile="0.5"}} {s["latency_ms"]["e2e"]["p50"]}',
            f'llm_e2e_latency_ms{{quantile="0.9"}} {s["latency_ms"]["e2e"]["p90"]}',
            f'llm_e2e_latency_ms{{quantile="0.99"}} {s["latency_ms"]["e2e"]["p99"]}',
        ]
        return "\n".join(lines) + "\n"
