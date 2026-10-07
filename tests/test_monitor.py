"""Unit tests for hardware monitor and metrics tracker."""
import time
import pytest

from inference.monitor.gpu import HardwareMonitor
from inference.monitor.metrics import MetricsTracker, RequestRecord, calculate_percentile


def test_percentile_calculation():
    vals = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert calculate_percentile(vals, 50) == 30.0
    assert calculate_percentile(vals, 0) == 10.0
    assert calculate_percentile(vals, 100) == 50.0


def test_hardware_monitor_sampling():
    mon = HardwareMonitor(sample_interval=0.1)
    snap = mon.sample()
    assert snap.device_type in ("cuda", "mps", "cpu")
    assert snap.ram_total_mb > 0
    assert snap.ram_pct >= 0

    summary = mon.get_summary()
    assert "latest" in summary
    assert "peak_gpu_memory_mb" in summary


def test_metrics_tracker_recording():
    tracker = MetricsTracker(window_size=50)

    t0 = time.time()
    tracker.record_request_arrival()
    tracker.record_schedule_start()

    rec = RequestRecord(
        request_id="req-1",
        prompt_tokens=10,
        generated_tokens=20,
        arrival_time=t0,
        start_schedule_time=t0 + 0.01,
        first_token_time=t0 + 0.05,
        finish_time=t0 + 0.25,
        inter_token_latencies=[0.01] * 19,
        success=True,
    )
    tracker.record_completion(rec)

    summary = tracker.get_summary()
    assert summary["counters"]["completed_requests"] == 1
    assert summary["counters"]["total_prompt_tokens"] == 10
    assert summary["counters"]["total_generated_tokens"] == 20
    assert summary["latency_ms"]["ttft"]["avg"] > 0
    assert summary["latency_ms"]["e2e"]["avg"] > 0

    prom_text = tracker.generate_prometheus_metrics()
    assert "llm_requests_total" in prom_text
    assert "llm_ttft_ms" in prom_text
