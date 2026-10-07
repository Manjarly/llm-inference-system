# High-Performance LLM Inference System

A production-grade Large Language Model (LLM) serving and inference engine designed from the ground up for high throughput, low latency, and efficient GPU resource utilization.

Built with **Continuous Batching (Iteration-Level Scheduling)**, **Group-Wise Weight Quantization (INT4/INT8/FP16)**, **Universal Hardware Telemetry (NVIDIA CUDA & Apple Silicon MPS)**, **OpenAI-Compatible Streaming APIs**, and a **Built-in Interactive Web Dashboard & Load Testing Suite**.

---

## Key Highlights

- **Model Serving**: Production FastAPI server providing drop-in OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/completions`) and high-speed native endpoints (`/generate`) with real-time Server-Sent Events (SSE) token streaming.
- **Continuous Batching**: Orca/vLLM-style iteration-level scheduler that dynamically admits arriving requests into running decode steps and frees finished sequences immediately, eliminating head-of-line blocking and padding bubble overhead.
- **Quantization Subsystem**:
  - **INT4 Group-Wise (W4A16)**: Packs weights into 4-bit nibbles with group-wise affine scaling (scales + zero points per group of 64), yielding **up to 83% memory savings** and **1.76x - 3.5x compression**.
  - **Dynamic INT8**: Per-channel symmetric quantization with on-the-fly dequantization and dynamic range protection.
  - **FP16 / BF16**: Native half-precision execution.
- **Hardware & GPU Utilization Monitoring**: Universal real-time hardware telemetry engine supporting:
  - Apple Silicon MPS: macOS `IOAccelerator` PerformanceStatistics device utilization % and unified memory tracking.
  - NVIDIA CUDA: `pynvml` / `torch.cuda` SM compute utilization, VRAM allocated, and peak reserved memory.
  - CPU / Host RAM: Core utilization % and memory telemetry via `psutil`.
- **Latency & Throughput Telemetry**:
  - Fine-grained instrumentation measuring **TTFT (Time To First Token)**, **TPOT / ITL (Time Per Output Token / Inter-Token Latency)**, and **E2E Latency**.
  - Percentile analysis: **P50, P90, P95, P99** distributions.
  - Native **Prometheus `/metrics`** exposition format for Grafana / Datadog scraping.
- **Load Testing & Benchmarking Suite**:
  - Asynchronous multi-client load tester (`load_test/load_tester.py`) simulating concurrent users and variable prompt distributions.
  - End-to-end comparative benchmark runner (`benchmarks/run_benchmarks.py`) generating visual matplotlib plots and comprehensive markdown reports.
- **Interactive Web Dashboard**: Modern dark-mode web console (`/dashboard`) featuring real-time GPU/VRAM gauges, live streaming token playground, and an in-browser load test controller.

---

## Architecture Overview

```
                        Incoming Requests (HTTP / OpenAI API / SSE)
                                          │
                                          ▼
                         ┌─────────────────────────────────┐
                         │      FastAPI Serving Engine     │
                         │  (/v1/chat, /generate, /metrics)│
                         └────────────────┬────────────────┘
                                          │
                                          ▼
                         ┌─────────────────────────────────┐
                         │   Iteration-Level Scheduler     │
                         │  (Continuous / Static / None)   │
                         └───────┬─────────────────┬───────┘
                                 │                 │
                   Prefill Phase │                 │ Decode Phase
                                 ▼                 ▼
                         ┌─────────────────────────────────┐
                         │   Dynamic LLM Inference Loop    │
                         │  (DynamicCache KV Retention)    │
                         │  (Quantized Linear Layers: W4)  │
                         └──────────────┬──────────────────┘
                                        │
                         ┌──────────────┴──────────────────┐
                         │                                 │
                         ▼                                 ▼
              ┌─────────────────────┐           ┌─────────────────────┐
              │ Hardware Monitor    │           │ Metrics Tracker     │
              │ (Apple MPS / CUDA)  │           │ (TTFT, TPOT, P95)   │
              └─────────────────────┘           └─────────────────────┘
```

---

## Directory Structure

```
llm_inference_system/
├── README.md                      # Complete system documentation
├── requirements.txt               # Dependencies
├── pytest.ini                     # Pytest configuration
├── cli.py                         # Unified command line tool
├── inference/
│   ├── __init__.py
│   ├── config.py                  # Model, Scheduler & Engine configurations
│   ├── monitor/
│   │   ├── __init__.py
│   │   ├── gpu.py                 # Universal CUDA / MPS / CPU hardware monitor
│   │   └── metrics.py             # TTFT, TPOT, throughput & Prometheus exporter
│   ├── quantization/
│   │   ├── __init__.py
│   │   ├── int8.py                # Per-channel dynamic INT8 linear layer
│   │   ├── int4.py                # Group-wise 4-bit packed linear layer (W4A16)
│   │   └── quantizer.py           # Model quantizer & memory profiler
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── request.py             # InferenceRequest, SequenceState, Response
│   │   ├── scheduler.py           # Continuous, static & sequential schedulers
│   │   └── engine.py              # LLM engine with async iterative decode loop
│   └── server/
│       ├── __init__.py
│       ├── app.py                 # FastAPI application factory
│       ├── api.py                 # Route definitions & OpenAI schemas
│       └── static/
│           ├── index.html         # Real-time web dashboard
│           ├── style.css          # Glassmorphic dark theme
│           └── app.js             # Telemetry poller, streaming UI & load tester
├── load_test/
│   ├── __init__.py
│   ├── load_tester.py             # Asynchronous load testing engine
│   └── prompts.json               # Benchmark prompt dataset
├── benchmarks/
│   ├── __init__.py
│   ├── run_benchmarks.py          # Benchmark pipeline
│   ├── plot_results.py            # Matplotlib visual plot generator
│   ├── BENCHMARK_REPORT.md        # Generated comparison report
│   ├── results.json               # Raw benchmark JSON data
│   ├── batching_benchmark.png     # Batching comparison chart
│   └── quantization_benchmark.png # Quantization memory chart
└── tests/
    ├── test_quantization.py       # INT8, INT4 & profiler tests
    ├── test_engine.py             # Scheduler & sampling tests
    ├── test_monitor.py            # GPU monitor & metrics tracker tests
    └── test_server.py             # FastAPI endpoint integration tests
```

---

## Installation

```bash
# Clone and enter directory
cd /Users/manjarly/Amit\ Data/ML/Projects/llm_inference_system

# Install dependencies
pip install -r requirements.txt
```

---

## Quickstart Guide

### 1. Start the Inference Server with Web Dashboard
```bash
python cli.py serve --model Qwen/Qwen2.5-0.5B --batching continuous --max-batch 8 --port 8000
```
Open **[http://localhost:8000/dashboard](http://localhost:8000/dashboard)** in your browser to access the live dashboard with real-time GPU gauges, streaming playground, and load tester!

### 2. Run the Benchmark Suite (Batching & Quantization)
```bash
python cli.py benchmark
```
This runs the automated comparison across:
- **Batching**: Sequential vs Static Batching vs Continuous Batching
- **Quantization**: FP32 vs FP16 vs INT8 vs INT4
Generates `benchmarks/BENCHMARK_REPORT.md` and high-resolution PNG charts.

### 3. Run Asynchronous Load Testing
```bash
# Dispatch 24 requests with 4 concurrent clients to the running server
python cli.py load-test --concurrency 4 --requests 24 --url http://127.0.0.1:8000/generate
```

### 4. Inspect Model Quantization & Memory Savings
```bash
python cli.py quantize --model Qwen/Qwen2.5-0.5B --mode int4 --group-size 64
```

### 5. Generate Text from Terminal
```bash
python cli.py generate --prompt "Explain the benefits of continuous batching in 2 sentences:" --stream
```

### 6. Run Test Suite
```bash
pytest
```

---

## API Reference

### OpenAI Chat Completion (`POST /v1/chat/completions`)
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is continuous batching?"}],
    "max_tokens": 64,
    "temperature": 0.7,
    "stream": true
  }'
```

### Native Inference (`POST /generate`)
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Artificial intelligence is",
    "max_new_tokens": 32,
    "temperature": 0.7,
    "stream": false
  }'
```

### Prometheus Metrics (`GET /metrics`)
```bash
curl http://localhost:8000/metrics
```
Exposes:
- `llm_requests_total`
- `llm_in_flight_requests`
- `llm_tokens_total`
- `llm_generation_throughput_tokens_per_sec`
- `llm_ttft_ms{quantile="0.5|0.9|0.99"}`
- `llm_tpot_ms{quantile="0.5|0.9|0.99"}`
- `llm_e2e_latency_ms{quantile="0.5|0.9|0.99"}`

### Hardware & GPU Telemetry (`GET /gpu/status`)
```bash
curl http://localhost:8000/gpu/status
```
Returns:
```json
{
  "device_type": "mps",
  "device_name": "Apple M3 (MPS)",
  "latest": {
    "gpu_utilization_pct": 42.0,
    "gpu_memory_used_mb": 1134.3,
    "gpu_memory_total_mb": 16384.0,
    "ram_pct": 68.5,
    "cpu_utilization_pct": 12.4
  },
  "peak_gpu_memory_mb": 1284.1
}
```

---

## License
MIT License.
