from __future__ import annotations

import os
import sys

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import argparse
import asyncio

from inference.config import EngineConfig, ModelConfig, SchedulerConfig


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn
    from inference.server.app import create_app
    from inference.engine.engine import LLMInferenceEngine

    cfg = EngineConfig(
        model=ModelConfig(
            model_id=args.model,
            device=args.device,
            dtype=args.dtype,
            quantization=args.quantization,
            int4_group_size=args.group_size,
        ),
        scheduler=SchedulerConfig(
            policy=args.batching,
            max_batch_size=args.max_batch,
            batch_wait_ms=args.batch_wait_ms,
        ),
    )

    print(f"\n{'='*60}")
    print(f"  Starting LLM Inference Server")
    print(f"  Model:        {args.model}")
    print(f"  Batching:     {args.batching} (max_batch_size={args.max_batch})")
    print(f"  Quantization: {args.quantization}")
    print(f"  Device:       {args.device}")
    print(f"  Host/Port:    http://{args.host}:{args.port}")
    print(f"  Dashboard:    http://{args.host}:{args.port}/dashboard")
    print(f"{'='*60}\n")

    engine = LLMInferenceEngine(cfg)
    app = create_app(engine=engine)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def cmd_benchmark(args: argparse.Namespace) -> None:
    from benchmarks.run_benchmarks import main as bench_main
    print("\nRunning comprehensive benchmark suite...")
    asyncio.run(bench_main())


def cmd_load_test(args: argparse.Namespace) -> None:
    from load_test.load_tester import AsyncLoadTester, LoadTestConfig

    cfg = LoadTestConfig(
        target_url=args.url,
        concurrency=args.concurrency,
        total_requests=args.requests,
        max_new_tokens=args.max_tokens,
        stream=args.stream,
    )
    print(f"\nStarting load test: Concurrency={args.concurrency}, Requests={args.requests}, Target={args.url}")
    tester = AsyncLoadTester(cfg)
    result = asyncio.run(tester.run())
    result.print_summary()


def cmd_quantize(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM
    from inference.quantization.quantizer import quantize_model, profile_model_memory

    print(f"\nLoading {args.model} for quantization profiling...")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    prof_orig = profile_model_memory(model)
    print(f"Original Model Memory: {prof_orig.total_memory_mb:.2f} MB ({prof_orig.total_params:,} parameters)")

    print(f"Applying quantization mode: {args.mode} (group_size={args.group_size})...")
    q_model, stats = quantize_model(model, mode=args.mode, group_size=args.group_size)

    print(f"\n{'='*50}")
    print(f"  Quantization Results ({args.mode.upper()})")
    print(f"{'='*50}")
    print(f"  Layers Replaced:    {stats['layers_replaced']}")
    print(f"  Original Size:      {stats['original_memory_mb']:.2f} MB")
    print(f"  Quantized Size:     {stats['quantized_memory_mb']:.2f} MB")
    print(f"  Compression Ratio:  {stats['compression_ratio']:.2f}x")
    print(f"  Memory Savings:     {stats['savings_pct']:.1f}%")
    print(f"{'='*50}\n")


def cmd_generate(args: argparse.Namespace) -> None:
    from inference.engine.engine import LLMInferenceEngine

    cfg = EngineConfig(
        model=ModelConfig(
            model_id=args.model,
            device=args.device,
            dtype=args.dtype,
            quantization=args.quantization,
        ),
        scheduler=SchedulerConfig(policy="continuous", max_batch_size=4),
    )
    engine = LLMInferenceEngine(cfg)
    engine.start()

    async def run_gen():
        print(f"\nPrompt: {args.prompt}\n---")
        if args.stream:
            async for chunk in engine.generate_stream(args.prompt, max_new_tokens=args.max_tokens):
                sys.stdout.write(chunk)
                sys.stdout.flush()
            print()
        else:
            resp = await engine.generate(args.prompt, max_new_tokens=args.max_tokens)
            print(resp.generated_text)
            print(f"\nMetrics: TTFT={resp.ttft_ms:.1f}ms, TPOT={resp.tpot_ms:.1f}ms, E2E={resp.e2e_latency_ms:.1f}ms")
        engine.stop()

    asyncio.run(run_gen())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="High-Performance LLM Inference System CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Subcommand: serve
    p_serve = subparsers.add_parser("serve", help="Start the FastAPI inference server")
    p_serve.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B", help="HuggingFace model ID")
    p_serve.add_argument("--device", type=str, default="auto", help="auto | cuda | mps | cpu")
    p_serve.add_argument("--dtype", type=str, default="float16", help="float32 | float16 | bfloat16")
    p_serve.add_argument("--batching", type=str, default="continuous", choices=["continuous", "static", "none"], help="Batching scheduler policy")
    p_serve.add_argument("--max-batch", type=int, default=8, help="Maximum batch size")
    p_serve.add_argument("--batch-wait-ms", type=float, default=10.0, help="Static batch wait window in ms")
    p_serve.add_argument("--quantization", type=str, default="none", choices=["none", "fp16", "int8", "int4"], help="Quantization strategy")
    p_serve.add_argument("--group-size", type=int, default=64, help="INT4 group size")
    p_serve.add_argument("--host", type=str, default="127.0.0.1", help="Server host IP")
    p_serve.add_argument("--port", type=int, default=8000, help="Server port")
    p_serve.set_defaults(func=cmd_serve)

    # Subcommand: benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run batching & quantization benchmarks")
    p_bench.set_defaults(func=cmd_benchmark)

    # Subcommand: load-test
    p_lt = subparsers.add_parser("load-test", help="Run concurrent HTTP load tester")
    p_lt.add_argument("--url", type=str, default="http://127.0.0.1:8000/generate", help="Target URL")
    p_lt.add_argument("--concurrency", type=int, default=4, help="Concurrent clients")
    p_lt.add_argument("--requests", type=int, default=16, help="Total requests to dispatch")
    p_lt.add_argument("--max-tokens", type=int, default=24, help="Generated tokens per request")
    p_lt.add_argument("--stream", action="store_true", help="Use SSE streaming mode")
    p_lt.set_defaults(func=cmd_load_test)

    # Subcommand: quantize
    p_quant = subparsers.add_parser("quantize", help="Profile and evaluate model quantization")
    p_quant.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B", help="Model ID")
    p_quant.add_argument("--mode", type=str, default="int4", choices=["fp16", "int8", "int4"], help="Quantization mode")
    p_quant.add_argument("--group-size", type=int, default=64, help="INT4 group size")
    p_quant.set_defaults(func=cmd_quantize)

    # Subcommand: generate
    p_gen = subparsers.add_parser("generate", help="Run one-off inference in terminal")
    p_gen.add_argument("--prompt", type=str, required=True, help="Input prompt")
    p_gen.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B", help="Model ID")
    p_gen.add_argument("--device", type=str, default="auto", help="auto | mps | cuda | cpu")
    p_gen.add_argument("--dtype", type=str, default="float16", help="float32 | float16")
    p_gen.add_argument("--quantization", type=str, default="none", help="none | int8 | int4")
    p_gen.add_argument("--max-tokens", type=int, default=32, help="Max output tokens")
    p_gen.add_argument("--stream", action="store_true", help="Stream tokens to stdout")
    p_gen.set_defaults(func=cmd_generate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
