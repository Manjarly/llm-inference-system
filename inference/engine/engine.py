"""High-performance LLM Inference Engine.
Features:
- Continuous batching (iteration-level scheduling)
- Static batching & sequential baselines
- DynamicCache KV-cache retention
- Real-time token streaming with AsyncIterator
- Fine-grained TTFT, TPOT, and E2E latency instrumentation
- Integrated hardware monitoring and Prometheus metrics
"""
from __future__ import annotations

import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import asyncio
import logging
import threading
import time
from typing import AsyncIterator, List, Optional, Dict, Any, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from inference.config import EngineConfig
from inference.engine.request import InferenceRequest, InferenceResponse, SequenceState
from inference.engine.scheduler import Scheduler
from inference.monitor.gpu import HardwareMonitor
from inference.monitor.metrics import MetricsTracker, RequestRecord
from inference.quantization.quantizer import quantize_model, profile_model_memory

logger = logging.getLogger("inference.engine")


def sample_token(logits: torch.Tensor, temperature: float = 0.7, top_p: float = 0.9) -> int:
    """Sample a token id from raw logits using temperature and nucleus (top-p) sampling."""
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1).item()

    # Temperature scaling
    scaled = logits / max(1e-5, temperature)
    probs = F.softmax(scaled, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        sorted_probs[sorted_indices_to_remove] = 0.0
        sum_p = sorted_probs.sum(dim=-1, keepdim=True)
        if sum_p.item() > 0:
            probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)
            probs = probs / probs.sum(dim=-1, keepdim=True)

    next_tok = torch.multinomial(probs, num_samples=1)
    return next_tok.item()


class LLMInferenceEngine:
    """Core Inference Engine coordinating model execution, batch scheduling, and metrics."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.device = self._resolve_device(config.model.device)
        self.dtype = self._resolve_dtype(config.model.dtype)

        logger.info(
            "Initializing LLM Engine: model=%s, device=%s, dtype=%s, policy=%s, quant=%s",
            config.model.model_id,
            self.device,
            self.dtype,
            config.scheduler.policy,
            config.model.quantization,
        )

        # Hardware Monitor
        self.hw_monitor = HardwareMonitor(
            device_override=str(self.device),
            sample_interval=config.gpu_monitor_interval_s,
        )
        self.hw_monitor.start()

        # Metrics Tracker
        self.metrics_tracker = MetricsTracker()

        # Scheduler
        self.scheduler = Scheduler(config.scheduler)

        # Load Tokenizer & Model
        self._load_model_and_tokenizer()

        # Background worker loop
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _resolve_device(self, dev: str) -> torch.device:
        if dev == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(dev)

    def _resolve_dtype(self, dt: str) -> torch.dtype:
        if dt == "auto":
            if self.device.type in ("cuda", "mps"):
                return torch.float16
            return torch.float32
        mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return mapping.get(dt, torch.float16)

    def _load_model_and_tokenizer(self) -> None:
        model_id = self.config.model.model_id
        logger.info("Loading tokenizer for %s...", model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=self.config.model.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        logger.info("Loading weights for %s...", model_id)
        raw_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            trust_remote_code=self.config.model.trust_remote_code,
        )

        # Quantize if configured
        quant_mode = self.config.model.quantization.lower()
        if quant_mode not in ("none", "fp16", "bf16", "fp32"):
            self.model, self.quant_stats = quantize_model(
                raw_model,
                mode=quant_mode,
                group_size=self.config.model.int4_group_size,
                quantize_lm_head=self.config.model.quantize_lm_head,
                device=self.device,
            )
        else:
            self.quant_stats = {"mode": quant_mode, "savings_pct": 0.0}
            self.model = raw_model.to(self.device)

        self.model.eval()
        self.eos_token_id = self.tokenizer.eos_token_id
        self.profile = profile_model_memory(self.model)
        logger.info(
            "Model ready! Params: %d, Memory: %.2f MB on %s",
            self.profile.total_params,
            self.profile.total_memory_mb,
            self.device,
        )

    def start(self) -> None:
        """Start the background execution step loop."""
        if self._running:
            return
        self._running = True
        self._loop = asyncio.get_event_loop()
        self._worker_task = asyncio.create_task(self._step_loop())
        logger.info("LLM Engine worker task started successfully.")

    def stop(self) -> None:
        """Stop engine processing and hardware monitor."""
        self._running = False
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
        self.hw_monitor.stop()
        logger.info("LLM Engine stopped.")

    async def submit_request(self, request: InferenceRequest) -> None:
        """Tokenize request and enqueue for execution."""
        # Encode prompt
        token_ids = self.tokenizer.encode(request.prompt, add_special_tokens=True)
        request.prompt_token_ids = token_ids
        self.metrics_tracker.record_request_arrival()

        enqueued = self.scheduler.enqueue(request)
        if not enqueued:
            if request.future and not request.future.done():
                request.future.set_exception(RuntimeError("Scheduler queue full"))
            if request.stream_queue:
                await request.stream_queue.put(None)

    async def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_strings: Optional[List[str]] = None,
    ) -> InferenceResponse:
        """Standard request-response generation."""
        max_tokens = max_new_tokens or self.config.default_max_new_tokens
        loop = asyncio.get_running_loop()
        future: asyncio.Future[InferenceResponse] = loop.create_future()

        req = InferenceRequest(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_strings=stop_strings or [],
            future=future,
        )
        await self.submit_request(req)
        return await future

    async def generate_stream(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_strings: Optional[List[str]] = None,
    ) -> AsyncIterator[str]:
        """Streaming token generation yielding text chunks as they are generated."""
        max_tokens = max_new_tokens or self.config.default_max_new_tokens
        stream_q: asyncio.Queue[Optional[str]] = asyncio.Queue()

        req = InferenceRequest(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_strings=stop_strings or [],
            stream_queue=stream_q,
        )
        await self.submit_request(req)

        while True:
            chunk = await stream_q.get()
            if chunk is None:
                break
            yield chunk

    def _sync_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()

    async def _step_loop(self) -> None:
        """Main engine iteration loop driving prefill and continuous/static decode."""
        logger.info("Engine step loop entered.")
        while self._running:
            try:
                to_prefill, to_decode = self.scheduler.schedule_next_step()

                if not to_prefill and not to_decode:
                    await asyncio.sleep(0.002)
                    continue

                # Run step synchronously in executor to avoid blocking event loop
                await asyncio.to_thread(self._execute_step, to_prefill, to_decode)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Error in engine step loop: %s", e)
                await asyncio.sleep(0.01)

    def _execute_step(
        self, to_prefill: List[InferenceRequest], to_decode: List[SequenceState]
    ) -> None:
        """Execute one engine iteration: prefill new requests, decode active sequences."""
        t_step_start = time.time()
        finished_sequences: List[SequenceState] = []

        # ---------------- PREFILL PHASE ----------------
        for req in to_prefill:
            self.metrics_tracker.record_schedule_start()
            t_sched = time.time()

            input_tensor = torch.tensor([req.prompt_token_ids], dtype=torch.long, device=self.device)
            cache = DynamicCache()

            with torch.no_grad():
                out = self.model(input_tensor, past_key_values=cache, use_cache=True)
                logits = out.logits[:, -1, :][0]
                next_tok_id = sample_token(logits, temperature=req.temperature, top_p=req.top_p)

            self._sync_device()
            t_now = time.time()
            tok_text = self.tokenizer.decode([next_tok_id], skip_special_tokens=False)

            seq = SequenceState(
                request=req,
                cache=cache,
                start_schedule_time=t_sched,
            )
            seq.record_first_token(next_tok_id, tok_text, t_now)

            # Stream first token chunk
            if req.stream_queue:
                self._dispatch_stream_chunk(req.stream_queue, tok_text)

            # Check if finished at first token
            if next_tok_id == self.eos_token_id:
                seq.finish_reason = "stop"
                finished_sequences.append(seq)
            elif seq.generated_tokens_count >= req.max_new_tokens:
                seq.finish_reason = "length"
                finished_sequences.append(seq)
            else:
                self.scheduler.running.append(seq)

        # ---------------- DECODE PHASE ----------------
        for seq in to_decode:
            if seq.is_finished:
                continue

            input_tok = torch.tensor([[seq.current_token_id]], dtype=torch.long, device=self.device)

            with torch.no_grad():
                out = self.model(input_tok, past_key_values=seq.cache, use_cache=True)
                logits = out.logits[:, -1, :][0]
                next_tok_id = sample_token(logits, temperature=seq.request.temperature, top_p=seq.request.top_p)

            self._sync_device()
            t_now = time.time()
            tok_text = self.tokenizer.decode([next_tok_id], skip_special_tokens=False)
            seq.record_next_token(next_tok_id, tok_text, t_now)

            # Stream token chunk
            if seq.request.stream_queue:
                self._dispatch_stream_chunk(seq.request.stream_queue, tok_text)

            # Check stopping conditions
            is_stop_token = next_tok_id == self.eos_token_id
            hit_max_tokens = seq.generated_tokens_count >= seq.request.max_new_tokens
            hit_stop_string = False

            if seq.request.stop_strings:
                accum_text = "".join(seq.generated_text_chunks)
                for s in seq.request.stop_strings:
                    if s in accum_text:
                        hit_stop_string = True
                        break

            if is_stop_token or hit_stop_string:
                seq.finish_reason = "stop"
                finished_sequences.append(seq)
            elif hit_max_tokens:
                seq.finish_reason = "length"
                finished_sequences.append(seq)

        # ---------------- FINALIZE COMPLETED SEQUENCES ----------------
        t_finish = time.time()
        for seq in finished_sequences:
            response = seq.finalize(t_finish)

            # Complete response future
            if seq.request.future and not seq.request.future.done():
                self._dispatch_future_result(seq.request.future, response)

            # Close stream queue with sentinel
            if seq.request.stream_queue:
                self._dispatch_stream_chunk(seq.request.stream_queue, None)

            # Log to metrics tracker
            rec = RequestRecord(
                request_id=seq.request.request_id,
                prompt_tokens=len(seq.request.prompt_token_ids),
                generated_tokens=seq.generated_tokens_count,
                arrival_time=seq.request.arrival_time,
                start_schedule_time=seq.start_schedule_time,
                first_token_time=seq.first_token_time,
                finish_time=t_finish,
                inter_token_latencies=seq.inter_token_latencies,
                success=True,
            )
            self.metrics_tracker.record_completion(rec)

        if finished_sequences:
            self.scheduler.remove_finished(finished_sequences)

    def _dispatch_stream_chunk(self, queue: asyncio.Queue, chunk: Optional[str]) -> None:
        """Safely push token chunk to asyncio stream queue from worker thread."""
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(queue.put_nowait, chunk)

    def _dispatch_future_result(self, future: asyncio.Future, result: InferenceResponse) -> None:
        """Safely resolve future from worker thread."""
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(future.set_result, result)

    def get_status(self) -> Dict[str, Any]:
        """Aggregate engine status, hardware metrics, and performance counters."""
        hw = self.hw_monitor.get_summary()
        metrics = self.metrics_tracker.get_summary()

        return {
            "model": {
                "model_id": self.config.model.model_id,
                "device": str(self.device),
                "dtype": str(self.dtype),
                "quantization": self.config.model.quantization,
                "quant_stats": self.quant_stats,
                "memory_profile": self.profile.to_dict(),
            },
            "scheduler": {
                "policy": self.config.scheduler.policy,
                "max_batch_size": self.config.scheduler.max_batch_size,
                "running_sequences": self.scheduler.running_size,
                "queued_requests": self.scheduler.queue_size,
            },
            "hardware": hw,
            "metrics": metrics,
        }
