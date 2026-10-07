"""FastAPI API routes for LLM inference serving.
Includes OpenAI-compatible endpoints (/v1/chat/completions, /v1/completions, /v1/models),
native /generate endpoint, Prometheus /metrics, and real-time hardware status endpoints.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import List, Optional, Union, Dict, Any

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from inference.engine.engine import LLMInferenceEngine

router = APIRouter()


# ---------------- Schema Definitions ----------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "default"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 128
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None


class CompletionRequest(BaseModel):
    model: Optional[str] = "default"
    prompt: str
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 128
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None


class NativeGenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stop_strings: Optional[List[str]] = Field(default_factory=list)
    stream: Optional[bool] = False


# ---------------- Helper Functions ----------------

def format_messages_to_prompt(messages: List[ChatMessage]) -> str:
    """Format chat messages list into a prompt."""
    lines = []
    for msg in messages:
        if msg.role == "system":
            lines.append(f"System: {msg.content}")
        elif msg.role == "user":
            lines.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            lines.append(f"Assistant: {msg.content}")
        else:
            lines.append(f"{msg.role.capitalize()}: {msg.content}")
    lines.append("Assistant:")
    return "\n".join(lines)


def get_engine(request: Request) -> LLMInferenceEngine:
    engine: LLMInferenceEngine = getattr(request.app.state, "engine", None)
    if not engine:
        raise HTTPException(status_code=503, detail="Inference engine is not initialized")
    return engine


# ---------------- Endpoints ----------------

@router.get("/health")
async def health_check(request: Request) -> Dict[str, Any]:
    engine = get_engine(request)
    hw = engine.hw_monitor.get_latest()
    return {
        "status": "healthy",
        "model_id": engine.config.model.model_id,
        "device": str(engine.device),
        "scheduler_policy": engine.config.scheduler.policy,
        "gpu_utilization_pct": hw.gpu_utilization_pct,
        "gpu_memory_used_mb": hw.gpu_memory_used_mb,
    }


@router.get("/v1/models")
async def list_models(request: Request) -> Dict[str, Any]:
    engine = get_engine(request)
    return {
        "object": "list",
        "data": [
            {
                "id": engine.config.model.model_id,
                "object": "model",
                "created": int(engine.metrics_tracker.start_time),
                "owned_by": "custom-inference-system",
            }
        ],
    }


@router.post("/generate")
async def generate_native(req: NativeGenerateRequest, request: Request):
    """Native high-performance endpoint with optional SSE streaming."""
    engine = get_engine(request)

    if req.stream:
        async def stream_generator():
            async for token in engine.generate_stream(
                prompt=req.prompt,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                stop_strings=req.stop_strings,
            ):
                payload = json.dumps({"token": token})
                yield f"data: {payload}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    response = await engine.generate(
        prompt=req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        stop_strings=req.stop_strings,
    )
    return response.to_dict()


@router.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    """OpenAI-compatible chat completion endpoint supporting streaming SSE."""
    engine = get_engine(request)
    prompt = format_messages_to_prompt(req.messages)

    stop_list: List[str] = []
    if isinstance(req.stop, str):
        stop_list = [req.stop]
    elif isinstance(req.stop, list):
        stop_list = req.stop

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())

    if req.stream:
        async def sse_event_stream():
            async for token in engine.generate_stream(
                prompt=prompt,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                stop_strings=stop_list,
            ):
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": engine.config.model.model_id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": token},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            # Final finish chunk
            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": engine.config.model.model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse_event_stream(), media_type="text/event-stream")

    resp = await engine.generate(
        prompt=prompt,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        stop_strings=stop_list,
    )

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_ts,
        "model": engine.config.model.model_id,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": resp.generated_text,
                },
                "finish_reason": resp.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": resp.prompt_tokens_count,
            "completion_tokens": resp.generated_tokens_count,
            "total_tokens": resp.prompt_tokens_count + resp.generated_tokens_count,
        },
        "metrics": {
            "ttft_ms": round(resp.ttft_ms, 2),
            "tpot_ms": round(resp.tpot_ms, 2),
            "e2e_latency_ms": round(resp.e2e_latency_ms, 2),
        },
    }


@router.post("/v1/completions")
async def raw_completions(req: CompletionRequest, request: Request):
    """OpenAI-compatible text completion endpoint."""
    engine = get_engine(request)

    stop_list = [req.stop] if isinstance(req.stop, str) else (req.stop or [])
    completion_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())

    resp = await engine.generate(
        prompt=req.prompt,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        stop_strings=stop_list,
    )

    return {
        "id": completion_id,
        "object": "text_completion",
        "created": created_ts,
        "model": engine.config.model.model_id,
        "choices": [
            {
                "text": resp.generated_text,
                "index": 0,
                "finish_reason": resp.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": resp.prompt_tokens_count,
            "completion_tokens": resp.generated_tokens_count,
            "total_tokens": resp.prompt_tokens_count + resp.generated_tokens_count,
        },
        "metrics": {
            "ttft_ms": round(resp.ttft_ms, 2),
            "tpot_ms": round(resp.tpot_ms, 2),
            "e2e_latency_ms": round(resp.e2e_latency_ms, 2),
        },
    }


@router.get("/metrics")
async def prometheus_metrics(request: Request):
    """Expose Prometheus exposition format metrics."""
    engine = get_engine(request)
    body = engine.metrics_tracker.generate_prometheus_metrics()
    return Response(content=body, media_type="text/plain; version=0.0.4")


@router.get("/metrics/json")
async def json_metrics(request: Request) -> Dict[str, Any]:
    """Expose structured latency percentiles and throughput counters."""
    engine = get_engine(request)
    return engine.metrics_tracker.get_summary()


@router.get("/gpu/status")
async def gpu_status(request: Request) -> Dict[str, Any]:
    """Get live hardware and GPU utilization and memory."""
    engine = get_engine(request)
    return engine.hw_monitor.get_summary()


@router.get("/engine/status")
async def engine_status(request: Request) -> Dict[str, Any]:
    """Get comprehensive engine, scheduler, and hardware status."""
    engine = get_engine(request)
    return engine.get_status()
