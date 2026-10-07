"""Inference request, sequence state, and response abstractions."""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict


@dataclass
class InferenceResponse:
    request_id: str
    prompt: str
    generated_text: str
    prompt_tokens_count: int
    generated_tokens_count: int
    finish_reason: str
    ttft_ms: float
    tpot_ms: float
    e2e_latency_ms: float
    queue_latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "prompt": self.prompt,
            "generated_text": self.generated_text,
            "usage": {
                "prompt_tokens": self.prompt_tokens_count,
                "completion_tokens": self.generated_tokens_count,
                "total_tokens": self.prompt_tokens_count + self.generated_tokens_count,
            },
            "finish_reason": self.finish_reason,
            "metrics": {
                "ttft_ms": round(self.ttft_ms, 2),
                "tpot_ms": round(self.tpot_ms, 2),
                "e2e_latency_ms": round(self.e2e_latency_ms, 2),
                "queue_latency_ms": round(self.queue_latency_ms, 2),
            },
        }


@dataclass
class InferenceRequest:
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    stop_strings: List[str] = field(default_factory=list)
    request_id: str = field(default_factory=lambda: f"req-{uuid.uuid4().hex[:8]}")
    arrival_time: float = field(default_factory=time.time)

    # Internal runtime handles
    prompt_token_ids: List[int] = field(default_factory=list)
    stream_queue: Optional[asyncio.Queue[Optional[str]]] = None
    future: Optional[asyncio.Future[InferenceResponse]] = None
    aborted: bool = False


@dataclass
class SequenceState:
    request: InferenceRequest
    cache: Any = None  # DynamicCache or past_key_values tuple
    current_token_id: int = 0
    generated_token_ids: List[int] = field(default_factory=list)
    generated_text_chunks: List[str] = field(default_factory=list)
    is_finished: bool = False
    finish_reason: str = "length"

    # Latency tracking
    start_schedule_time: float = field(default_factory=time.time)
    first_token_time: Optional[float] = None
    last_token_time: Optional[float] = None
    inter_token_latencies: List[float] = field(default_factory=list)

    @property
    def generated_tokens_count(self) -> int:
        return len(self.generated_token_ids)

    def record_first_token(self, token_id: int, token_text: str, t_now: float) -> None:
        self.first_token_time = t_now
        self.last_token_time = t_now
        self.current_token_id = token_id
        self.generated_token_ids.append(token_id)
        self.generated_text_chunks.append(token_text)

    def record_next_token(self, token_id: int, token_text: str, t_now: float) -> None:
        if self.last_token_time is not None:
            self.inter_token_latencies.append(t_now - self.last_token_time)
        self.last_token_time = t_now
        self.current_token_id = token_id
        self.generated_token_ids.append(token_id)
        self.generated_text_chunks.append(token_text)

    def finalize(self, t_now: float) -> InferenceResponse:
        self.is_finished = True
        ttft = (self.first_token_time - self.request.arrival_time) * 1000.0 if self.first_token_time else 0.0
        queue_latency = (self.start_schedule_time - self.request.arrival_time) * 1000.0
        e2e = (t_now - self.request.arrival_time) * 1000.0

        if self.inter_token_latencies:
            tpot = (sum(self.inter_token_latencies) / len(self.inter_token_latencies)) * 1000.0
        elif self.first_token_time and len(self.generated_token_ids) > 1:
            tpot = ((t_now - self.first_token_time) / (len(self.generated_token_ids) - 1)) * 1000.0
        else:
            tpot = 0.0

        full_text = "".join(self.generated_text_chunks)
        return InferenceResponse(
            request_id=self.request.request_id,
            prompt=self.request.prompt,
            generated_text=full_text,
            prompt_tokens_count=len(self.request.prompt_token_ids),
            generated_tokens_count=len(self.generated_token_ids),
            finish_reason=self.finish_reason,
            ttft_ms=ttft,
            tpot_ms=tpot,
            e2e_latency_ms=e2e,
            queue_latency_ms=queue_latency,
        )
