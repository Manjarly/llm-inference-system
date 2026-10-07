"""Scheduler implementations for LLM inference batching:
- Continuous Batching (iteration-level scheduling)
- Static Batching (request-level batching with batch_wait_ms)
- Sequential (baseline 1-request-at-a-time)
"""
from __future__ import annotations

import collections
import time
from typing import List, Tuple, Optional, Deque
from inference.config import SchedulerConfig
from inference.engine.request import InferenceRequest, SequenceState


class Scheduler:
    """Manages queueing and batch scheduling according to the selected policy."""

    def __init__(self, config: SchedulerConfig) -> None:
        self.config = config
        self.policy = config.policy
        self.max_batch_size = config.max_batch_size
        self.batch_wait_s = config.batch_wait_ms / 1000.0

        self.queue: Deque[InferenceRequest] = collections.deque()
        self.running: List[SequenceState] = []

    @property
    def queue_size(self) -> int:
        return len(self.queue)

    @property
    def running_size(self) -> int:
        return len(self.running)

    def enqueue(self, request: InferenceRequest) -> bool:
        """Add request to queue. Returns False if queue capacity exceeded."""
        if len(self.queue) >= self.config.max_queue_size:
            return False
        self.queue.append(request)
        return True

    def remove_finished(self, finished_seqs: List[SequenceState]) -> None:
        """Retire completed sequences from the running set."""
        finished_ids = {s.request.request_id for s in finished_seqs}
        self.running = [s for s in self.running if s.request.request_id not in finished_ids]

    def schedule_next_step(self) -> Tuple[List[InferenceRequest], List[SequenceState]]:
        """Determine which requests to prefill and which active sequences to decode.
        Returns:
            (requests_to_prefill, sequences_to_decode)
        """
        if self.policy == "continuous":
            return self._schedule_continuous()
        elif self.policy == "static":
            return self._schedule_static()
        else:  # "none" / sequential
            return self._schedule_sequential()

    def _schedule_continuous(self) -> Tuple[List[InferenceRequest], List[SequenceState]]:
        """Iteration-level scheduling: fill available batch slots immediately."""
        num_slots = max(0, self.max_batch_size - len(self.running))
        to_prefill: List[InferenceRequest] = []

        while num_slots > 0 and self.queue:
            req = self.queue.popleft()
            if req.aborted:
                continue
            to_prefill.append(req)
            num_slots -= 1

        to_decode = list(self.running)
        return to_prefill, to_decode

    def _schedule_static(self) -> Tuple[List[InferenceRequest], List[SequenceState]]:
        """Static batching: if a batch is currently executing, wait until it finishes completely.
        When idle, accumulate requests up to max_batch_size or until batch_wait_ms elapsed.
        """
        if self.running:
            # Existing batch is still decoding in lockstep
            return [], list(self.running)

        if not self.queue:
            return [], []

        # Check if batch is full or wait timeout reached
        oldest_arrival = self.queue[0].arrival_time
        time_waiting = time.time() - oldest_arrival

        if len(self.queue) >= self.max_batch_size or time_waiting >= self.batch_wait_s:
            to_prefill: List[InferenceRequest] = []
            while len(to_prefill) < self.max_batch_size and self.queue:
                req = self.queue.popleft()
                if not req.aborted:
                    to_prefill.append(req)
            return to_prefill, []

        # Not yet ready to form batch
        return [], []

    def _schedule_sequential(self) -> Tuple[List[InferenceRequest], List[SequenceState]]:
        """Sequential baseline: run 1 request to completion before starting another."""
        if self.running:
            return [], list(self.running)

        while self.queue:
            req = self.queue.popleft()
            if not req.aborted:
                return [req], []

        return [], []
