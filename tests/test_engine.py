"""Unit tests for scheduler and engine components."""
import torch
import pytest

from inference.config import SchedulerConfig
from inference.engine.request import InferenceRequest, SequenceState
from inference.engine.scheduler import Scheduler
from inference.engine.engine import sample_token


def test_scheduler_continuous():
    cfg = SchedulerConfig(policy="continuous", max_batch_size=2)
    sched = Scheduler(cfg)

    req1 = InferenceRequest(prompt="Hello")
    req2 = InferenceRequest(prompt="World")
    req3 = InferenceRequest(prompt="Third")

    sched.enqueue(req1)
    sched.enqueue(req2)
    sched.enqueue(req3)

    to_prefill, to_decode = sched.schedule_next_step()
    assert len(to_prefill) == 2
    assert to_prefill[0].prompt == "Hello"
    assert to_prefill[1].prompt == "World"
    assert sched.queue_size == 1


def test_scheduler_sequential():
    cfg = SchedulerConfig(policy="none", max_batch_size=1)
    sched = Scheduler(cfg)

    req1 = InferenceRequest(prompt="One")
    req2 = InferenceRequest(prompt="Two")
    sched.enqueue(req1)
    sched.enqueue(req2)

    to_prefill, _ = sched.schedule_next_step()
    assert len(to_prefill) == 1
    assert to_prefill[0].prompt == "One"

    # Simulate req1 being active in running
    seq1 = SequenceState(request=to_prefill[0])
    sched.running.append(seq1)

    # Next step should NOT schedule req2 while req1 is running
    to_prefill_2, to_decode_2 = sched.schedule_next_step()
    assert len(to_prefill_2) == 0
    assert len(to_decode_2) == 1

    # Remove finished
    sched.remove_finished([seq1])
    to_prefill_3, _ = sched.schedule_next_step()
    assert len(to_prefill_3) == 1
    assert to_prefill_3[0].prompt == "Two"


def test_sample_token_greedy():
    logits = torch.tensor([[1.0, 5.0, 2.0]])
    tok = sample_token(logits[0], temperature=0.0)
    assert tok == 1


def test_sample_token_sampling():
    logits = torch.tensor([[0.1, 0.2, 10.0]])
    tok = sample_token(logits[0], temperature=0.7, top_p=0.9)
    assert tok == 2
