"""Integration tests for FastAPI server and routes."""
import pytest
from starlette.testclient import TestClient

from inference.config import EngineConfig, ModelConfig, SchedulerConfig
from inference.engine.engine import LLMInferenceEngine
from inference.server.app import create_app


@pytest.fixture(scope="module")
def client():
    cfg = EngineConfig(
        model=ModelConfig(model_id="Qwen/Qwen2.5-0.5B", device="auto", dtype="float16", quantization="none"),
        scheduler=SchedulerConfig(policy="continuous", max_batch_size=2),
    )
    engine = LLMInferenceEngine(cfg)
    app = create_app(engine=engine)
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "model_id" in data


def test_models_endpoint(client):
    res = client.get("/v1/models")
    assert res.status_code == 200
    data = res.json()
    assert len(data["data"]) >= 1


def test_metrics_prometheus_endpoint(client):
    res = client.get("/metrics")
    assert res.status_code == 200
    assert "llm_requests_total" in res.text


def test_gpu_status_endpoint(client):
    res = client.get("/gpu/status")
    assert res.status_code == 200
    data = res.json()
    assert "latest" in data


def test_generate_endpoint(client):
    payload = {
        "prompt": "Hello world",
        "max_new_tokens": 5,
        "temperature": 0.0,
        "stream": False,
    }
    res = client.post("/generate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "generated_text" in data
    assert "metrics" in data
    assert data["metrics"]["ttft_ms"] > 0
