// Real-time Dashboard Logic for LLM Inference Engine

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const deviceName = document.getElementById("deviceName");
  const gpuDeviceType = document.getElementById("gpuDeviceType");
  const gpuUtilVal = document.getElementById("gpuUtilVal");
  const gpuUtilBar = document.getElementById("gpuUtilBar");
  const peakVramVal = document.getElementById("peakVramVal");
  const vramUsedVal = document.getElementById("vramUsedVal");
  const throughputVal = document.getElementById("throughputVal");
  const rpsVal = document.getElementById("rpsVal");
  const completedReqsVal = document.getElementById("completedReqsVal");
  const runningSeqsVal = document.getElementById("runningSeqsVal");
  const maxBatchVal = document.getElementById("maxBatchVal");
  const maxBatchSubVal = document.getElementById("maxBatchSubVal");
  const queueDepthVal = document.getElementById("queueDepthVal");
  const schedulerPolicyBadge = document.getElementById("schedulerPolicyBadge");

  const ttftP50Val = document.getElementById("ttftP50Val");
  const ttftP90Val = document.getElementById("ttftP90Val");
  const tpotP50Val = document.getElementById("tpotP50Val");
  const tpotP90Val = document.getElementById("tpotP90Val");
  const e2eP50Val = document.getElementById("e2eP50Val");
  const e2eP90Val = document.getElementById("e2eP90Val");

  // Playground elements
  const promptInput = document.getElementById("promptInput");
  const maxTokensSlider = document.getElementById("maxTokensSlider");
  const maxTokensVal = document.getElementById("maxTokensVal");
  const tempSlider = document.getElementById("tempSlider");
  const tempVal = document.getElementById("tempVal");
  const btnGenerate = document.getElementById("btnGenerate");
  const btnClear = document.getElementById("btnClear");
  const outputBox = document.getElementById("outputBox");
  const pgBadge = document.getElementById("playgroundLatencyBadge");
  const pgTtftTag = document.getElementById("pgTtftTag");
  const pgTpotTag = document.getElementById("pgTpotTag");
  const pgE2eTag = document.getElementById("pgE2eTag");

  // Load test elements
  const ltConcurrency = document.getElementById("ltConcurrency");
  const ltConcurrencyVal = document.getElementById("ltConcurrencyVal");
  const ltTotalRequests = document.getElementById("ltTotalRequests");
  const ltTotalReqVal = document.getElementById("ltTotalReqVal");
  const btnRunLoadTest = document.getElementById("btnRunLoadTest");
  const ltProgressContainer = document.getElementById("ltProgressContainer");
  const ltProgressBar = document.getElementById("ltProgressBar");
  const ltProgressText = document.getElementById("ltProgressText");
  const ltStatusText = document.getElementById("ltStatusText");
  const ltResultsCard = document.getElementById("ltResultsCard");
  const ltResSuccess = document.getElementById("ltResSuccess");
  const ltResThroughput = document.getElementById("ltResThroughput");
  const ltResTtft = document.getElementById("ltResTtft");
  const ltResTpot = document.getElementById("ltResTpot");
  const ltResP95 = document.getElementById("ltResP95");
  const ltResE2e = document.getElementById("ltResE2e");

  // Specs
  const specModelId = document.getElementById("specModelId");
  const specPrecision = document.getElementById("specPrecision");
  const specBatchPolicy = document.getElementById("specBatchPolicy");

  // Slider events
  maxTokensSlider.addEventListener("input", (e) => {
    maxTokensVal.textContent = e.target.value;
  });
  tempSlider.addEventListener("input", (e) => {
    tempVal.textContent = e.target.value;
  });
  ltConcurrency.addEventListener("input", (e) => {
    ltConcurrencyVal.textContent = e.target.value;
  });
  ltTotalRequests.addEventListener("input", (e) => {
    ltTotalReqVal.textContent = e.target.value;
  });

  btnClear.addEventListener("click", () => {
    outputBox.textContent = "";
    pgBadge.style.display = "none";
  });

  // Telemetry Poller
  async function fetchTelemetry() {
    try {
      const res = await fetch("/engine/status");
      if (!res.ok) return;
      const data = await res.json();

      // Hardware
      if (data.hardware && data.hardware.latest) {
        const hw = data.hardware.latest;
        deviceName.textContent = hw.device_name || "Hardware";
        gpuDeviceType.textContent = hw.device_type.toUpperCase();
        const util = hw.gpu_utilization_pct || 0;
        gpuUtilVal.textContent = `${util.toFixed(1)}%`;
        gpuUtilBar.style.width = `${Math.min(100, util)}%`;
        vramUsedVal.textContent = `${hw.gpu_memory_used_mb.toFixed(1)} MB`;
        peakVramVal.textContent = `${(data.hardware.peak_gpu_memory_mb || hw.gpu_memory_used_mb).toFixed(1)} MB`;
      }

      // Throughput
      if (data.metrics && data.metrics.throughput) {
        const tp = data.metrics.throughput;
        throughputVal.innerHTML = `${tp.gen_tokens_per_sec.toFixed(1)} <span class="unit">tok/s</span>`;
        rpsVal.textContent = `${tp.requests_per_sec.toFixed(2)} req/s`;
      }
      if (data.metrics && data.metrics.counters) {
        completedReqsVal.textContent = data.metrics.counters.completed_requests;
      }

      // Scheduler
      if (data.scheduler) {
        runningSeqsVal.textContent = data.scheduler.running_sequences;
        maxBatchVal.textContent = data.scheduler.max_batch_size;
        maxBatchSubVal.textContent = `${data.scheduler.max_batch_size} slots`;
        queueDepthVal.textContent = data.scheduler.queued_requests;
        schedulerPolicyBadge.textContent = data.scheduler.policy.toUpperCase();
        if (specBatchPolicy) {
          specBatchPolicy.textContent = data.scheduler.policy.toUpperCase();
        }
      }

      // Latency Percentiles
      if (data.metrics && data.metrics.latency_ms) {
        const lat = data.metrics.latency_ms;
        if (lat.ttft.avg > 0) {
          ttftP50Val.textContent = `${lat.ttft.p50.toFixed(1)} ms`;
          ttftP90Val.textContent = `(${lat.ttft.p90.toFixed(1)} ms p90)`;
        }
        if (lat.tpot.avg > 0) {
          tpotP50Val.textContent = `${lat.tpot.p50.toFixed(1)} ms`;
          tpotP90Val.textContent = `(${lat.tpot.p90.toFixed(1)} ms p90)`;
        }
        if (lat.e2e.avg > 0) {
          e2eP50Val.textContent = `${lat.e2e.p50.toFixed(1)} ms`;
          e2eP90Val.textContent = `(${lat.e2e.p90.toFixed(1)} ms p90)`;
        }
      }

      // Model Specs
      if (data.model) {
        if (specModelId) specModelId.textContent = data.model.model_id;
        if (specPrecision) {
          const quant = data.model.quantization || "none";
          specPrecision.textContent = quant !== "none" ? `Quantized (${quant.toUpperCase()})` : data.model.dtype.toUpperCase();
        }
      }
    } catch (err) {
      console.debug("Telemetry error:", err);
    }
  }

  // Poll every 1s
  setInterval(fetchTelemetry, 1000);
  fetchTelemetry();

  // ---------------- Streaming Generation ----------------
  btnGenerate.addEventListener("click", async () => {
    const prompt = promptInput.value.trim();
    if (!prompt) return;

    btnGenerate.disabled = true;
    outputBox.textContent = "";
    pgBadge.style.display = "none";

    const tStart = performance.now();
    let tFirstToken = null;
    let tokenCount = 0;

    try {
      const response = await fetch("/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: prompt,
          max_new_tokens: parseInt(maxTokensSlider.value, 10),
          temperature: parseFloat(tempSlider.value),
          top_p: 0.9,
          stream: true,
        }),
      });

      if (!response.ok) {
        outputBox.textContent = `Error: ${response.status} ${response.statusText}`;
        btnGenerate.disabled = false;
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop(); // keep remainder

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const raw = line.slice(6).trim();
            if (raw === "[DONE]") continue;
            try {
              const parsed = JSON.parse(raw);
              if (parsed.token) {
                if (tFirstToken === null) {
                  tFirstToken = performance.now();
                }
                tokenCount++;
                outputBox.textContent += parsed.token;
                outputBox.scrollTop = outputBox.scrollHeight;
              }
            } catch (e) {
              // Non-JSON chunk
            }
          }
        }
      }

      const tEnd = performance.now();
      const ttft = tFirstToken ? (tFirstToken - tStart) : 0;
      const e2e = tEnd - tStart;
      const tpot = (tokenCount > 1 && tFirstToken) ? ((tEnd - tFirstToken) / (tokenCount - 1)) : 0;

      pgTtftTag.textContent = `TTFT: ${ttft.toFixed(1)}ms`;
      pgTpotTag.textContent = `TPOT: ${tpot.toFixed(1)}ms`;
      pgE2eTag.textContent = `E2E: ${e2e.toFixed(1)}ms (${tokenCount} tokens)`;
      pgBadge.style.display = "flex";

    } catch (err) {
      outputBox.textContent = `Request failed: ${err.message}`;
    } finally {
      btnGenerate.disabled = false;
      fetchTelemetry();
    }
  });

  // ---------------- Load Tester ----------------
  btnRunLoadTest.addEventListener("click", async () => {
    const concurrency = parseInt(ltConcurrency.value, 10);
    const totalRequests = parseInt(ltTotalRequests.value, 10);

    btnRunLoadTest.disabled = true;
    ltProgressContainer.style.display = "flex";
    ltResultsCard.style.display = "none";
    ltProgressBar.style.width = "0%";
    ltStatusText.textContent = `Running ${concurrency} concurrent clients...`;

    const samplePrompts = [
      "Artificial intelligence is defined as",
      "Deep learning requires significant GPU memory bandwidth because",
      "The key difference between latency and throughput in LLM serving is",
      "Continuous batching improves overall system throughput by",
      "Quantization from FP16 to INT4 reduces memory footprint and",
      "Modern transformer architectures utilize attention mechanisms to",
      "High throughput inference servers schedule incoming requests using",
      "Time to first token (TTFT) measures how fast the engine can prefill and",
    ];

    let completed = 0;
    let failed = 0;
    const requestResults = [];
    const tBenchStart = performance.now();

    let nextIndex = 0;

    async function worker() {
      while (nextIndex < totalRequests) {
        const idx = nextIndex++;
        const prompt = samplePrompts[idx % samplePrompts.length];
        const tStart = performance.now();

        try {
          const resp = await fetch("/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              prompt: prompt,
              max_new_tokens: 16,
              temperature: 0.7,
              stream: false,
            }),
          });

          const tEnd = performance.now();
          if (resp.ok) {
            const data = await resp.json();
            requestResults.push({
              ttft_ms: data.metrics.ttft_ms || (tEnd - tStart),
              tpot_ms: data.metrics.tpot_ms || 0,
              e2e_ms: tEnd - tStart,
              tokens: data.usage ? data.usage.completion_tokens : 16,
            });
            completed++;
          } else {
            failed++;
          }
        } catch (err) {
          failed++;
        }

        const pct = Math.round(((completed + failed) / totalRequests) * 100);
        ltProgressBar.style.width = `${pct}%`;
        ltProgressText.textContent = `${completed + failed} / ${totalRequests} completed`;
      }
    }

    // Launch worker pool
    const workers = [];
    for (let i = 0; i < concurrency; i++) {
      workers.push(worker());
    }

    await Promise.all(workers);
    const tBenchEnd = performance.now();
    const benchTotalSeconds = (tBenchEnd - tBenchStart) / 1000.0;

    // Calculate metrics
    const totalGenTokens = requestResults.reduce((acc, r) => acc + r.tokens, 0);
    const throughput = benchTotalSeconds > 0 ? (totalGenTokens / benchTotalSeconds) : 0;
    const avgTtft = requestResults.length > 0 ? (requestResults.reduce((acc, r) => acc + r.ttft_ms, 0) / requestResults.length) : 0;
    const avgTpot = requestResults.length > 0 ? (requestResults.reduce((acc, r) => acc + r.tpot_ms, 0) / requestResults.length) : 0;
    const avgE2e = requestResults.length > 0 ? (requestResults.reduce((acc, r) => acc + r.e2e_ms, 0) / requestResults.length) : 0;

    const e2eSorted = requestResults.map(r => r.e2e_ms).sort((a, b) => a - b);
    const p95Idx = Math.floor(e2eSorted.length * 0.95);
    const p95E2e = e2eSorted[p95Idx] || avgE2e;

    // Render results
    ltResSuccess.textContent = `${completed} / ${totalRequests}`;
    ltResThroughput.textContent = `${throughput.toFixed(1)} tok/s`;
    ltResTtft.textContent = `${avgTtft.toFixed(1)} ms`;
    ltResTpot.textContent = `${avgTpot.toFixed(1)} ms`;
    ltResP95.textContent = `${p95E2e.toFixed(1)} ms`;
    ltResE2e.textContent = `${avgE2e.toFixed(1)} ms`;

    ltStatusText.textContent = `Completed in ${benchTotalSeconds.toFixed(2)}s`;
    ltResultsCard.style.display = "block";
    btnRunLoadTest.disabled = false;

    fetchTelemetry();
  });
});
