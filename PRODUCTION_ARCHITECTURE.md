# Production Inference Architecture Reference
## Kafka Ingestion · GPU Serving · TensorRT · Triton · Prometheus UI

---

## 1. Kafka Multi-Topic Ingestion with Aggregation

### The Problem

Your model expects a single flat feature vector per sample, but the data arrives split
across 3 Kafka topics — each topic carrying a different subset of columns. You need to:

1. Consume all three topics concurrently.
2. Join/aggregate records that belong to the same logical entity (e.g. same `user_id`
   and time window).
3. Run preprocessing (scaling, encoding, embedding) using the transforms fitted during
   training.
4. Batch the prepared tensors and forward them to inference.

### Topic partitioning strategy

```
topic-A  →  numerical + ordinal columns    (high-frequency, e.g. network metrics)
topic-B  →  nominal columns                (medium-frequency, e.g. categorical metadata)
topic-C  →  text columns                   (lower-frequency, e.g. log lines / user agents)
```

Each message carries a join key (e.g. `session_id`) and a timestamp so the aggregator
can correlate records across topics within a time window.

### Structure & pseudocode

```
dfp_pipeline/
└── ingestion/
    ├── __init__.py
    ├── consumer.py        # per-topic Kafka consumer wrapper
    ├── aggregator.py      # sliding-window join across 3 topics
    ├── preprocessor.py    # applies fitted transforms → tensor
    └── pipeline.py        # wires consumer → aggregator → preprocessor → inference
```

#### consumer.py

```python
# Uses confluent-kafka (or aiokafka for async)
# pip install confluent-kafka

class TopicConsumer:
    """
    Wraps a confluent_kafka.Consumer for a single topic.
    Yields deserialized message dicts with a guaranteed 'join_key' field.
    """
    def __init__(self, topic: str, bootstrap_servers: str, group_id: str):
        ...

    def poll_batch(self, max_records: int = 256, timeout_ms: int = 100) -> list[dict]:
        """
        Non-blocking poll. Returns up to max_records messages deserialized
        from JSON (or Avro / Protobuf if you use a schema registry).
        Returns [] if no messages are ready within timeout_ms.
        """
        ...

    def commit(self):
        """Commit offsets after successful processing of a batch."""
        ...
```

#### aggregator.py

```python
from collections import defaultdict
from datetime import datetime, timedelta

class WindowAggregator:
    """
    Holds partial records keyed by join_key.
    A record is 'complete' when all 3 topics have contributed a row for
    the same join_key within the configured time window.

    Strategy: sliding window. Entries older than window_seconds are
    evicted and emitted as-is (partial) or discarded, depending on policy.
    """
    def __init__(self, window_seconds: int = 30, min_topics: int = 3):
        self.window_seconds = window_seconds
        self.min_topics = min_topics
        self._buffer: dict[str, dict] = defaultdict(dict)
        self._timestamps: dict[str, datetime] = {}

    def ingest(self, topic_id: str, record: dict) -> None:
        """
        Merge a single record from topic_id into the buffer under its join_key.

        record must contain: { 'join_key': str, 'event_time': str (ISO), ...columns }
        """
        key = record['join_key']
        # Deep-merge columns from this topic into the buffer entry
        self._buffer[key].update(record)
        self._buffer[key]['__topics_seen__'].add(topic_id)
        self._timestamps[key] = datetime.fromisoformat(record['event_time'])

    def drain_complete(self) -> list[dict]:
        """
        Returns and removes all buffer entries that:
          (a) have seen all min_topics topics, OR
          (b) are older than window_seconds (force-emit partials).
        Caller is responsible for deciding what to do with partials.
        """
        now = datetime.utcnow()
        complete = []
        for key in list(self._buffer):
            entry = self._buffer[key]
            age = (now - self._timestamps[key]).total_seconds()
            is_complete = len(entry.get('__topics_seen__', set())) >= self.min_topics
            is_expired  = age > self.window_seconds
            if is_complete or is_expired:
                complete.append(entry)
                del self._buffer[key]
                del self._timestamps[key]
        return complete
```

#### preprocessor.py

```python
class StreamPreprocessor:
    """
    Applies the same transforms fitted on the training set to streaming records.
    Receives aggregated dicts from WindowAggregator and returns float32 tensors.

    fitted_state comes from TabularDFPDataset.fitted_state (serialized with joblib
    and loaded at startup — see notes on serializing sklearn transforms below).
    """
    def __init__(self, fitted_state: dict, column_config: dict, text_model_name: str,
                 device: str = 'cuda'):
        self.num_scaler   = fitted_state['num_scaler']
        self.ord_encoder  = fitted_state['ord_encoder']
        self.nom_encoder  = fitted_state['nom_encoder']
        self.column_config = column_config
        # SentenceTransformer stays resident in GPU memory — loaded ONCE at startup
        self.st_model = SentenceTransformer(text_model_name, device=device)
        self.device   = device

    def process_batch(self, records: list[dict]) -> torch.Tensor:
        """
        records: list of aggregated dicts from WindowAggregator
        returns: float32 tensor of shape (N, input_dim), on self.device
        """
        df = pd.DataFrame(records)
        # ... same logic as TabularDFPDataset.__init__ but transform-only (no fit)
        # num_arr = self.num_scaler.transform(...)
        # ord_arr = self.ord_encoder.transform(...)
        # nom_arr = self.nom_encoder.transform(...)
        # txt_arr = self.st_model.encode(sentences, ...)
        # return torch.from_numpy(np.concatenate(...)).to(self.device)
        ...
```

#### pipeline.py — the main inference loop

```python
def run_inference_pipeline(
    topics: list[str],           # ['topic-A', 'topic-B', 'topic-C']
    bootstrap_servers: str,
    fitted_state: dict,
    column_config: dict,
    model: TabularAutoencoder,   # pre-loaded, on GPU, in eval() mode
    anomaly_threshold: float,
    prometheus_registry,         # passed in from the metrics module
    inference_batch_size: int = 256,
    window_seconds: int = 30,
):
    """
    Main loop. Runs forever (or until KeyboardInterrupt).

    Flow:
      consumers (3 threads) → shared queue → main thread aggregates
      → preprocessor → model.forward() → anomaly scoring → metrics export
    """
    consumers    = [TopicConsumer(t, bootstrap_servers, group_id='dfp') for t in topics]
    aggregator   = WindowAggregator(window_seconds=window_seconds, min_topics=3)
    preprocessor = StreamPreprocessor(fitted_state, column_config, device='cuda')

    # Each consumer runs in its own thread and pushes to a shared queue
    raw_queue = queue.Queue(maxsize=10_000)

    # --- consumer threads ---
    def _consume(consumer, topic_id):
        while True:
            for record in consumer.poll_batch():
                raw_queue.put((topic_id, record))

    for i, consumer in enumerate(consumers):
        threading.Thread(target=_consume, args=(consumer, topics[i]), daemon=True).start()

    # --- main aggregation + inference loop ---
    pending: list[dict] = []
    while True:
        # Drain raw queue into aggregator
        try:
            while True:
                topic_id, record = raw_queue.get_nowait()
                aggregator.ingest(topic_id, record)
        except queue.Empty:
            pass

        # Get completed records
        complete = aggregator.drain_complete()
        pending.extend(complete)

        # Fire inference when we have a full batch (or flush on timeout)
        if len(pending) >= inference_batch_size:
            batch     = pending[:inference_batch_size]
            pending   = pending[inference_batch_size:]
            tensor    = preprocessor.process_batch(batch)
            _run_and_score(model, tensor, batch, anomaly_threshold, prometheus_registry)

        time.sleep(0.01)  # yield to consumer threads


def _run_and_score(model, tensor, records, threshold, registry):
    """Forward pass + anomaly scoring + metric export. Called from the main loop."""
    with torch.no_grad():
        z     = model.encode(tensor)
        x_hat = model.decoder(z)
        errors = ((tensor - x_hat) ** 2).mean(dim=1).cpu().numpy()

    for i, record in enumerate(records):
        score   = float(errors[i])
        flagged = score > threshold
        # push to Prometheus (see section 4)
        registry.observe(score, flagged, join_key=record['join_key'])
```

### Serializing sklearn transforms for production

The `fitted_state` dict contains sklearn objects. Serialize them with `joblib`, not
`pickle`, for portability:

```python
import joblib

# after training:
joblib.dump(train_ds.fitted_state, 'artifacts/fitted_transforms.joblib')

# at inference startup:
fitted_state = joblib.load('artifacts/fitted_transforms.joblib')
```

---

## 2. Keeping the Model Always on GPU

The key is to load the model **once at process startup** and keep it on the GPU for the
lifetime of the process. Never load/unload per request.

```python
# startup.py  (called once when the inference service starts)

device = torch.device('cuda:0')  # pin to a specific GPU in multi-GPU hosts

model, _, meta = load_checkpoint('checkpoints/best.pth', device=device)
model.eval()
model = model.to(device)

# Optional: torch.compile() for a free ~10-20% speed-up on PyTorch 2.x
# Requires CUDA + a modern GPU. Warm-up pass required before first timed inference.
model = torch.compile(model, mode='reduce-overhead')   # or 'max-autotune'

# Warm up: one dummy forward pass so CUDA kernels are compiled/cached
with torch.no_grad():
    dummy = torch.zeros(1, meta['model_config']['input_dim'], device=device)
    _ = model(dummy)

# model is now resident in GPU VRAM and ready for zero-latency inference
```

`torch.compile()` (introduced in PyTorch 2.0) is your first stop before TensorRT — it's
a one-liner, requires no export step, and often gives a meaningful speedup for MLP-style
models like this autoencoder.

---

## 3. TensorRT and Triton — Education

### 3.1 What is TensorRT?

TensorRT is NVIDIA's inference optimization library. It takes a trained model and
compiles it into a GPU-specific execution engine (.plan file) that runs significantly
faster than standard PyTorch inference.

**What TensorRT does under the hood:**

```
Your PyTorch model
       ↓
  Layer fusion       → combines e.g. Linear + LayerNorm + GELU into a single kernel
                        (eliminates kernel launch overhead and intermediate memory writes)
  Precision tuning   → FP32 → FP16 (or INT8/FP8 with calibration)
                        halves memory bandwidth, doubles theoretical throughput on tensor cores
  Kernel auto-tuning → benchmarks multiple CUDA kernel implementations for each op,
                        picks the fastest one for your specific GPU and batch size
  Memory planning    → pre-allocates all tensor buffers; no dynamic allocation at runtime
       ↓
  .plan file         → compiled, GPU-specific, immutable execution engine
```

The compiled engine is then loaded at inference time — it's just fast binary code
running on the GPU, with near-zero Python overhead.

**How to use it with your PyTorch autoencoder (via Torch-TensorRT):**

```python
# pip install torch-tensorrt
import torch_tensorrt

# After load_checkpoint() and model.eval():

# Step 1: Export to TorchScript (required by Torch-TensorRT)
scripted_model = torch.jit.script(model)

# Step 2: Compile with TensorRT
#   - input_signature defines the shapes TensorRT will optimize for
#   - enabled_precisions: {torch.float32} for FP32, {torch.float16} for FP16
trt_model = torch_tensorrt.compile(
    scripted_model,
    inputs=[
        torch_tensorrt.Input(
            min_shape  =[1,   input_dim],
            opt_shape  =[256, input_dim],   # optimize for this batch size
            max_shape  =[512, input_dim],
            dtype=torch.float16,
        )
    ],
    enabled_precisions={torch.float16},
    workspace_size=1 << 30,   # 1 GB workspace for TRT kernel selection
)

# Step 3: Save the compiled engine
torch.jit.save(trt_model, 'artifacts/autoencoder.trt.pt')

# Inference: identical API, much faster
with torch.no_grad():
    out = trt_model(batch_tensor.half())   # note: FP16 input
```

**Build time vs inference time trade-off:**

The compilation step takes 5–30 minutes (it benchmarks many kernel variants). But the
resulting engine starts in milliseconds and runs at peak GPU throughput. For a production
service that runs for hours or days, this is an excellent trade-off.

**Important caveats:**

- The compiled engine is GPU-architecture specific. An engine compiled on an A100 will
  not run on a T4. Recompile for each target GPU.
- Dynamic shapes are supported but cost some optimization quality. If your batch size
  is fixed or bounded, use tight min/opt/max bounds.
- LayerNorm has historically had limited TRT support. As of TRT 10.x / Torch-TRT 2.x
  it is well supported, but verify with a correctness test after compilation.
- For your MLP autoencoder, `torch.compile()` (section 2) likely gets you 80% of the
  TensorRT benefit with 0% of the complexity. Reach for TensorRT when you need the last
  mile of latency.

**Speedup expectations for an MLP autoencoder:**
- `torch.compile()` alone:          ~10–25% faster than raw PyTorch
- Torch-TensorRT FP16:              ~2–4x faster than raw PyTorch FP32
- Torch-TensorRT FP16 + INT8 quant: ~4–6x, with slight accuracy trade-off

### 3.2 What is Triton Inference Server?

TensorRT is an optimization library — it makes a single model fast. Triton (now
officially NVIDIA Dynamo-Triton as of March 2025) is a model serving framework — it
handles the production deployment concerns that TensorRT doesn't address:

```
                 ┌─────────────────────────────────────────┐
  HTTP/gRPC  →   │          Triton Inference Server        │
  requests       │                                         │
                 │  ┌─────────────┐  ┌─────────────────┐  │
                 │  │  Scheduler  │  │ Dynamic Batching │  │
                 │  └──────┬──────┘  └────────┬────────┘  │
                 │         └─────────┬─────────┘           │
                 │               ┌───▼────┐                │
                 │               │ Model  │  ← TRT engine  │
                 │               │ Backend│     or PyTorch │
                 │               └────────┘                │
                 │  Metrics: GPU util, latency, throughput │
                 └─────────────────────────────────────────┘
```

**Key features relevant to your use case:**

**Dynamic Batching** — Triton collects individual inference requests arriving at
different times and groups them into a single GPU batch automatically. You configure
`preferred_batch_size` and `max_queue_delay_microseconds`. This is what turns a
"1 sample per request" stream into efficient GPU utilization without your application
having to manage batching logic.

```
# model_config for your autoencoder (config.pbtxt in the model repository)
name: "dfp_autoencoder"
backend: "pytorch"           # or "tensorrt" if you compiled it
max_batch_size: 512
input  [{ name: "INPUT__0",  data_type: TYPE_FP32, dims: [384] }]  # input_dim
output [{ name: "OUTPUT__0", data_type: TYPE_FP32, dims: [384] }]
dynamic_batching {
  preferred_batch_size: [64, 128, 256]
  max_queue_delay_microseconds: 2000    # wait up to 2ms to fill a batch
}
instance_group [{ count: 2, kind: KIND_GPU }]  # 2 parallel model instances per GPU
```

**Ensemble pipelines** — Triton can chain models: your sentence-transformer
(pre-processing step) and your autoencoder can be defined as a Triton ensemble, so a
single client request goes through both models server-side. No round-trip.

**Built-in Prometheus metrics** — Triton exposes a `/metrics` endpoint on port 8002 out
of the box (see section 4.2).

**Model repository** — models are files on disk (or S3). Triton hot-reloads them without
downtime when you update the directory.

**When to use Triton vs plain PyTorch serving:**

| Scenario                                         | Recommendation          |
|--------------------------------------------------|-------------------------|
| Single model, single process, tight Kafka loop   | Plain PyTorch + torch.compile |
| Multiple models served (e.g. AE + sentence-TF)  | Triton                  |
| Need dynamic batching without application code   | Triton                  |
| Need horizontal scaling / Kubernetes deployment  | Triton                  |
| Need model versioning + zero-downtime updates    | Triton                  |
| Absolute minimum latency, fixed batch size       | TensorRT engine only    |

**For the DFP pipeline:** If you're running a single autoencoder in a tight Kafka loop
on one GPU, plain PyTorch (with `torch.compile`) is simpler and sufficient. If you need
to serve multiple models, handle concurrent REST/gRPC requests from multiple consumers,
or deploy on Kubernetes, Triton is the right layer to add.

### 3.3 The full NVIDIA inference stack — how it fits together

```
Training   →  PyTorch (.pth checkpoint)
               ↓
Optimise   →  Torch-TensorRT  →  .trt.pt engine
               ↓
Serve      →  Triton (config.pbtxt + engine file in model repo)
               ↓  (HTTP/gRPC)
Consumer   →  Your Kafka pipeline calls Triton via tritonclient
               ↓
Observe    →  Triton /metrics endpoint  →  Prometheus  →  Grafana
```

---

## 4. Prometheus-Based Inference Monitoring UI

### 4.1 What Prometheus scrapes

Prometheus is a pull-based metrics system. Your application exposes an HTTP endpoint
(`/metrics`), and Prometheus scrapes it on a configured interval (e.g. every 15s).
You view and alert on the collected time series in Grafana (or the built-in Prometheus
expression browser).

**Metrics you want for the DFP inference pipeline:**

```
# Counters (monotonically increasing)
dfp_messages_consumed_total{topic}          → records ingested per topic
dfp_records_aggregated_total               → records that completed all 3 topics
dfp_inferences_total                        → total model forward passes
dfp_anomalies_detected_total               → samples above threshold

# Histograms (track distributions + percentiles)
dfp_reconstruction_error_bucket{le}        → distribution of per-sample MSE scores
dfp_inference_latency_seconds_bucket{le}   → end-to-end latency per batch
dfp_batch_size_bucket{le}                  → actual batch sizes at inference time

# Gauges (current value, can go up/down)
dfp_aggregation_buffer_size               → current records waiting in WindowAggregator
dfp_kafka_consumer_lag{topic, partition}  → Kafka consumer lag per topic/partition
dfp_gpu_memory_used_bytes                 → GPU VRAM in use (from pynvml or Triton)
dfp_model_threshold                       → current anomaly threshold value
```

### 4.2 Structure & pseudocode

```
dfp_pipeline/
└── monitoring/
    ├── __init__.py
    ├── metrics.py          # Prometheus metric definitions + registry
    ├── server.py           # exposes /metrics HTTP endpoint
    └── dashboard.json      # Grafana dashboard JSON (import in Grafana UI)
```

#### metrics.py

```python
# pip install prometheus-client
from prometheus_client import (
    Counter, Histogram, Gauge,
    CollectorRegistry, push_to_gateway,
    REGISTRY,
)

# Use a custom registry if you want to avoid the default global one
# (useful when running multiple workers in the same process)
REGISTRY = CollectorRegistry()

# --- Counters ---
MESSAGES_CONSUMED = Counter(
    'dfp_messages_consumed_total',
    'Kafka messages consumed',
    labelnames=['topic'],
    registry=REGISTRY,
)
RECORDS_AGGREGATED = Counter(
    'dfp_records_aggregated_total',
    'Records that completed aggregation (all topics present)',
    registry=REGISTRY,
)
INFERENCES_TOTAL = Counter(
    'dfp_inferences_total',
    'Total samples forwarded through the model',
    registry=REGISTRY,
)
ANOMALIES_TOTAL = Counter(
    'dfp_anomalies_detected_total',
    'Samples flagged as anomalous (score > threshold)',
    registry=REGISTRY,
)

# --- Histograms ---
# Buckets chosen for typical MSE reconstruction error range.
# Tune based on your validation set's error distribution.
RECON_ERROR_HIST = Histogram(
    'dfp_reconstruction_error',
    'Per-sample reconstruction error (MSE)',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, float('inf')],
    registry=REGISTRY,
)
INFERENCE_LATENCY = Histogram(
    'dfp_inference_latency_seconds',
    'End-to-end latency per inference batch (preprocess + forward + score)',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1.0],
    registry=REGISTRY,
)
BATCH_SIZE_HIST = Histogram(
    'dfp_batch_size',
    'Batch sizes at inference time',
    buckets=[1, 8, 16, 32, 64, 128, 256, 512],
    registry=REGISTRY,
)

# --- Gauges ---
BUFFER_SIZE = Gauge(
    'dfp_aggregation_buffer_size',
    'Current number of partial records in the aggregation buffer',
    registry=REGISTRY,
)
GPU_MEMORY = Gauge(
    'dfp_gpu_memory_used_bytes',
    'GPU VRAM currently in use',
    registry=REGISTRY,
)
ANOMALY_THRESHOLD = Gauge(
    'dfp_model_threshold',
    'Current anomaly detection threshold',
    registry=REGISTRY,
)


class InferenceMetrics:
    """
    Thin wrapper passed into the inference pipeline.
    Encapsulates all metric recording so the pipeline code stays clean.
    """

    def record_consumed(self, topic: str, n: int = 1):
        MESSAGES_CONSUMED.labels(topic=topic).inc(n)

    def record_aggregated(self, n: int = 1):
        RECORDS_AGGREGATED.inc(n)

    def record_inference_batch(
        self,
        errors: np.ndarray,
        threshold: float,
        latency_seconds: float,
    ):
        batch_size = len(errors)
        INFERENCES_TOTAL.inc(batch_size)
        BATCH_SIZE_HIST.observe(batch_size)
        INFERENCE_LATENCY.observe(latency_seconds)

        n_anomalies = int((errors > threshold).sum())
        ANOMALIES_TOTAL.inc(n_anomalies)

        for score in errors:
            RECON_ERROR_HIST.observe(float(score))

    def set_buffer_size(self, n: int):
        BUFFER_SIZE.set(n)

    def set_gpu_memory(self):
        # requires: pip install pynvml
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info   = pynvml.nvmlDeviceGetMemoryInfo(handle)
            GPU_MEMORY.set(info.used)
        except Exception:
            pass

    def set_threshold(self, threshold: float):
        ANOMALY_THRESHOLD.set(threshold)
```

#### server.py — expose /metrics endpoint

```python
# prometheus_client has a built-in WSGI server for the /metrics endpoint.
# Run this in a background thread alongside your Kafka pipeline.

from prometheus_client import start_http_server

def start_metrics_server(port: int = 8001):
    """
    Starts the Prometheus /metrics HTTP server on the given port.
    Prometheus scrapes: GET http://<host>:<port>/metrics
    Call once at startup, before the inference loop.
    """
    start_http_server(port, registry=REGISTRY)
    print(f"Prometheus metrics available at http://0.0.0.0:{port}/metrics")
```

#### Integration into pipeline.py

```python
# In run_inference_pipeline():

metrics = InferenceMetrics()
metrics.set_threshold(anomaly_threshold)
start_metrics_server(port=8001)  # runs in background thread

# In the consumer thread:
for record in consumer.poll_batch():
    raw_queue.put((topic_id, record))
    metrics.record_consumed(topic_id)

# In _run_and_score():
import time
t0 = time.perf_counter()

with torch.no_grad():
    z     = model.encode(tensor)
    x_hat = model.decoder(z)
    errors = ((tensor - x_hat) ** 2).mean(dim=1).cpu().numpy()

latency = time.perf_counter() - t0
metrics.record_inference_batch(errors, threshold, latency)
metrics.set_buffer_size(aggregator.buffer_size)
metrics.set_gpu_memory()
```

### 4.3 Triton's built-in Prometheus endpoint

If you use Triton, it already exposes these metrics at port 8002 with no extra code:

```
nv_gpu_utilization{gpu_uuid}              → GPU utilisation %
nv_gpu_memory_used_bytes{gpu_uuid}        → VRAM used
nv_inference_request_success{model}       → successful inference count
nv_inference_queue_duration_us{model}     → time requests spent queuing (batch fill)
nv_inference_compute_duration_us{model}   → GPU compute time
nv_inference_output_tensor_io_bytes{model}→ output bytes (throughput proxy)
```

Scrape config for prometheus.yml:

```yaml
scrape_configs:
  - job_name: dfp_pipeline
    static_configs:
      - targets: ['localhost:8001']     # your app metrics
    scrape_interval: 15s

  - job_name: triton
    static_configs:
      - targets: ['localhost:8002']     # triton built-in metrics
    metrics_path: /metrics
    scrape_interval: 15s
```

### 4.4 Grafana dashboard panels (what to build)

```
Row 1: Throughput & Anomaly Rate
  [Stat]  Records/sec ingested (rate of dfp_messages_consumed_total)
  [Stat]  Anomaly rate % (dfp_anomalies_detected_total / dfp_inferences_total)
  [Graph] Records ingested vs anomalies detected over time

Row 2: Model Performance
  [Heatmap]   Reconstruction error distribution over time
  [Graph]     P50, P95, P99 reconstruction error (histogram_quantile)
  [Graph]     P50, P95, P99 inference latency

Row 3: Infrastructure Health
  [Gauge]     GPU VRAM used (dfp_gpu_memory_used_bytes)
  [Graph]     GPU utilization % (from Triton nv_gpu_utilization)
  [Graph]     Kafka consumer lag per topic (dfp_kafka_consumer_lag)
  [Graph]     Aggregation buffer size (dfp_aggregation_buffer_size)
```

**Useful PromQL queries:**

```promql
# Anomaly rate (1-minute window)
rate(dfp_anomalies_detected_total[1m]) / rate(dfp_inferences_total[1m])

# 99th percentile reconstruction error
histogram_quantile(0.99, rate(dfp_reconstruction_error_bucket[5m]))

# P95 inference latency
histogram_quantile(0.95, rate(dfp_inference_latency_seconds_bucket[5m]))

# Throughput (records/sec)
sum(rate(dfp_messages_consumed_total[1m])) by (topic)

# Inference batch fill efficiency
histogram_quantile(0.50, rate(dfp_batch_size_bucket[5m]))
```

### 4.5 Alerting rules

```yaml
# prometheus/rules/dfp_alerts.yml
groups:
  - name: dfp
    rules:
      - alert: AnomalyRateHigh
        expr: |
          rate(dfp_anomalies_detected_total[5m])
          / rate(dfp_inferences_total[5m]) > 0.1
        for: 2m
        labels:    { severity: warning }
        annotations:
          summary: "Anomaly rate above 10% for 2 minutes"

      - alert: InferenceLatencyHigh
        expr: |
          histogram_quantile(0.95,
            rate(dfp_inference_latency_seconds_bucket[5m])) > 0.1
        for: 1m
        labels:    { severity: warning }
        annotations:
          summary: "P95 inference latency above 100ms"

      - alert: KafkaConsumerLagHigh
        expr: dfp_kafka_consumer_lag > 10000
        for: 5m
        labels:    { severity: critical }
        annotations:
          summary: "Kafka consumer is falling behind"

      - alert: GPUMemoryFull
        expr: dfp_gpu_memory_used_bytes / 1e9 > 20   # 20 GB
        for: 1m
        labels:    { severity: critical }
        annotations:
          summary: "GPU VRAM nearly exhausted"
```

---

## 5. Updated Module Map

```
dfp_pipeline/
├── data/
│   ├── dataset.py           # TabularDFPDataset (training-time)
│   └── dataloader.py        # build_dataloaders()
├── model/
│   └── autoencoder.py       # TabularAutoencoder
├── training/
│   ├── trainer.py           # Trainer + TensorBoard
│   └── checkpoint.py        # save / load
├── inference/
│   └── infer.py             # batch offline inference
├── ingestion/               # ← NEW
│   ├── consumer.py          # Kafka topic consumer
│   ├── aggregator.py        # sliding-window join
│   ├── preprocessor.py      # streaming transform → tensor
│   └── pipeline.py          # main inference loop
├── monitoring/              # ← NEW
│   ├── metrics.py           # Prometheus counters / histograms / gauges
│   ├── server.py            # /metrics HTTP endpoint
│   └── dashboard.json       # Grafana dashboard (importable)
├── utils/
│   ├── seed.py
│   └── config.py
├── train.py
└── config.yaml
```
