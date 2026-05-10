# Run this script on an AMD ROCm machine to generate a real GPU trace.
# Output: assets/gpu_trace.json — open in chrome://tracing
# Note: commit the script, not the trace file (add gpu_trace.json to .gitignore)
import torch
from torch.profiler import profile, ProfilerActivity
import pandas as pd
from engine.ai_inference import run_batch_inference

# Generate dummy data
data = {
    "symbol": ["AMD"] * 1000,
    "timestamp": pd.date_range("2024-01-01", periods=1000, freq="1min"),
    "open": 150.0,
    "high": 152.0,
    "low": 149.0,
    "close": 151.0,
    "volume": 1_000_000,
}
df = pd.DataFrame(data)

print("Starting GPU profiling...")

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
) as prof:
    result = run_batch_inference(df, prefer_gpu=True)

print(f"Device: {result.device}")
print(f"Latency: {result.latency_ms:.2f}ms")
print(f"Throughput: {result.throughput_rows_per_second:.0f} rows/sec")

# Export trace
prof.export_chrome_trace("assets/gpu_trace.json")
print("\n✅ Trace saved to assets/gpu_trace.json")
print("📊 Open chrome://tracing in Chrome and load the file")
