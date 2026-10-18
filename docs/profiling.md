# Performance traces on Apple Silicon

Use Xcode **Metal GPU Capture + GPU Debugger** for individual dispatches, shader resource usage and replay counters. Use Instruments **Metal System Trace** for CPU submission, GPU scheduling, gaps and competing work. These are the Apple tools used here for the profiling workflow requested alongside benchmarking.

## Capture a numerically checked workload

```sh
uv run python -m m5kernels.profile --case h3.ffn_down.m4096 \
  --precision int8 --online --tile 128 128 8 8 \
  --output results/local/ffn-int8.gputrace
uv run python scripts/profile_report.py results/local/ffn-int8.gputrace \
  --expect-dispatches 2 --output results/local/ffn-int8-profile.json
open results/local/ffn-int8.gputrace
```

`--online` captures activation quantization and GEMM; omit it for prequantized GEMM. Weights are prepared offline. Expected target dispatch count is two versus one for these GEMM paths. Other operators can have different counts. Inspect shader names and invocation counts too. MLX can retain a command encoder across capture boundaries; the capture runner flushes it with a disposable marker before the measured work. A marker may appear on other toolchain versions, so inspect a count mismatch instead of silently ignoring it. The capture runner keeps the workload on its original stream.

The report script serializes GPU use with the benchmark lock, replays at the default GPU state, checks the capture, embeds/reloads counter data before querying the lazily populated performance tree, waits for counter postprocessing, exports a compact JSON report and closes its debugger session. Profiling is invasive. Replay GPU durations, synchronized wall timings and Instruments intervals are distinct measurements. Do not compute a benchmark speedup by mixing them.

For manual exploration:

```sh
xcrun gpudebug -t results/local/ffn-int8.gputrace -c status --timeout 300 --json
# Substitute the returned session ID for SESSION.
xcrun gpudebug -s SESSION -c 'go /commands' -c 'list --all' --json
xcrun gpudebug -s SESSION -c 'profile run --gpu-state default --exec serial' --json
xcrun gpudebug -s SESSION -c 'go /performance/shaders' -c 'list --all' --json
xcrun gpudebug -s SESSION -c 'info /performance/shaders/comp0' --json
xcrun gpudebug --terminate SESSION
```

`Temp registers`, `Spilled bytes`, threadgroup memory and instruction counts help form bottleneck hypotheses. They are not direct measurements of DRAM bandwidth or proof of the limiting resource. Compare the same shape, precision and output type before/after tuning; use the unprofiled benchmark to decide whether a change helped. The GPU debugger UI exposes additional timeline/counter detail when supported by the device/toolchain.

## Record CPU/GPU scheduling

```sh
uv run python -m m5kernels.profile --kind system --seconds 5 \
  --case h3.ffn_down.m4096 --precision int8 --online --tile 128 128 8 8 \
  --output results/local/ffn-system.trace
open results/local/ffn-system.trace
xcrun xctrace export --input results/local/ffn-system.trace --toc \
  --output results/local/ffn-system-toc.xml
xcrun xctrace export --input results/local/ffn-system.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]' \
  --output results/local/ffn-system-gpu.xml
# Read the launched Python target PID from the local TOC/recording output.
uv run python scripts/summarize_system_trace.py results/local/ffn-system-gpu.xml \
  --pid TARGET_PID --output results/local/ffn-system-summary.json
```

The worker prints `M5BENCH_READY` after warmup and correctness validation and then runs the hot loop. System tracing includes process initialization; inspect that boundary before attributing a gap to a kernel. The exported summary filters to the target process and unions overlapping/nested GPU intervals, but covers the entire recorded launch. Its busy fraction is not hot-loop utilization and its interval count is not the number of kernel launches.

macOS may request local Developer Tools/GPU profiling authorization the first time. Complete the OS dialog on the machine; credentials never belong in logs. If the worker remains suspended without `M5BENCH_READY`, the recording is incomplete. A successful CLI exit alone is insufficient: confirm target GPU events in the exported table.

Full `.gputrace` bundles contain tensor data and can exceed 1 GB; `.trace` recordings may include system process metadata. Keep them in ignored `results/local/`. Public records contain synthetic-workload metadata and selected counters. Do not commit a real-model capture without reviewing its contents.

References: [MLX Metal debugger](https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html), [Apple interactive GPU tools](https://developer.apple.com/documentation/xcode/debugging-with-interactive-command-line-tools), [programmatic capture](https://developer.apple.com/documentation/xcode/capturing-a-metal-workload-programmatically).
