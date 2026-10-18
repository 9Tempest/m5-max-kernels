# Measurement contract

## Timing

The Python/MLX harness measures **synchronized host wall latency**, using `perf_counter_ns`, `mx.eval` and `mx.synchronize`. It includes Python dispatch, command submission, required output conversion and synchronization; it is not a Metal GPU timestamp. Compilation and initial warmup are excluded. Keep the raw samples, median and p10/p90. Tiny operators can be dominated by host overhead; GPU timestamp measurements or graph-level fusion are necessary before attributing those differences to arithmetic throughput.

Inputs are resident and reused (warm-cache microbenchmarks). Random generation, model loading, offline weight quantization and correctness reference construction are outside timing. For W8A8/FP8 GEMM, record prequantized GEMM, activation quantization alone, and online activation quantization plus GEMM. The deployed comparison uses the last of these. Output dtype is matched to the floating baseline (BF16 by default). The INT8 epilogue applies FP32 row scales to an INT32 accumulator before BF16 output.

A baseline is installed MLX's `matmul` / fused SDPA / compiled elementwise expression. The candidate is a direct custom MPP TensorOps shader or fused Metal RMSNorm. Report installed framework versions. The FP8 API can execute without proving that hardware has a higher FP8 peak.

`effective_tops = operation_count / synchronized_wall_seconds / 1e12`, counting a multiply-add as two operations. It is an effective rate, not a hardware counter. For GEMM the count is 2MNK. Dense attention uses 4BHS²D (even for causal cases; do not compare its nominal rate to dense full attention without noting masked work). Byte estimates are lower bounds, not measured traffic or roofline bandwidth.

## Correctness and precision

GEMM references use NumPy float64 on up to 32 sampled rows × 32 sampled columns with **full K**, including boundaries. Small dedicated tests cover full outputs. Attention references use 16 selected queries, every key and every batch/head; stable softmax runs in float64. The larger attention reference does not materialize a full S×S matrix. Elementwise references cover all elements in float64.

Every result separates implementation error against decoded operands from quality error against unquantized operands. The original reference starts from actual BF16/FP16 inputs, so it does not charge the input rounding to a quantized candidate. Report relative L2, maximum absolute error, RMSE and cosine similarity. All values must be finite. Default relative-L2 implementation gates are FP32 1e-4, FP16 1e-3, and BF16-output 8e-3. Illustrative total-error budgets are 5% INT8/MXINT8 and 10% FP8. These are microbenchmark screening limits, not H3 visual/audio quality criteria.

An incorrect implementation is not timed or eligible for tuning. A correct implementation that exceeds the quantization quality budget is timed but cannot be selected. Zero and outlier distributions are available. Production promotion additionally requires recorded real activations, repeated seeds, denoiser trajectory checks and final visual/audio evaluation.

## Tuning and evidence

The tuner sweeps 8 combinations of output tiles, SIMD group count and grouped tile traversal, then reruns the selected tile with a fresh seed. For quantized GEMM it selects by the measured online path. This is one local exploration, not a universal kernel dispatch policy. Repeat the baseline and selected tile in a separate run before claiming a stable speedup.

Result JSON is versioned and cannot overwrite an existing file. It contains hardware, OS/compiler/library versions, git revision and dirty state, source hash, seed, workload and every timing sample. Keep failed candidates. Unavailable configurations stay explicit rather than silently falling back. Host/device identifiers, serial numbers, credentials and absolute home paths are not captured.

Compare paper/blog numbers only after matching hardware/core count, dimensions, layout, precision, output type, accuracy and timing scope. Draw Things' reported ~60 TFLOP/s FP16 and ~110 effective TOPS INT8 are context: their shapes, fusion and timing differ from this harness. They are not acceptance thresholds for arbitrary H3 components. A pinned local MFA baseline is still required to evaluate state-of-the-art status.

## Rotated operands

The `mpp-hadamard` path applies the same orthogonal block rotation to A and W. W rotation and quantization are offline. Its `online_activation_plus_gemm` timing includes activation rotation, FP32 conversion, quantization and GEMM. Independent rotation and dot-preservation checks precede timing; decoded quantization error is recorded separately from total original-operand error. See [hadamard.md](hadamard.md).
