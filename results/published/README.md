# Published experiment records

Immutable raw JSON from the first M5 Max campaign. Host-wall benchmark samples, GPU replay counters and system timeline summaries have distinct timing scopes. See [the interpreted report](../../docs/initial-results.md).

| Record | Contents |
|---|---|
| [2026-10-18-attn-q-tune.json](2026-10-18-attn-q-tune.json) | 24 ok |
| [2026-10-18-ffn-down-tune.json](2026-10-18-ffn-down-tune.json) | 24 ok |
| [2026-10-18-ffn-int8-before-profile.json](2026-10-18-ffn-int8-before-profile.json) | Metal GPU replay: one quantizer + one GEMM; resource counters |
| [2026-10-18-ffn-int8-tuned-profile.json](2026-10-18-ffn-int8-tuned-profile.json) | Metal GPU replay: one quantizer + one GEMM; resource counters |
| [2026-10-18-ffn-up-tune.json](2026-10-18-ffn-up-tune.json) | 24 ok |
| [2026-10-18-h3-attention-qdq.json](2026-10-18-h3-attention-qdq.json) | 9 ok |
| [2026-10-18-h3-long-attention.json](2026-10-18-h3-long-attention.json) | 3 ok |
| [2026-10-18-h3-mlx-bf16.json](2026-10-18-h3-mlx-bf16.json) | 55 ok |
| [2026-10-18-h3-mpp-precision-sweep.json](2026-10-18-h3-mpp-precision-sweep.json) | 132 ok, 33 unsupported |
| [2026-10-18-h3-mxint8-emulated.json](2026-10-18-h3-mxint8-emulated.json) | 46 ok, 9 unsupported |
| [2026-10-18-h3-rms-fusion.json](2026-10-18-h3-rms-fusion.json) | 6 ok |
| [2026-10-18-independent-validation.json](2026-10-18-independent-validation.json) | 18 ok |
| [2026-10-18-instruments-summary.json](2026-10-18-instruments-summary.json) | Instruments target-only union of GPU activity intervals |
| [2026-10-18-llm-precision-sweep.json](2026-10-18-llm-precision-sweep.json) | 60 ok, 30 unsupported |
| [2026-10-18-mxint8-native-probe.json](2026-10-18-mxint8-native-probe.json) | Native scaled MXINT8 compiler probe: failed |
| [2026-10-18-mxint8-outlier-fallback.json](2026-10-18-mxint8-outlier-fallback.json) | 1 ok |
| [2026-10-18-outliers-validation.json](2026-10-18-outliers-validation.json) | 18 ok; 1 quality rejection |

| [2026-10-18-h3-fp16-sweep.json](2026-10-18-h3-fp16-sweep.json) | 101 ok, 9 unsupported; MLX + MPP FP16 |

`unsupported` entries are explicit adapter/precision exclusions, not silently substituted measurements. `ok` means implementation correctness; the separate quality gate may still reject a quantized result. Tuning records also contain independent-seed holdouts outside the main results list.

The local full captures corresponding to the two GPU reports are `results/local/ffn-int8-before-v2.gputrace` and `results/local/ffn-int8-tuned.gputrace`; the system recording is `results/local/ffn-int8-authorized.trace`. These large binary traces are intentionally not published. [Reproduction commands](../../docs/profiling.md).

## Hadamard follow-up

| Record | Contents |
|---|---|
| [2026-10-18-hadamard-block-sweep.json](2026-10-18-hadamard-block-sweep.json) | 48 ok; 3 quality rejections |
| [2026-10-18-hadamard-down-tune.json](2026-10-18-hadamard-down-tune.json) | 8 ok |
| [2026-10-18-hadamard-gpu-profile.json](2026-10-18-hadamard-gpu-profile.json) | GPU replay: one Hadamard + one quantizer + one GEMM dispatch |
| [2026-10-18-hadamard-implementation-comparison.json](2026-10-18-hadamard-implementation-comparison.json) | 14 ok; 1 quality rejections |
| [2026-10-18-hadamard-independent-normal.json](2026-10-18-hadamard-independent-normal.json) | 6 ok |
| [2026-10-18-hadamard-independent-outliers.json](2026-10-18-hadamard-independent-outliers.json) | 6 ok |
| [2026-10-18-hadamard-large-blocks.json](2026-10-18-hadamard-large-blocks.json) | 12 ok; 1 quality rejections |
| [2026-10-18-hadamard-q-tune.json](2026-10-18-hadamard-q-tune.json) | 8 ok |
| [2026-10-18-hadamard-up-tune.json](2026-10-18-hadamard-up-tune.json) | 8 ok |

## AdaLN + Hadamard + INT8 fusion

See the [fusion report](../../docs/norm-hadamard-fusion.md) for timing scopes, rounding boundaries and measured coverage.

| Record | Contents |
|---|---|
| [2026-10-18-norm-hadamard-sweep.json](2026-10-18-norm-hadamard-sweep.json) | Seed17, 128/256 threads and one/two input reads, 12 ok |
| [2026-10-18-norm-hadamard-independent.json](2026-10-18-norm-hadamard-independent.json) | Seed29, normal/outliers, Q M128/4096/9914 and FFN-up M4096, 24 ok |
| [2026-10-18-norm-hadamard-rerun.json](2026-10-18-norm-hadamard-rerun.json) | Independent seed43 normal-input rerun, 12 ok |
| [2026-10-18-norm-hadamard-fused-profile.json](2026-10-18-norm-hadamard-fused-profile.json) | One fused preparation + one GEMM dispatch; register/spill counters |
| [2026-10-18-norm-hadamard-staged-profile.json](2026-10-18-norm-hadamard-staged-profile.json) | AdaLN + Hadamard + quantization + GEMM: four dispatches |
| [2026-10-18-norm-hadamard-128-profile.json](2026-10-18-norm-hadamard-128-profile.json) | Slower 128-thread buffered variant: 208 spilled bytes in preparation |

## Gated residual + AdaLN + Hadamard + INT8 fusion

See the [gated fusion report](../../docs/gated-residual-fusion.md) for the fair separate-gate comparator, retained BF16 residual output, repeated results and the initial timing anomaly. The seed29/43 online comparison includes writing the updated residual; profile percentages have a separate serial-replay scope.

| Record | Contents |
|---|---|
| [2026-10-18-gated-norm-sweep.json](2026-10-18-gated-norm-sweep.json) | Seed17 Q4096 normal/outliers, 128/256 threads and one/two reads, 12 ok; initial timing anomaly retained |
| [2026-10-18-gated-norm-independent.json](2026-10-18-gated-norm-independent.json) | Seed29 Q4096/9914 and FFN-up4096 normal/outliers, 18 ok |
| [2026-10-18-gated-norm-rerun.json](2026-10-18-gated-norm-rerun.json) | Independent seed43 comparison on the same three shapes/distributions, 18 ok |
| [2026-10-18-gated-norm-separate-profile.json](2026-10-18-gated-norm-separate-profile.json) | Gate addition + existing fused norm/Hadamard/quantizer + GEMM: three dispatches |
| [2026-10-18-gated-norm-fused-profile.json](2026-10-18-gated-norm-fused-profile.json) | Gated preparation + GEMM: two dispatches; preparation has 52 temporary registers and zero spills |

## RMS scale folding and affine shift splitting

See [RMS results](../../docs/rms-scale-folding.md) and [affine policy/costs](../../docs/affine-shift-splitting.md). No full-path performance promotion is claimed. Each RMS campaign has 12 correct records and two weighted/no-H outlier quality rejections; each shift campaign has six correct, quality-passing records.

| Records (2026-10-18 prefix) | Contents |
|---|---|
| `rms-quant-initial`, `rms-quant-initial-rerun` | Seed29/43; eager-gamma folded shader, initial negative result |
| `rms-quant-scheduled`, `rms-quant-scheduled-rerun` | Delayed gamma schedule, same 256 threads; approximately tied |
| `rms-quant-512-independent`, `rms-quant-512-rerun` | 512 threads, 50 warmups; no stable full-path gain |
| `rms-direct-profile`, `rms-folded-initial-profile`, `rms-folded-scheduled-profile` | Two dispatches each; preparation temporary registers 72/96/75, no preparation spills |
| `shift-split-independent`, `shift-split-rerun` | Native-boundary/full-FP32/split policies, including P1 condition cost |
| `shift-split-native-bias`, `shift-split-native-bias-rerun` | Native BF16-to-FP32 bias construction, 50 warmups; complete path still slower |
| `shift-split-profile` | Cached split preparation plus fused-bias INT8 GEMM, two dispatches |
| `bias-cache-independent`, `bias-cache-rerun` | Eight records each: P1/3/60/117, FP32-converting MLX vs BF16-input/FP32-output MPP |
| `shift-split-reference-failure` | Preserved incorrect/not-timed development record: algebraic reference mixed unrounded and FP32-rounded targets; fixed by comparing the same mathematical expression, without loosening its tolerance |

Filenames end in `.json`. All files were copied byte-for-byte from immutable local records. The historical RMS shader versions are preserved in `experiments/rms-scale-folding/`. Large `.gputrace` bundles remain local.
