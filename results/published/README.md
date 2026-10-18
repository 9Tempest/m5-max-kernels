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
