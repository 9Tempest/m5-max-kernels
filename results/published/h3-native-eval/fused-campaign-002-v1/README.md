# Native gate epilogue fused-campaign-002

All 12 cases in this package pass independent full-K selected-row numerical checks, full-output finite/zero/tail checks, and exact paired hidden/codes/scales checks. This is one campaign using real block-0 weights and synthetic activations, including zero and sparse-outlier rows. This package does not assess independent reruns, full-model trajectory parity or matched resident performance; those require separate evidence. Neither flag is enabled by default by this experiment.

[Summary and complete raw timing samples](summary.json). Every linked JSON in `raw/` is copied byte-for-byte, with checksums in the summary. No weights, activation arrays, executables or private logs are included.

| Policy | Logical / padded rows | Seed | Native median ms | Fused median ms | Native / fused |
|---|---:|---:|---:|---:|---:|
| h256 | 2049 / 2176 | 17 | 4.2305 | 4.2500 | 0.99541 |
| attention | 2049 / 2176 | 17 | 1.8670 | 1.8345 | 1.01772 |
| group1024 | 2049 / 2176 | 17 | 4.6330 | 4.1545 | 1.11518 |
| h256 | 9914 / 9984 | 17 | 14.8795 | 14.8455 | 1.00229 |
| attention | 9914 / 9984 | 17 | 8.3090 | 8.2440 | 1.00788 |
| group1024 | 9914 / 9984 | 17 | 17.3395 | 16.8550 | 1.02875 |
| h256 | 2049 / 2176 | 29 | 4.2635 | 4.1820 | 1.01949 |
| attention | 2049 / 2176 | 29 | 2.5905 | 2.6040 | 0.99482 |
| group1024 | 2049 / 2176 | 29 | 4.7200 | 4.2130 | 1.12034 |
| h256 | 9914 / 9984 | 29 | 14.9320 | 15.0445 | 0.99252 |
| attention | 9914 / 9984 | 29 | 8.4370 | 8.2575 | 1.02174 |
| group1024 | 9914 / 9984 | 29 | 17.4400 | 16.9605 | 1.02827 |

Observed ratio ranges within this campaign (native / fused):

- `attention`: 0.99482–1.02174; fused median lower in 3/4 cases, higher in 1/4, exactly equal in 0/4.
- `group1024`: 1.02827–1.12034; fused median lower in 4/4 cases, higher in 0/4, exactly equal in 0/4.
- `h256`: 0.99252–1.01949; fused median lower in 2/4 cases, higher in 2/4, exactly equal in 0/4.

These are operator-path observations from one campaign, not an end-to-end speedup or a selected backend. Raw samples retain timing variation; independent reruns and complete model checks are needed before adoption.

Each variant has 3 warmups and 10 alternating-order measured repetitions. First measured variants from the driver records: seed 17: fused first; seed 29: native first. Wall, CPU encode, submit/wait and supplemental command-buffer GPU intervals are retained separately. CPU residual restoration is outside timing. Both paths contain three GPU dispatches: activation quantization/rotation, projection, and next AdaLN/quantization. The native backend already fuses gate/AdaLN/quantization, so moving the gate to GEMM does not remove a dispatch in this measured chain.

The independent reference uses all K terms and all 5,376 channels for rows 0/middle/last. Quantization drift from original BF16 operands is separate from decoded-operand implementation error. Full-output exact paired checks cover all rows, but this does not validate all 50 blocks, real timestep maps, both flags together or the final block endpoint. See the [integration contract](../../../../integrations/h3/fused.md).
