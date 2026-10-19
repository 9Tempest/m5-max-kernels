# Native gate epilogue fused-campaign-001

All 12 cases pass independent full-K selected-row numerical checks, full-output finite/zero/tail checks, and exact paired hidden/codes/scales checks. This is one campaign using real block-0 weights and synthetic activations, including zero and sparse-outlier rows. Independent rerun, full-model trajectory parity and matched resident performance remain pending. Both flags remain off by default.

[Summary and complete raw timing samples](summary.json). Every linked JSON in `raw/` is copied byte-for-byte, with checksums in the summary. No weights, activation arrays, executables or private logs are included.

| Policy | Logical / padded rows | Seed | Native median ms | Fused median ms | Native / fused |
|---|---:|---:|---:|---:|---:|
| h256 | 2049 / 2176 | 17 | 3.5140 | 3.5270 | 0.99631 |
| attention | 2049 / 2176 | 17 | 1.9790 | 1.9390 | 1.02063 |
| group1024 | 2049 / 2176 | 17 | 4.4390 | 4.3965 | 1.00967 |
| h256 | 9914 / 9984 | 17 | 14.9160 | 14.9080 | 1.00054 |
| attention | 9914 / 9984 | 17 | 8.3460 | 8.1570 | 1.02317 |
| group1024 | 9914 / 9984 | 17 | 18.1370 | 18.0325 | 1.00580 |
| h256 | 2049 / 2176 | 29 | 3.5150 | 3.5250 | 0.99716 |
| attention | 2049 / 2176 | 29 | 2.3765 | 2.3665 | 1.00423 |
| group1024 | 2049 / 2176 | 29 | 4.6060 | 4.5030 | 1.02287 |
| h256 | 9914 / 9984 | 29 | 15.3790 | 15.4130 | 0.99779 |
| attention | 9914 / 9984 | 29 | 8.0410 | 7.5375 | 1.06680 |
| group1024 | 9914 / 9984 | 29 | 17.4060 | 16.9490 | 1.02696 |

These are operator-path observations from one campaign, not an end-to-end speedup or a selected backend. Raw samples retain timing variation; independent reruns and complete model checks are needed before adoption.

Each variant has 3 warmups and 10 alternating-order measured repetitions; seed 29 reverses the starting order. Wall, CPU encode, submit/wait and supplemental command-buffer GPU intervals are retained separately. CPU residual restoration is outside timing. Both paths contain three GPU dispatches: activation quantization/rotation, projection, and next AdaLN/quantization. The native backend already fuses gate/AdaLN/quantization, so moving the gate to GEMM does not remove a dispatch in this measured chain.

The independent reference uses all K terms and all 5,376 channels for rows 0/middle/last. Quantization drift from original BF16 operands is separate from decoded-operand implementation error. Full-output exact paired checks cover all rows, but this does not validate all 50 blocks, real timestep maps, both flags together or the final block endpoint. See the [integration contract](../../../../integrations/h3/fused.md).
