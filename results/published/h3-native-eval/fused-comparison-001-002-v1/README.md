# Independent native fusion campaign comparison

This joins `fused-campaign-001` and `fused-campaign-002` after revalidating all 24 raw case/driver reports. Both campaigns cover two seeds (17/29), logical rows 2049/9914, and H256 FC2, group1024 FC2 and attention output. All independent numerical and exact paired output contracts pass.

[Machine-readable comparison, raw timing samples and source hashes](comparison.json). The original published campaign JSON is unchanged. Each ratio is native wall median / fused wall median; a ratio above one means a lower fused median in that observation, not a statistical significance result.

| Policy | Rows | Seed | First native / fused ms | First ratio | Second native / fused ms | Second ratio | Lower fused median in both? |
|---|---:|---:|---:|---:|---:|---:|---|
| h256 | 2049 | 17 | 3.5140 / 3.5270 | 0.99631 | 4.2305 / 4.2500 | 0.99541 | no |
| attention | 2049 | 17 | 1.9790 / 1.9390 | 1.02063 | 1.8670 / 1.8345 | 1.01772 | yes |
| group1024 | 2049 | 17 | 4.4390 / 4.3965 | 1.00967 | 4.6330 / 4.1545 | 1.11518 | yes |
| h256 | 9914 | 17 | 14.9160 / 14.9080 | 1.00054 | 14.8795 / 14.8455 | 1.00229 | yes |
| attention | 9914 | 17 | 8.3460 / 8.1570 | 1.02317 | 8.3090 / 8.2440 | 1.00788 | yes |
| group1024 | 9914 | 17 | 18.1370 / 18.0325 | 1.00580 | 17.3395 / 16.8550 | 1.02875 | yes |
| h256 | 2049 | 29 | 3.5150 / 3.5250 | 0.99716 | 4.2635 / 4.1820 | 1.01949 | no |
| attention | 2049 | 29 | 2.3765 / 2.3665 | 1.00423 | 2.5905 / 2.6040 | 0.99482 | no |
| group1024 | 2049 | 29 | 4.6060 / 4.5030 | 1.02287 | 4.7200 / 4.2130 | 1.12034 | yes |
| h256 | 9914 | 29 | 15.3790 / 15.4130 | 0.99779 | 14.9320 / 15.0445 | 0.99252 | no |
| attention | 9914 | 29 | 8.0410 / 7.5375 | 1.06680 | 8.4370 / 8.2575 | 1.02174 | yes |
| group1024 | 9914 | 29 | 17.4060 / 16.9490 | 1.02696 | 17.4400 / 16.9605 | 1.02827 | yes |

The recorded device/software environment, model identity, runtime patch, driver/reference sources, wrapper and linked executable match. Library archive hashes differ and remain visible; the comparison requires byte-identical linked executable hashes. Each path has the same warmup/repetition counts and alternating order. The campaign intervals do not overlap.

CPU encode, submit/wait and supplemental command-buffer GPU timings remain separate in every raw sample. Timing includes online activation packing/rotation, native layout/padding, GEMM/gate and next AdaLN/packing. Residual restoration, correctness readback, prior compilation and offline weight preparation are excluded. Both paths have three timed dispatches.

Read gains at the matched policy and shape. A group1024 FC2 gain is not evidence for H256 FC2, and two block-0 operator campaigns do not validate real timestep maps, all 50 blocks, final endpoints or complete generation performance. Whole-model exact parity and matched resident requests remain separate evidence.
