# Native resident performance evidence

The [matched fusion comparison](../fused-resident-seed42-v1/README.md) adds same-source, same-policy none/attention results: complete-request median 151.126785→147.752793 s, synchronized denoise median 121.909995→118.566479 s. Each arm has one full warmup and three timed requests; all eight videos are byte-identical. The one-seed, fixed-order comparison is descriptive and does not establish stable promotion. Raw sample ranges, actual process intervals and source snapshots are bound in its comparison record.

[H256 seed42 evidence, schema 2](hadamard-resident-seed42-001-v2.json) preserves the original session events, raw timings, cache checks, precision flags, selected environment and source digests. The earlier schema-1 summary is retained unchanged. Both derive from the same immutable local run; they are not two independent experiments.

The [original wrapper](hadamard-resident-seed42-001-harness/run_native_session.py) and [C driver](hadamard-resident-seed42-001-harness/session.c) preserve the exact source bytes used before the subsequent fusion instrumentation update. Their SHA-256 values match the run evidence. Other pinned dependencies are in `integrations/h3`; use the recorded source hashes when reproducing this historical experiment.

One complete warmup precedes three complete same-seed requests. Denoising time uses fresh synchronized callback boundaries. The upstream cached DiT profile wall timer spans work between requests and is not used as denoising latency. Root command-buffer GPU timestamps do not cover all MPS child work.

Cached full-request latency includes audio VAE setup on every request; compilation-free end-to-end latency is not claimed. This initial one-mode, one-seed run establishes repeatability, not a cross-mode speedup.

The [15-second diagnostic comparison](whale-maid-15s-diagnostic-v1.json) separates generation-process wall time, denoising, decoding and tracked DiT allocations for the BF16/H256 pair. It is a single captured pair, not repeated warm timing evidence.

Two later audits preserve the historical raw records: [VAE runtime config pins](runtime-config-audit-15s-v1.json) and [decoded audio duration](audio-duration-audit-v1.json). Their scope and posthoc status are recorded explicitly.
