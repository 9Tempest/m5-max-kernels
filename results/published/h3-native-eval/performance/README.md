# Native resident performance evidence

[H256 seed42 evidence, schema 2](hadamard-resident-seed42-001-v2.json) preserves the original session events, raw timings, cache checks, precision flags, selected environment and source digests. The earlier schema-1 summary is retained unchanged. Both derive from the same immutable local run; they are not two independent experiments.

The [original wrapper](hadamard-resident-seed42-001-harness/run_native_session.py) and [C driver](hadamard-resident-seed42-001-harness/session.c) preserve the exact source bytes used before the subsequent fusion instrumentation update. Their SHA-256 values match the run evidence. Other pinned dependencies are in `integrations/h3`; use the recorded source hashes when reproducing this historical experiment.

One complete warmup precedes three complete same-seed requests. Denoising time uses fresh synchronized callback boundaries. The upstream cached DiT profile wall timer spans work between requests and is not used as denoising latency. Root command-buffer GPU timestamps do not cover all MPS child work.

Cached full-request latency includes audio VAE setup on every request; compilation-free end-to-end latency is not claimed. This initial one-mode, one-seed run establishes repeatability, not a cross-mode speedup.
