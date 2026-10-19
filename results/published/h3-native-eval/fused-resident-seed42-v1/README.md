# Matched resident none / attention comparison

Seed 42; H256 FC2 fixed; the same frozen fused runtime with attention-output gate fusion disabled/enabled.

Each arm completes one full warmup and three timed requests. All eight original 124-frame videos are byte-identical. Raw timing events, callback boundaries, policy/cache log markers, source snapshots and media checksums are verified before export.

| Synchronized host measurement | Baseline median (s) | Attention median (s) | Baseline / attention | Median seconds saved |
| --- | ---: | ---: | ---: | ---: |
| Complete cached generation | 151.126785 | 147.752793 | 1.022835× | 3.373992 |
| Denoise callback | 121.909995 | 118.566479 | 1.028200× | 3.343516 |

Raw samples, ranges and exact process ordering are in [comparison.json](comparison.json). [Baseline evidence](baseline.json) and [attention evidence](candidate.json) are sanitized schema-2 summaries, not byte-identical copies of the raw run records. Original raw-record SHA256s are retained; no model weights or private paths are exported.

One complete warmup and three same-seed timed requests per arm; one sequential process per policy, no interleaving within process.
Generation includes decode and original 124-frame AV export; repeated audio VAE setup prevents a compilation-free end-to-end claim.
Denoise uses synchronized native callback offsets, including online preparation and callback overhead; cached native profile wall is not substituted.
No latent capture is enabled. Shared-lock scheduling plus nonoverlap checks prevent these two arms overlapping; absence of unrelated external GPU load is not proven.
A second seed with reversed process order and independent repeated comparisons is still needed before stable performance promotion.

These descriptive ratios do not establish statistical significance, independent replication or a new best result.
