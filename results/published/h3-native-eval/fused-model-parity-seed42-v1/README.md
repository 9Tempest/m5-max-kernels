# Seed 42 whole-model fusion parity

Attention-only and both-flags runs match the original on-load H256 reference exactly for every packed Euler video/audio tensor (20 steps × two modalities), both final unpacked VAE-input tensors, and both original/exact-duration MP4 files. All 42 tensors in each run were re-read, size/hash checked and scanned for finite values. The completed runner's video-parity boolean is explicitly required to be true and media hashes are independently recomputed.

[Detailed checksums and comparisons](summary.json). Source/patch/model identities, recipe, reference-run binding and generation-log policy markers are verified. Raw JSON files are included only after privacy review; the summary explicitly lists whether each source was copied byte-for-byte or only represented by a sanitized summary. Latent arrays, weights and original logs are excluded.

These are diagnostic five-second fox runs on on-load H256 runtime sources, with readbacks/synchronization. They are not latency samples or a perceptual quality decision. Prepared-checkpoint import uses a different runtime path and has separate byte-parity evidence; this does not validate prepared loading combined with fusion. One seed does not establish every prompt/shape/timestep configuration. Neither flag is enabled by default. Matched same-source resident performance remains a separate experiment.
