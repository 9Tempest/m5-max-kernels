# RMS scheduling snapshots

These original snapshots preserve the shader variants used by the published 2026-10-18 experiment:

- `rms_quant_eager_gamma.metal`: the initial folded variant weights all buffered values in the input loop, then reduces square sums after H. Its H256 profile reports 96 registers.
- `rms_quant_scheduled.metal`: weights values immediately before their butterflies and reduces the SIMD-local square sum before H. Its H256 profile reports 75 registers. This is the current active shader.

Both use the substitutions in `src/m5kernels/rms_quant.py`. To reproduce the initial experiment in an isolated checkout, substitute its snapshot for `src/m5kernels/shaders/rms_quant.metal` and run `scripts/rms_quant_campaign.py`; retain the dirty-state/source digest and choose a new result filename. Direct-branch arithmetic is unchanged. These files are evidence, not separate production APIs.
