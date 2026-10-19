# Native QKV / AdaLN-H256 integration evidence

[Measured results](results.md) · [Playable gallery](gallery.html) · [File manifest and checksums](manifest.json)

This package preserves the selected operator campaigns, model evaluations, resident comparisons, default decisions, activation analyses and run metadata as byte-identical files. Each manifest entry records its repository-relative source, public destination, byte count and SHA256. Source report statuses, unsupported cases and numerical limits are unchanged.

Publication verifies file identity and report-to-run joins. It does not rerun numerical evaluation, decode media, judge perceptual quality, or change defaults. Exact-five-second videos are hash-bound to the completed runs' validated artifacts. Weights, activation/latent arrays, executables and private logs remain excluded.

## Selected videos

| Policy | Seed | Exact 5 s video | Source run SHA256 |
|---|---:|---|---|
| qkv-only | 42 | [Play](videos/kernel-qkv-capture-seed42-001.mp4) | `d74e1f87a1ffb6180819cd5190b7806944538a76537714d02ab0c141397c0e88` |
| adaln-h-only | 42 | [Play](videos/kernel-adaln-h-capture-seed42-001.mp4) | `4f042cad8fcf6de84983db168ebfe67d9bd0f21713315c6aca9d8d859306721d` |
| combined | 42 | [Play](videos/kernel-combined-capture-seed42-001.mp4) | `b25c59e78e85e6d0c010ee274b9a4b89bd223f74dacad44313d082f18dbb4dd5` |
| qkv-only | 43 | [Play](videos/kernel-qkv-capture-seed43-001.mp4) | `3bda5568153423910fc430997d6fe4ca01d3ae18a29b2e9507d36bcec5fc265b` |
| adaln-h-only | 43 | [Play](videos/kernel-adaln-h-capture-seed43-001.mp4) | `6e6de69fc7f470883103ee4308ea4869cb8de71cf93b39ca9a72056420c9701d` |
| combined | 43 | [Play](videos/kernel-combined-capture-seed43-001.mp4) | `ef016c8ddd715b336dfe9857d490c7c2ed5d8860ba3b05c69e3cdcb818f60d6b` |

## Default-decision reports

- [completed: keep_experimental](raw/decisions/kernel-default-combined-001.json) — the original report controls its scope and eligibility.

Recorded-hash .py/.c/.h/.patch/.json/license whitelist included for new runs and campaigns.
