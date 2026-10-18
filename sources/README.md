# Pinned model facts

These JSON configuration snapshots were retrieved from the official MiniMaxAI/MiniMax-H3 repository at the immutable revision in manifest.json. The snapshots retain upstream terms; the repository's MIT license covers our original harness and kernels, not the MiniMax model release. No weights are included.

The source implementation used to verify projection semantics is Hugging Face Diffusers at `b800f828f3951c4d206e154411bcbfa693cd322e`, `src/diffusers/models/transformers/transformer_minimax_h3.py`. Config URLs and SHA256 digests are in manifest.json.

Primary references:

- [H3 official model card](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [Diffusers transformer implementation](https://github.com/huggingface/diffusers/blob/b800f828f3951c4d206e154411bcbfa693cd322e/src/diffusers/models/transformers/transformer_minimax_h3.py)
- [Apple MPP programming guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf)
- [Apple WWDC26 TensorOps and quantization](https://developer.apple.com/videos/play/wwdc2026/330/)
- [OCP MX specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [Draw Things quantized attention and INT8 results](https://releases.drawthings.ai/p/metal-quantized-attention-pulling)
- [Draw Things MFA source](https://github.com/liuliu/ccv/tree/unstable/lib/nnc/mfa)

External performance claims are context, not local measurements. The current harness measures installed MLX and our direct TensorOps kernels. An apples-to-apples pinned MFA integration remains a separate benchmark target.
