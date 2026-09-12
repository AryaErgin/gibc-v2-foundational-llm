# EXP-020 public release — verified 2026-09-09

**Permanent model URL: [AryaErgin/gibc-v2-exp020](https://huggingface.co/AryaErgin/gibc-v2-exp020).**

Public and ungated. Immutable verified Hub revision: `80703cd304d1d1b16abf6539dcb0ebe83386f0ca`.

## Evidence

The user reported a successful remote round-trip download matching the released safetensors hash. During final repository reconciliation, a separate anonymous HTTPS check resolved the Hub revision and independently streamed **all 19 release files / 200,166,267 bytes** through SHA-256. Every byte count and digest matched the unchanged local, smoke-tested package. No credentials, local download cache, model loading, inference or benchmark requests were used for that remote check.

[Machine-readable public receipt](../../results/exp020-public-release.json) · [Original export receipt](../../results/exp020-inference-release.json) · [Isolated judge smoke](PREPUBLICATION_SMOKE.md).

| Identity | SHA-256 |
| --- | --- |
| Released model.safetensors | `4c4f97801ce0c3d0cee52172bfe851b57690b81d153775a33107b8aaed1bc129` |
| Original frozen training checkpoint | `95338530dcfa660acb149c94d72c07173c14dece6de7da4a22d7aa856723ce89` |
| Package manifest | `b3f9cdadce6757d7134834303b76c254caa3433c667fb2f191e4c21375c2fe2e` |
| Frozen config | `25f473f27a3ba673d3e32cb10845e47bf7f4abf0dd47e891db812d6871c3bace` |
| Tokenizer | `c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14` |

The original 598,447,091-byte checkpoint is not the 199,450,256-byte released weights file. The latter is its exact FP32 inference-only model-state export: 83 tensors and 49,860,480 trainable parameters. Their distinct hashes must not be confused.

## Judge use and provenance boundary

Follow the [pinned download/install/verification instructions](EXP020_RELEASE.md). Exclude Hub-added `.gitattributes` when downloading the strict 19-file package; existing verifier semantics already ignore client cache metadata. No Transformers AutoModel support or hosted inference endpoint is claimed: this is public weight delivery for local custom-PyTorch inference.

The packaged README and evidence remain frozen snapshots of the original export. Current source-repository documentation records subsequent smoke and publication events; it does not silently rewrite package manifests or historical no-inference observations. Public delivery changes no benchmark score, selected checkpoint, model weight, scientific gate or license limitation.

Model delivery is complete. Final source commit/tag/push, demo video, team/eligibility and Devpost completion remain separate human actions. No upload, source publication or model execution was performed in this reconciliation.
