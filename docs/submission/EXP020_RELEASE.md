# EXP-020 inference release

This is an inference-only copy of the immutable step-219726 model, not a new model or training checkpoint. The original checkpoint is never modified. [Model-card template](EXP020_MODEL_CARD.md) and [license audit](LICENSE_AUDIT.md) are included in the package.

## Public package and immutable revision

**[Download EXP-020 from Hugging Face](https://huggingface.co/AryaErgin/gibc-v2-exp020)**. Current documentation-polished public revision: `c341f396bfb62d263e90d023c319a9dcdf21db4c`. The original byte-frozen package remains at `80703cd304d1d1b16abf6539dcb0ebe83386f0ca`; only README and its checksums differ. [Latest clean-user audit](PUBLIC_USABILITY.md). [Release evidence](PUBLIC_RELEASE.md) records the user-reported round trip and an independent anonymous streamed hash comparison of all 19 package files.

### Preserved local release copy

Release directory in the authoritative WSL tree:

`/home/aryae/gibc-v2-foundational-llm/artifacts/exp020-public-inference-20260908-release`

Verified package size: **200,166,267 bytes** (weights: 199,450,256 bytes). [Exact release hashes and file inventory](../../results/exp020-inference-release.json) · [Executed verification and changed-file list](COMPLIANCE_VALIDATION.md).

This directory is ignored by Git. Do not stage weights/data. It remains byte-identical to the smoke-tested package and the 19 corresponding public files at the original `80703cd...` revision, not the later README/checksum revision. The repository's evolving documentation does not rewrite this frozen package.

## Exact content / no-inference verification

- `model.safetensors`: exact finite CPU FP32 tensor values; every original state_dict key retained.
- Frozen `config.yaml` and `tokenizer.json`, byte-identical to the selected model's inputs.
- Four exact project model/config/generation source modules from commit `37332797909df963ca7c77a945cea8752b60d481`, plus isolated release verifier.
- `verify.py`, manual `generate.py`, minimal dependency pins and model card.
- `provenance.json`: bitwise tensor equality, strict CPU loading, parameter inventory, original/released SHA values and new uncommitted export-tool hashes.
- Safe aggregate training/evaluation evidence, evaluator-source supplement, Apache-2.0 LICENSE and third-party notices.
- `manifest.json` and `SHA256SUMS`: exact payload inventory, bytes and hashes. These provide integrity, not a cryptographic signature; compare with the source repository's independently published release receipt.

The original state has 83 tensors and 49,860,480 stored elements. `token_embedding.weight` is used directly for output projection; no separate output-head tensor or duplicate embedding is introduced. No optimizer/scheduler/RNG training state, datasets, benchmark response examples, caches or credentials are released.

Source checkpoint SHA:
`95338530dcfa660acb149c94d72c07173c14dece6de7da4a22d7aa856723ce89`

CPU loading and exact tensor comparisons do not call `forward`, perform inference or rerun benchmarks. The historical EXP-012 exporter is unchanged; it was not invoked because its forward-equivalence check would violate this task's no-inference boundary. New tooling preserves torch RNG during strict-load verification.

## Reproduce the export (artifact owner only)

The following command already refuses an existing output directory. A reproduction requires a different NEW path, never overwriting the prepared release:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
.venv/bin/python scripts/build_exp020_publication.py \
  --checkpoint /mnt/c/Users/aryae/Desktop/gibc_v2/artifacts/exp020-final-7p2b-train-20260905/checkpoints/checkpoint-step-219726.pt \
  --tokenizer /mnt/c/Users/aryae/Desktop/gibc_v2/artifacts/exp020-final-7p2b-data-rebuild-2/tokenizer/tokenizer.json \
  --output /home/aryae/gibc-v2-foundational-llm/artifacts/exp020-public-inference-20260908-release
```

## Judge verification / use

Use the copy-and-paste [root quickstart](../../README.md#run-the-public-model--no-gpu-or-benchmark-download), or the standalone package README, for Python 3.11 installation/download/verification/generation. The [isolated judge smoke](PREPUBLICATION_SMOKE.md) passed installation, offline verification/loading and exactly one deterministic generation. With the package available:

```bash
CUDA_VISIBLE_DEVICES="" python -B /path/to/package/verify.py
# Optional manual local inference; not part of repository verification:
CUDA_VISIBLE_DEVICES="" python -B /path/to/package/generate.py "The purpose of scientific measurement is" --max-new-tokens 64
```

The generic source-repository `scripts/generate.py` also supports safetensors with explicit `--config`, `--checkpoint`, `--tokenizer`, `--device cpu` arguments. No Transformers AutoModel support is claimed. The frozen official evaluator protocol is documented separately; loading this inference package does not execute it.

## Pinned public download (no account required)

In an environment with the [Hub CLI](https://huggingface.co/docs/huggingface_hub/guides/cli), choose a fresh destination:

```bash
hf download AryaErgin/gibc-v2-exp020 \
  --revision c341f396bfb62d263e90d023c319a9dcdf21db4c \
  --local-dir ./exp020-package \
  --exclude .gitattributes
cd exp020-package
# After installing the dependencies from the package README:
CUDA_VISIBLE_DEVICES="" python -B verify.py
```

Released `model.safetensors` SHA-256:
`4c4f97801ce0c3d0cee52172bfe851b57690b81d153775a33107b8aaed1bc129`.
Compare all payload hashes with the [source release receipt](../../results/exp020-inference-release.json) and [public receipt](../../results/exp020-public-release.json). The command excludes Hub-added `.gitattributes`, which the exact-inventory verifier would otherwise reject. Client `.cache/` metadata is already excluded by the frozen verifier; neither is part of the 19-file package manifest. Do not overwrite the owner's original release directory with a download.

Public model delivery is complete. Competition source is published at tag `gibc-v2-track01-exp020-v1.0.0` / `69f56d7b1f4f377f94a30216a9945df8a6662cce`. Devpost/video/team actions remain user-managed; see [compliance checklist](COMPLIANCE.md). No upload is needed in this reconciliation.
