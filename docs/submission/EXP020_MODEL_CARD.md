---
language: en
license: apache-2.0
library_name: pytorch
tags:
- gibc-v2
- from-scratch
- causal-lm
- research
---

# GIBC V2 EXP-020 — frozen 49.86M base language model

Public model: [AryaErgin/gibc-v2-exp020](https://huggingface.co/AryaErgin/gibc-v2-exp020). Immutable verified Hub revision: `80703cd304d1d1b16abf6539dcb0ebe83386f0ca`. [Public release receipt](PUBLIC_RELEASE.md).

This is the current source-repository model documentation. The original packaged README remains byte-preserved as a release-time snapshot.

**49,860,480 trainable parameters**, trained entirely from scratch with seed 42 on **7,199,981,568 prediction tokens**. No pretrained initialization, fine-tuning or distillation. This is a base English language model, not an instruction-tuned assistant; it may generate inaccurate, biased or unsafe text and is not suitable for high-stakes advice.

## Identity and selection

Original frozen training checkpoint SHA-256:
`95338530dcfa660acb149c94d72c07173c14dece6de7da4a22d7aa856723ce89`
(598,447,091 bytes, step 219,726).

The release is an exact CPU FP32 safetensors export of its model state only. No quantization, pruning, tensor changes, optimizer, scheduler, training RNG or resume state. Every tensor name, shape, dtype, numerical value and raw bit pattern is verified against that checkpoint. EXP-020 stores one `token_embedding.weight` key, used directly for the output projection. There are exactly 83 tensors and 49,860,480 stored elements; no separate output-head tensor is added.

`provenance.json` records the released-weight SHA, exact parameter inventory, source/checkpoint/config/tokenizer hashes, tool hashes and verification. `SHA256SUMS` and `manifest.json` cover the package. Compare their hashes with the source repository's release receipt, not an untrusted download alone.

Training source: `d88800733846c7a30e2044fa0a20f9e4a448f328`.
Bundled architecture/loader/generation source and official-evaluation source: `37332797909df963ca7c77a945cea8752b60d481`.
Official raw lm-eval artifacts record `3733279`; the full hash resolves that recorded abbreviation. New packaging tools are identified by file hashes; they are not falsely claimed to exist in that older commit.

Checkpoint selection used frozen General/Edu internal validation before EXP-020 official benchmark scoring. Benchmark results cannot authorize further training or checkpoint selection.

## Model and training

8,192 vocabulary; width 640; nine blocks; twenty heads, head dimension 32; SwiGLU FFN 1,728; pre-RMSNorm; RoPE; context 512; tied embedding/output; no bias/dropout. QK-Norm and CWD are OFF. Ordinary AdamW, full-horizon cosine, microbatch 32, accumulation 2, BF16 forward/FP32 state.

Training used one NVIDIA RTX 5090 Laptop GPU, WSL, 136,589.44783848198 seconds (**37.94 h**) including fixed 0.300-second operational pacing. Approximate theoretical 6NT training compute: **2,153,967,221,829,795,840 FLOPs (2.154 EFLOPs)**. This is not measured energy or exact executed hardware FLOPs.

Training data: deterministic whole-document 2:1 FineWeb/FineWeb-Edu, global exact-document deduplication and frozen normalized 13-gram contamination screening; frozen tokenizer trained from scratch. Exact source revisions, accounting and validation/stream hashes are in `training-evaluation-evidence.json`. No training or benchmark dataset is distributed.

## Frozen official results

| Task | Metric | Result |
|---|---|---:|
| HellaSwag | acc_norm | 30.163314% |
| ARC-Easy | acc_norm | 39.141414% |
| PIQA | acc_norm | 60.446137% |
| WinoGrande | acc | 48.776638% |
| WikiText-103 held-out | token perplexity | 31.783151406614728 |

CPU FP32, zero-shot, batch 16, context 512, lm-eval 0.4.9.1. WikiText uses the separate frozen rolling evaluator; mean NLL 3.4589363193959604, BPB 1.3600661174573045. Runtime: Python 3.11.9, torch 2.13.0+cu132, datasets 3.5.1, pyarrow 25.0.1, sacrebleu 2.6.0; no sacrebleu scoring metrics in the four harness tasks.

HellaSwag/ARC improved against EXP-012, PIQA was roughly stable, WinoGrande regressed. No universal reasoning or significance claim. Exact PIQA `baber/piqa` versus historical exclusion-source snapshot equivalence is **not proved**; equal task name/split size does not prove equal content. Exact n-gram exclusion is not comprehensive semantic decontamination.

## Install, verify and use (manual commands)

Python 3.11 is required. No GPU is required for loading or CPU generation. Allow several GiB of free RAM for installation/loading and generation; no new inference-memory measurement is claimed here. A separate [clean-environment Linux/WSL smoke](PREPUBLICATION_SMOKE.md) passed dependency installation, offline loading and one deterministic generation; no inference-memory measurement or portability claim for untested platforms is added.

The existing qualified runtime was torch 2.13.0+cu132. Install that official wheel plus the small pinned package dependencies in a new environment outside the package directory, not by changing a qualified evaluation environment:

```bash
python3.11 -m venv ../exp020-inference-venv
source ../exp020-inference-venv/bin/activate
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu132
python -m pip install -r requirements.txt
CUDA_VISIBLE_DEVICES="" python -B verify.py
```

The package is a custom PyTorch architecture, **not** a Transformers AutoModel checkpoint. Its verifier checks hashes, tensor inventory, strict CPU FP32 loading, finiteness and exact parameters without a forward pass. To generate locally after verification:

```bash
CUDA_VISIBLE_DEVICES="" python -B generate.py "The purpose of scientific measurement is" --max-new-tokens 64
```

That command is for optional manual use. The separately authorized smoke used an innocuous non-benchmark prompt; final reconciliation did not generate another example. Greedy decoding is the default; context is cropped to 512 tokens by the unchanged project generation routine. No hosted service or network access is needed after installation/download.

## Provenance, credits and license

Project source: [AryaErgin/gibc-v2-foundational-llm](https://github.com/AryaErgin/gibc-v2-foundational-llm).
See its RESULTS, architecture, data, source ledger and evaluation protocol for complete experimental lineage. The bundled evidence contains no per-example benchmark responses.

Apache-2.0 covers project code and the exported project weights/tokenizer, consistent with the existing project release policy; the full text is in `LICENSE`. External data and libraries retain their own terms. `THIRD_PARTY_NOTICES.md` records primary license evidence, PIQA's unknown license and the WikiText card's version discrepancy. No claim is made to relicense these datasets.

Credit HuggingFaceFW/Common Crawl, benchmark authors, PyTorch, Hugging Face datasets/tokenizers/safetensors, EleutherAI lm-eval, NumPy, PyYAML, Python, NVIDIA hardware/CUDA and Windows/WSL. AI assistance: ChatGPT, Deep Research and OpenAI Codex supported research, explanations, planning, code drafting/review/debugging, experiments, writing and packaging under human scientific authority. These assistants do not constitute or supply pretrained weights for the submitted model.
