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

# EXP-020 — 31.783 WikiText-103 PPL at 49.86M parameters

**49,860,480 trainable parameters**, trained entirely from scratch with seed 42 on **7,199,981,568 prediction tokens**. No pretrained initialization, fine-tuning or distillation. This is a decoder-only base English language model, not an instruction-tuned assistant; it may generate inaccurate, biased or unsafe text and is not suitable for high-stakes advice.

**7.2B tokens · 37.94 h · one RTX 5090 Laptop GPU.** Public exact weights for local CPU inference; tokenizer, config and all required local model modules are included.

**Competition source:** [`gibc-v2-track01-exp020-v1.0.0`](https://github.com/AryaErgin/gibc-v2-foundational-llm/tree/gibc-v2-track01-exp020-v1.0.0), commit `69f56d7b1f4f377f94a30216a9945df8a6662cce`. Later documentation/media improvements do not change the competition model.

**Released model.safetensors SHA-256:**
`4c4f97801ce0c3d0cee52172bfe851b57690b81d153775a33107b8aaed1bc129`.

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

## Install, verify and generate — no source checkout needed

Use Linux/WSL with **Python 3.11**, a new virtual environment and several GiB of free RAM. Download is public and needs no login. The package is about 200 MB; the tested torch wheel installs multi-GB CUDA libraries even though generation below is CPU-only. Allow installation time and disk space. Native Windows/macOS are not qualified by these Linux commands.

Run from a fresh parent directory:

```bash
python3.11 -m venv exp020-inference-venv
source exp020-inference-venv/bin/activate
python -m pip install huggingface_hub==0.34.4
hf download AryaErgin/gibc-v2-exp020 --local-dir exp020-package --exclude .gitattributes
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu132
python -m pip install -r exp020-package/requirements.txt
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
sha256sum exp020-package/model.safetensors
python -B exp020-package/verify.py
python -B exp020-package/generate.py "The small robot moved" --temperature 0 --max-new-tokens 12
```

Compare the SHA output to the independently published weight hash above. Verification checks the complete manifest, 83-tensor structure, strict loading, CPU FP32 finiteness and **49,860,480 trainable parameters**, without a forward pass. Exclude Hub-added `.gitattributes`: it is not part of the strict package inventory. Client `.cache/` metadata is ignored by the verifier. Keep virtual environments outside the package directory.

The final command performs only a short deterministic greedy generation on an innocuous prompt, **not a benchmark**. No hosted inference endpoint or network is needed after installation/download. This custom PyTorch model is **not Transformers AutoModel-compatible**. Its unchanged generation routine crops context to 512 tokens.

[Independent clean-user test and pinned revision](https://github.com/AryaErgin/gibc-v2-foundational-llm/blob/main/docs/submission/PUBLIC_USABILITY.md). The original export remains available at revision `80703cd304d1d1b16abf6539dcb0ebe83386f0ca`; later model-card/checksum updates leave weights and executable code unchanged.

### Official evaluation is a separate, long-running workflow

Fast verification and generation above do **not** call lm-eval, download benchmark datasets or score WikiText. The completed CPU FP32 five-task suite took about **19.63 hours**. [Frozen protocol and evaluator source](https://github.com/AryaErgin/gibc-v2-foundational-llm/blob/gibc-v2-track01-exp020-v1.0.0/docs/EXP020_OFFICIAL_EVALUATION_PROTOCOL.md) document the method. The historical guarded evaluator requires original training-checkpoint/summary/provenance artifacts not all distributed here; this inference-only package is not a drop-in full benchmark replay bundle.

## Provenance, credits and license

Project source: [AryaErgin/gibc-v2-foundational-llm](https://github.com/AryaErgin/gibc-v2-foundational-llm).
See its RESULTS, architecture, data, source ledger and evaluation protocol for complete experimental lineage. The bundled evidence contains no per-example benchmark responses.

Apache-2.0 covers project code and the exported project weights/tokenizer, consistent with the existing project release policy; the full text is in `LICENSE`. External data and libraries retain their own terms. `THIRD_PARTY_NOTICES.md` records primary license evidence, PIQA's unknown license and the WikiText card's version discrepancy. No claim is made to relicense these datasets.

Credit HuggingFaceFW/Common Crawl, benchmark authors, PyTorch, Hugging Face datasets/tokenizers/safetensors, EleutherAI lm-eval, NumPy, PyYAML, Python, NVIDIA hardware/CUDA and Windows/WSL. AI assistance: ChatGPT, Deep Research and OpenAI Codex supported research, explanations, planning, code drafting/review/debugging, experiments, writing and packaging under human scientific authority. These assistants do not constitute or supply pretrained weights for the submitted model.
