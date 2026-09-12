# Reproducibility and frozen-artifact map

This is a read-only audit guide, not authorization to rerun training or official evaluation. The authoritative implementation is the local WSL repository `/home/aryae/gibc-v2-foundational-llm`; GitHub source is the publication artifact. Submission edits are uncommitted for human review.

## Source and environment

| Role | Recorded identity |
|---|---|
| Training/preregistration | `d88800733846c7a30e2044fa0a20f9e4a448f328` |
| Data builder | `79a61239bfa216e812fb93c81b082812dfeeca89` |
| Evaluator-containing Git commit | `37332797909df963ca7c77a945cea8752b60d481` (all four raw lm-eval outputs record `3733279`; uniquely resolved to this full SHA) |
| Config | `configs/exp020-final-7p2b-cosine.yaml` |
| Training runtime | Python 3.11.9; torch 2.13.0+cu132; CUDA 13.2; WSL2; one RTX 5090 Laptop GPU |
| Official scoring | CPU FP32; CUDA hidden/unavailable; zero-shot; batch 16; context 512 |
| Evaluation packages | datasets 3.5.1; lm-eval 0.4.9.1; pyarrow 25.0.1; sacrebleu 2.6.0 |

Do not install/downgrade packages to copy historical Windows workarounds. EXP-012 used sacrebleu 1.5.1; the four accuracy tasks do not use its scoring metrics. Frozen raw task definitions are in the evidence digest.

## Artifact map

Let `BASE=/mnt/c/Users/aryae/Desktop/gibc_v2`. Equivalent Windows root: `C:\Users\aryae\Desktop\gibc_v2`.

| Artifact | Path relative to BASE |
|---|---|
| Final checkpoint | `artifacts/exp020-final-7p2b-train-20260905/checkpoints/checkpoint-step-219726.pt` |
| Training summary / metrics / progress | `artifacts/exp020-final-7p2b-train-20260905/{summary.json,metrics.jsonl,progress.jsonl}` |
| Corpus / manifest | `artifacts/exp020-final-7p2b-data-rebuild-2/{train-token-stream.uint16,manifest.json}` |
| Tokenizer | `artifacts/exp020-final-7p2b-data-rebuild-2/tokenizer/tokenizer.json` |
| Internal validation | `artifacts/exp020-final-7p2b-data-rebuild-2/{general,edu}_validation.pt` |
| Four official task JSONs | `artifacts/exp020-official-eval/lm_eval/{hellaswag,arc_easy,piqa,winogrande}.json` |
| Held-out WikiText output | `artifacts/exp020-official-eval/wikitext103.json` |
| Official task sequence status | `artifacts/exp020-official-eval/status/sequence.status.json` |
| Suite status | `logs/exp020-official-eval-suite.status.json` |
| Runtime provenance | `logs/exp020-evaluation-runtime-provenance-20260907.json` |
| Benchmark-free selection preflight | `logs/exp020-official-eval-preflight-20260907.json` |
| Exact historical training launcher | `logs/exp020-final-7p2b-launch-20260905.sh` |

The final manifest abbreviates its build command with `...`; it is not a complete literal replay command. Source code, configuration, source revisions, counts and byte-prefix proof survive. Do not invent omitted command arguments.

## SHA-256 audit manifest

All entries below were read from existing files, not assumed from the prompt. The digest links them to exact local paths.

| Object | Bytes | SHA-256 |
|---|---:|---|
| arc_easy | 6945590 | `67ee8c6fa8615cfee78714c68676a4893634ed3f3923916f9a321ef4fc8ac1e0` |
| checkpoint | 598447091 | `95338530dcfa660acb149c94d72c07173c14dece6de7da4a22d7aa856723ce89` |
| config | 1962 | `25f473f27a3ba673d3e32cb10845e47bf7f4abf0dd47e891db812d6871c3bace` |
| edu_validation | 2099101 | `2d8bf192c0fc08de1a97540eb7b912ed432fd33597067625ed3f7fc23464a181` |
| general_validation | 2099069 | `7029411d75635cf593bbb082650a75161b85d0b398e7714492ee2837d9d5550f` |
| hellaswag | 55243798 | `f2fc1e7025099f4e2a86992d3667bfa104d5ac5b5181243de75fa8fe7d99042c` |
| index | 1019559936 | `4b47a02d0bfa793809b02adcc251eb2f3560217e1ddcc0c595a78906386e7a1f` |
| manifest | 7393 | `aedbbb8dcfe47c5b0d7de2ee052fbeea93232db5a917d09e568c394fe06eacc7` |
| metrics | 185950217 | `ca85d48ba63d895ffea2591ca20e922e7de0e9d9257b8957c8cc8154256937d0` |
| piqa | 3341629 | `054226c2c94b4e6d36ddeaacd94fcfc2c9689387625a5fdf965406faaf85aa41` |
| preflight | 1331 | `d25f0a1fdedf11929deb2ed5abc647ad21caa44d7057e05736d29b7ac147f642` |
| preregistration | 4300 | `a704ccce7b73955c1d8a2b792f39fc05e97e5e3a9b757793e34bf96026ea0bf4` |
| progress | 1544490 | `8906e7525ad8c131cd7b341030d29341d7d63f0d3a410e763834309e958d15c2` |
| runtime | 2212 | `6ca4dfec299e2cfdd316fc64376a48ea4671f44d31534fd9b398a67fd377a4d0` |
| sequence | 1492 | `4993e208695faf90a8b9905a410145c8ebd890f0ee4ddf95bfc72a0716d5c716` |
| stream | 14399963138 | `94e39c09e7696a9668568802c37e6458799b47284efa566f74fb6792f571440e` |
| suite | 1229 | `85acd30d04d7cf51c1af895f6659153d8e963c47be97b4ab477b53564a6b60b3` |
| summary | 5189 | `31c795afce30b7060b58bd8aa5c8ff14aad98ccb79302c450c8f75d48a995d8a` |
| tokenizer | 546551 | `c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14` |
| wikitext103 | 2771 | `3b600a4ed5f65c50f22b4054b427274a3a2cb3e8850f5b76cfae9cc1d43116e9` |
| winogrande | 2014409 | `2d1fc93f634a940bd67cc33a919b6090f9bc4487f08d9df123b6593db56a2835` |

EXP-011 and EXP-012 prefixes also pass their frozen expected hashes; see [DATA_SOURCES](../../DATA_SOURCES.md). Both internal validation tensors are byte-identical to EXP-012.

## Safe commands for judges

From the source checkout, with standard Python:

```bash
python scripts/verify_submission_package.py
```

This checks the committed-style digest, config identity, figure hashes and local Markdown links. It does not need model artifacts, external datasets, packages, GPU or network. It does not itself rehash the external corpus.

For an artifact owner wishing to repeat the read-only audit, the exporter requires a NEW output path and explicit artifact/log roots; `--full-stream` reads 14.4GB plus prefixes and `--checkpoint-tensors` reads tensors on CPU. The completed audit command is recorded in [VALIDATION.md](VALIDATION.md). Do not write into an existing frozen record.

## Reproduction boundaries and delivery

The code for training, exact parameter counting, tokenizer, builder and guarded CPU evaluators remains intact. This package does not execute those workflows. See [architecture](../../ARCHITECTURE.md), [data](../../DATA_SOURCES.md), [protocol](../EXP020_OFFICIAL_EVALUATION_PROTOCOL.md), [credits](../../SOURCE_LEDGER.md).

The exact inference-only [safetensors package is public](https://huggingface.co/AryaErgin/gibc-v2-exp020). [Pinned release instructions](EXP020_RELEASE.md) and [anonymous remote hash evidence](PUBLIC_RELEASE.md) bind it to revision `80703cd304d1d1b16abf6539dcb0ebe83386f0ca` and the unchanged frozen checkpoint. The separate [clean-environment smoke](PREPUBLICATION_SMOKE.md) passed. Raw training data/checkpoints remain outside Git; final source publication still requires human authorization.

The [scoped license audit](LICENSE_AUDIT.md) covers core datasets/dependencies and recorded code reuse. PIQA licensing/snapshot equivalence and the WikiText card's licensing-version discrepancy remain explicit. Exploratory bibliography entries labeled VERIFY remain historical research notes, not verified final-recipe claims.
