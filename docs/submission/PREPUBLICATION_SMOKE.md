# EXP-020 pre-publication judge smoke — 2026-09-08

> Historical smoke receipt. [Public delivery completed subsequently](PUBLIC_RELEASE.md); the observations and package hashes below are preserved.

## Verdict

**PASS on the tested Linux/WSL CPU setup.** A fresh environment installed the documented dependencies, verified and loaded the copied public package, and executed exactly one deterministic, non-benchmark generation without access to the development repository, original checkpoint, Windows artifacts, old caches, or network resources after installation.

Release candidate: `artifacts/exp020-public-inference-20260908-release`.

No correction to the release package was required. All 19 candidate files and their copied counterparts remained byte-identical after testing. No training, benchmark evaluation, weight modification, staging, commit, push, or upload occurred.

[Machine-readable evidence and installed package inventory](../../results/exp020-prepublication-smoke.json).

## Clean setup and commands

Temporary location: `/home/aryae/exp020-judge-smoke-0jqw3gqm`.

The package was copied into its `package/` directory. A fresh virtual environment was created outside the development environment. The host's default Python is 3.14.4, so the declared Python 3.11 prerequisite was supplied by copying the existing standalone CPython 3.11.9 interpreter/standard library into `python311/`; development site-packages were not reused. This was not a new operating-system installation.

From the copied package directory, with Python 3.11 on PATH:

```sh
python3.11 -m venv ../exp020-inference-venv
# The harness placed this new venv/bin first on PATH (activation equivalent).
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu132
python -m pip install -r requirements.txt
python -m pip check
python -m pip freeze --all
```

All commands exited 0. Installed versions included Python 3.11.9, torch 2.13.0+cu132, NumPy 2.4.6, PyYAML 6.0.2, tokenizers 0.21.4 and safetensors 0.8.0. The full transitive inventory is retained in the JSON record. CUDA/Triton wheel dependencies were installed by pip as part of the documented PyTorch installation; execution was CPU-only and did not compile anything.

No `datasets` or `lm_eval` package was installed. CUDA was hidden and unavailable. User site-packages and inherited PYTHONPATH were excluded; private cache directories started empty.

After installation, commands ran inside an existing `bwrap` network/mount namespace: no network route, only loopback, no repository or `/mnt/c` mount, read-only copied package, and ordinary system libraries plus the clean environment. Offline Hugging Face flags were also set. The exact isolation arguments are in the evidence JSON. This rules out hidden original-path/cache/network dependencies for the exercised loading and generation path, not every possible platform or optional code path.

## Public command and observed output

The unmodified documented package entry point was used:

```sh
CUDA_VISIBLE_DEVICES="" python -B verify.py
CUDA_VISIBLE_DEVICES="" python -B generate.py "The small robot moved" --temperature 0 --max-new-tokens 12
CUDA_VISIBLE_DEVICES="" python -B verify.py
```

Exactly one generation command ran. Verbatim stdout:

```text
The small robot moved from the ground to the ground, and the robot was
```

Exit code 0; 4.166 seconds for the entire generation command, including package verification and model loading. This is an operational smoke output, not a benchmark, quality claim, or pure decoding-speed measurement. The output was not selected from repeated attempts.

## Integrity and loading

| Check | Result |
| --- | --- |
| Trainable parameters | 49,860,480 exactly |
| Tensor inventory | 83 tensors; 49,860,480 stored elements |
| Parameters | CPU FP32, finite |
| Tokenizer | Loaded from package; vocabulary 8192 |
| Config/model | Loaded from package; strict state load passed |
| Manifest | PASS before and after generation |
| Export equivalence | Tensor names, shapes, dtypes, values and raw bits exactly equal |
| Candidate/copy immutability | All 19 files unchanged |

The original checkpoint was separately read by a custodian audit outside the judge sandbox solely to recompute export equality. It was not accessible to the public loading/generation commands. That audit ran no model forward pass and confirmed source step 219,726, prediction tokens 7,199,981,568 and next sequence index 14,062,464.

| Identity | SHA-256 |
| --- | --- |
| Released safetensors (199,450,256 bytes) | `4c4f97801ce0c3d0cee52172bfe851b57690b81d153775a33107b8aaed1bc129` |
| Source checkpoint (598,447,091 bytes) | `95338530dcfa660acb149c94d72c07173c14dece6de7da4a22d7aa856723ce89` |
| Release manifest | `b3f9cdadce6757d7134834303b76c254caa3433c667fb2f191e4c21375c2fe2e` |
| Config | `25f473f27a3ba673d3e32cb10845e47bf7f4abf0dd47e891db812d6871c3bace` |
| Tokenizer | `c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14` |

The 19-file package totals 200,166,267 bytes (approximately 190.89 MiB). Model behavior was not altered or quantized.

## Repository checks

Executed after the isolated generation in the authoritative repository:

```sh
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests/test_inference_release.py tests/test_submission_package.py tests/test_submission_contract.py -q
.venv/bin/python scripts/verify_submission_package.py
git diff --check
```

- Focused release/submission tests: **21 passed in 6.99 seconds**.
- Submission verifier: PASS for evidence, config identity, local Markdown links, and four PNG/SVG figure pairs/provenance.
- Whitespace check: PASS.
- Repository HEAD: `37332797909df963ca7c77a945cea8752b60d481`.
- Existing submission edits were preserved, not staged. The known untracked egg-info and stale Windows checkout were untouched.

This task adds this report and its JSON evidence only; it does not replace earlier packaging-time no-inference records, which describe an earlier phase. Public upload and final submission actions remain separate manual decisions. Fresh-user usability is verified for the documented Python 3.11 Linux/WSL installation, not asserted for untested operating systems.
