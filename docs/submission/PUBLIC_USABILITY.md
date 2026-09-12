# Public-user usability audit — 2026-09-12

## Outcome and scope

Two independent Python 3.11.9 virtual environments passed CPU FP32 release verification and a 12-token greedy generation: one in a fresh public GitHub clone, one using only the anonymously downloaded Hugging Face package. No development package was installed into either environment; no original checkpoint was loaded. Installer wheel caches may be reused: this was an isolated-environment test, not a fresh operating-system VM.

The first temporary test directory disappeared between WSL invocations (`/tmp` is tmpfs); testing restarted in fresh persistent scratch outside the authoritative repository. This infrastructure event did not change the release.

No training, fine-tuning, benchmark requests, benchmark dataset loading or score changes occurred.

## Verified identities

- Competition tag: `gibc-v2-track01-exp020-v1.0.0`.
- Peeled tag/source commit: `69f56d7b1f4f377f94a30216a9945df8a6662cce`.
- Public model: [AryaErgin/gibc-v2-exp020](https://huggingface.co/AryaErgin/gibc-v2-exp020).
- Original tested Hub revision: `80703cd304d1d1b16abf6539dcb0ebe83386f0ca`.
- Weight SHA: `4c4f97801ce0c3d0cee52172bfe851b57690b81d153775a33107b8aaed1bc129`.
- Original checkpoint SHA, retained in provenance: `95338530dcfa660acb149c94d72c07173c14dece6de7da4a22d7aa856723ce89`.
- Exact count: **49,860,480 parameters, 83 FP32 tensors**.
- Tensor names, shapes and dtypes match the original export inventory; strict load and finiteness passed. Bitwise export equivalence is the frozen export's recorded assertion, additionally protected by the unchanged public weight hash; the private source checkpoint was not reread for this public-user test.
- The original 19-file package (17 manifest payloads plus manifest and SHA256SUMS) remains preserved at the original revision.

## Executed clean-user paths

Prerequisite CPython 3.11.9 was already installed on the host; new venvs were created from that base interpreter, not with system-site-packages. Public GitHub source was freshly cloned. Both downloaded packages used fresh local destinations and excluded Hub-added `.gitattributes`.

Install commands tested:

```bash
python3.11 -m venv .judge-venv
source .judge-venv/bin/activate
python -m pip install huggingface_hub==0.34.4
hf download AryaErgin/gibc-v2-exp020 --revision 80703cd304d1d1b16abf6539dcb0ebe83386f0ca --local-dir exp020-package --exclude .gitattributes
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu132
python -m pip install -r exp020-package/requirements.txt
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -B exp020-package/verify.py
python -B exp020-package/generate.py "The small robot moved" --temperature 0 --max-new-tokens 12
python -m pip check
```

The standalone HF test used the same package requirements in its separate venv, with the package as its working directory. Core installed versions: torch 2.13.0+cu132, numpy 2.4.6, PyYAML 6.0.2, tokenizers 0.21.4 and safetensors 0.8.0. The GitHub download CLI was huggingface_hub 0.34.4; standalone package dependencies initially resolved its compatible transitive version 0.36.2. No missing dependency was found; `pip check` passed in both.

For execution, PYTHONPATH/PYTHONHOME were removed; CUDA was hidden; Hugging Face/datasets/Transformers offline flags were enabled. The HF package imported `gibc_llm` from its own bundled directory. An additional Python audit-hook verification rejected network connection attempts and file reads from the development repo or Windows-mounted artifacts; it passed. This is runtime evidence, not a claim of OS-level sandboxing.

| Path | Verify wall time | 12-token generation wall time | Result |
|---|---:|---:|---|
| Standalone public HF download | 5.185 s | 4.250 s | PASS |
| Fresh public GitHub clone + public package | 5.204 s | 4.284 s | PASS |

These short local timings include process/loading overhead and are not model performance benchmarks. Installation/download time is excluded.

Both generated exactly:

> The small robot moved from the ground to the ground, and the robot was

This is an innocuous smoke prompt, not an official benchmark example or quality claim. The weak/repetitive continuation is consistent with a small base LM; it is not edited or cherry-picked.

## Documentation and media corrections

- Root README now leads with achieved PPL, provides clone/install/download/verify/generate commands, and names the immutable competition tag/commit.
- Fast inference checks are separated from official CPU evaluation: frozen suite timestamps imply **19.6305 h**, approximately 20 hours.
- The historical full benchmark runner requires owner-side artifacts not all present in the public inference package; this limitation is explicit rather than hidden behind private paths.
- Seven approved native 1800×1200 PNGs and SVG counterparts replace old README image references, in narrative order. Every copy is byte-identical to its approved Desktop source; the old scientific figures remain untouched.
- Quickstart environments/model downloads are ignored by Git.
- The Hub documentation candidate changes only README.md and its manifest/SHA256SUMS checksums. Every model, tokenizer, config, executable module and scientific evidence payload hash is unchanged. [Model-card mirror](hf-release/README.md).

## Reproduction and known limits

Use the [root quickstart](../../README.md#run-the-public-model--no-gpu-or-benchmark-download). The package needs no source checkout for inference. The qualified commands target Linux/WSL/Python 3.11; native Windows/macOS were not tested. The CUDA-enabled wheel has a multi-GB installation footprint even for CPU-only use. No alternate CPU wheel qualification or dependency redesign was attempted.

[License limitations](LICENSE_AUDIT.md), PIQA exclusion-snapshot equivalence uncertainty, base-LM safety limits and development-history caveats remain unchanged. Video/team/Devpost completion remain human-managed.

The historical receipts and frozen scientific records were not rewritten. Final publication revision and repository checks are recorded in the accompanying [machine-readable receipt](../../results/exp020-public-usability.json).

## Publication and final checks

Documentation-only Hugging Face revision: `c341f396bfb62d263e90d023c319a9dcdf21db4c`. Published README, manifest and SHA256SUMS were independently downloaded and matched the reviewed three-file candidate exactly; manifest SHA `4ade69e49833a4fe16582a7f0bc03715e8013f80a08994eacf91509dc1304363`.

Targeted tests: **20 passed** (`tests/test_inference_release.py`, `tests/test_publication.py`, `tests/test_submission_package.py`). The initial test run had 19 passes and one missing-link failure while this audit document was not yet written; the completed rerun passed all 20. The frozen-evidence/submission verifier and all seven added image/hash/link checks passed.

The optional source parameter counter initially raised `ModuleNotFoundError` in the inference-only environment. Its README command now explicitly uses `PYTHONPATH=src`, and the exact command passed: 5,242,880 embedding + 14,745,600 attention + 29,859,840 MLP + 12,160 norms + 0 additional output-head = **49,860,480**. No source-model code or dependency change was needed.

Final anonymous HTTPS round-trip: **all 19 files / 200,168,172 bytes** at `c341f396bfb62d263e90d023c319a9dcdf21db4c` matched the verified candidate. The public model remained ungated. Only README.md, manifest.json and SHA256SUMS differ from the original release. The same executable/weights payload passed offline generation after the first doc revision and strict verification after the final presentation refinement.
