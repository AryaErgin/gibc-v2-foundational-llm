# EXP-020 official evaluation protocol

EXP-020's terminal 7.2B checkpoint was selected entirely from frozen General/Edu validation before its own official required-benchmark scoring (historical models had been evaluated earlier). Official benchmarks are final reporting only: no result may authorize further model selection, checkpoint selection, training, or hyperparameter changes.

## Frozen identity

- Checkpoint: `checkpoint-step-219726.pt`, SHA-256 `95338530dcfa660acb149c94d72c07173c14dece6de7da4a22d7aa856723ce89`, 598,447,091 bytes.
- Config: `configs/exp020-final-7p2b-cosine.yaml`, SHA-256 `25f473f27a3ba673d3e32cb10845e47bf7f4abf0dd47e891db812d6871c3bace`.
- Tokenizer SHA-256: `c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14`.
- Summary SHA-256: `31c795afce30b7060b58bd8aa5c8ff14aad98ccb79302c450c8f75d48a995d8a`.
- Training source commit: `d88800733846c7a30e2044fa0a20f9e4a448f328`.
- Terminal selection: step 219,726; 7,199,981,568 prediction tokens; 49,860,480 trainable parameters.

## Frozen CPU runtime

Every official task runs CPU FP32, zero-shot, batch size 16, context length 512, with `CUDA_VISIBLE_DEVICES=""` and CUDA unavailable. Runtime provenance freezes Python 3.11.9, torch 2.13.0+cu132, datasets 3.5.1, lm-eval 0.4.9.1, pyarrow 25.0.1, and sacrebleu 2.6.0. EXP-012 historically used sacrebleu 1.5.1; no downgrade is permitted because HellaSwag, ARC-Easy, PIQA, and WinoGrande use only accuracy/normalized-accuracy metrics, not sacrebleu-derived metrics.

## Completed execution and frozen outputs

The completed guarded suite ran, in order: HellaSwag, ARC-Easy, PIQA, WinoGrande, then held-out WikiText-103. WikiText uses `wikitext/wikitext-103-raw-v1`, `test`, revision `b08601e04326c79dfdd32d625aee71d232d685c3`; each document receives one EOD prefix, no added BOS/EOS, rolling context length 1, maximum sequence length 512, and every document token is scored exactly once.

Result directory: `artifacts/exp020-official-eval/`.

- `lm_eval/hellaswag.json`, `lm_eval/arc_easy.json`, `lm_eval/piqa.json`, `lm_eval/winogrande.json`
- `wikitext103.json`
- `status/sequence.status.json` and per-task status files; outer suite status is in the root `logs/` directory as described below
- `logs/*.stdout.log` and `logs/*.stderr.log`

The outer suite guard writes its own atomic status and logs under `logs/exp020-official-eval-suite.*`, outside the fresh result directory; the inner sequence owns the result-directory status files. This prevents the outer guard from being mistaken for an existing result artifact. Any existing nonempty official output directory or conflicting evaluator process aborts execution before a result can be overwritten or mixed.

## Completion — 2026-09-08

All five official outputs are frozen. No rerun or model/checkpoint change is authorized. CPU FP32 results: HellaSwag acc_norm 30.163314%, ARC-Easy acc_norm 39.141414%, PIQA acc_norm 60.446137%, WinoGrande acc 48.776638%; WikiText-103 token PPL 31.783151406614728. [RESULTS](../RESULTS.md) has uncertainty and comparisons; [audit digest](../results/exp020-submission-evidence.json) binds exact artifact hashes and runtime/task metadata.

The actual outer suite status is `logs/exp020-official-eval-suite.status.json`; sequence/task statuses are inside the result directory. The four raw lm-eval artifacts record `raw_lm_eval_result.git_hash = 3733279`, resolving uniquely to `37332797909df963ca7c77a945cea8752b60d481`. The full SHA expands recorded execution provenance, not merely the presence of evaluator code in Git. Dataset revisions used to build the exclusion index must not be presented as verified identities for every officially loaded task snapshot. See [audit limits](submission/EVIDENCE_AUDIT.md).
