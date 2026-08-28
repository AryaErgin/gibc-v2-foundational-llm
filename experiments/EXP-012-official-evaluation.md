# EXP-012 Official Evaluation Protocol — Frozen Before Benchmark Execution

## Selection and authority

Only `artifacts/exp012-full/checkpoints/checkpoint-step-73242.pt` is eligible. Its SHA-256 is `cacb728b3963c10af8f4613149d8b879b0ef6e44558069726c621c1cb1bb981c`; it is selected before observing any required benchmark score because its frozen combined validation loss `3.0204539299` is the lowest observed. No alternate checkpoint, architecture, tokenizer, data, LR, or training horizon may be selected or changed using evaluation results.

The current GIBC V2 Devpost Rules page was checked on 2026-08-27. Track 01 requires a <=50M from-scratch model and reports HellaSwag, ARC-Easy, PIQA, WinoGrande through lm-evaluation-harness plus perplexity on a held-out WikiText-103 slice. It also requires the evaluation script and results in the README.

## Common model gate

- Config: `configs/exp012.yaml`; loaded checkpoint recount: exactly 49,860,480 trainable parameters.
- Frozen tokenizer SHA-256: `c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14`; model context 512; causal decoder-only adapter; CUDA BF16 autocast model path with FP32 checkpoint tensors.
- No pretrained weights are loaded: evaluation constructs only the configured random-initialized architecture then loads the selected local checkpoint.
- Before execution: selected SHA, configuration/tokenizer hashes, adapter-vs-direct logits, causal shift, and max-context rolling behavior must pass tests. Any mismatch stops all evaluation.

## lm-evaluation-harness tasks

- Package: `lm-eval==0.4.9.1` from `pyproject.toml`; package version and installed source metadata are recorded in the raw artifact.
- Tasks, unchanged harness templates: `hellaswag`, `arc_easy`, `piqa`, `winogrande`.
- Full available examples once per task; zero-shot (`num_fewshot=0`); batch size 16; seed 42; CUDA device; model max context 512; `CustomCausalLM` TemplateLM adapter.
- Report unchanged harness metrics: HellaSwag `acc,none`, `acc_norm,none`; ARC-Easy `acc,none`, `acc_norm,none`; PIQA `acc,none`, `acc_norm,none`; WinoGrande `acc,none` (plus standard errors where emitted).
- Raw result files are one JSON record per task in `artifacts/exp012-official-eval/lm_eval/`, preserving the unmodified lm-eval result, command, output SHA-256, checkpoint/tokenizer hashes, package metadata, runtime, and task provenance.

## WikiText-103 held-out perplexity

This is a separate evaluator. The lm-eval task name `wikitext` is prohibited here because upstream lm-eval 0.4.9.1 defines it using `wikitext-2-raw-v1`, not WikiText-103.

- Dataset: Hugging Face `wikitext`, revision `b08601e04326c79dfdd32d625aee71d232d685c3`, config `wikitext-103-raw-v1`, held-out `test` split.
- Deterministic slice: every nonempty document in the pinned test split, in dataset order; no shuffle, sampling, filtering, or benchmark-driven selection.
- Tokenization: exact frozen tokenizer above; no tokenizer-added BOS/EOS. Each document's first scored token is conditioned on exactly one `<|endoftext|>` prefix token; document boundaries are not concatenated.
- Windows: lm-eval-compatible `get_rolling_token_windows(token_list, prefix_token=EOD, max_seq_len=512, context_len=1)` followed by `make_disjoint_window`. Each document token is scored exactly once, retaining up to 512 preceding causal tokens; stride is therefore the disjoint continuation length emitted by those windows, not a separately tuned constant.
- Scored tokens: all tokenizer-produced IDs for every included document; no prefix/EOD token is included in the scored-token count.
- Aggregation: `mean_nll = -sum(log p(token_i | causal_context_i)) / scored_tokens`; `perplexity = exp(mean_nll)`. `bits_per_byte = -sum(log p)/ln(2)/UTF-8-byte-count` is secondary only. No lm-eval WikiText-2 metric is reported or compared.
- Raw artifact: JSON includes pinned dataset/config/split/revision, exact document/token/UTF-8-byte counts, summed negative log likelihood, PPL, optional BPB, checkpoint/tokenizer/config hashes, command, environment/package versions, wall time, and SHA-256.

## Commands

```text
CUDA_VISIBLE_DEVICES="" python scripts/run_evaluation_guarded.py \
  --task exp012_official_cpu_suite \
  --status artifacts/exp012-official-eval/status/suite.status.json \
  --stdout artifacts/exp012-official-eval/logs/suite.stdout.log \
  --stderr artifacts/exp012-official-eval/logs/suite.stderr.log -- \
  python scripts/run_exp012_cpu_official_sequence.py \
  --checkpoint artifacts/exp012-eval-checkpoint/checkpoint-step-73242.pt \
  --tokenizer artifacts/exp012-eval-checkpoint/tokenizer/tokenizer.json \
  --output-dir artifacts/exp012-official-eval
```

This sole guarded command executes the frozen task order HellaSwag,
ARC-Easy, PIQA, WinoGrande, then the separate WikiText-103 evaluator. Each
task has a task-local PID/status artifact and logs; the suite status reaches a
terminal state only after the actual child sequence exits. These commands
execute only after the frozen pre-evaluation gate command passes. No benchmark
result authorizes further training.

## EVALUATION-ENVIRONMENT-COMPATIBILITY amendment

Before any benchmark request, the first HellaSwag invocation failed while importing `lm_eval.api.metrics`: `sacrebleu 2.6.0` imported `lxml.etree`, whose native DLL was blocked by Windows Smart App Control. No lm-eval task request, dataset example, or benchmark metric executed, and no raw result artifact was written.

`lm-eval` remains exactly `0.4.9.1`. Its installed metadata permits `sacrebleu>=1.5.0`. The required packaged task definitions are unchanged accuracy-only multiple-choice tasks: HellaSwag `acc`/`acc_norm`, ARC-Easy `acc`/`acc_norm`, PIQA `acc`/`acc_norm`, and WinoGrande `acc`. They define no BLEU, chrF, TER, or other sacrebleu metric. The smallest compatible environment change is therefore to pin only `sacrebleu==1.5.1`, which has no `lxml` runtime dependency; no lm-eval source, task YAML/template, checkpoint, model, tokenizer, or protocol setting changes.

Before resuming evaluation, record the old/new package versions, run `pip check`, and require clean `import sacrebleu`, `import lm_eval`, and `from lm_eval import evaluator` smoke tests. Re-run all selected-checkpoint, parameter, tokenizer/config, adapter, causal-shift, context, and unit-test gates. If any import or gate fails, stop without relaxing Windows security or executing a benchmark request.

## EVALUATION-INCIDENT amendment

The first post-recovery HellaSwag attempt began under OMEN Performance mode at approximately 159 W but was interrupted by severe display corruption that required a forced reboot. No final HellaSwag JSON artifact or score was produced or observed. It is therefore not an evaluation result and influenced no model, checkpoint, or protocol decision. This record does not infer that Performance mode caused the incident.

The selected checkpoint SHA, model/task settings, zero-shot protocol, batch size 16, CUDA BF16 evaluation path, and context 512 remain frozen. The sole systems-control change for the restart is OMEN **Balanced** mode as a hardware-stability control; evaluation semantics are unchanged. If display corruption, a CUDA error, driver reset, or other abnormal system behavior recurs, stop without another benchmark attempt or security/system-settings change.

## WSL evaluation-environment amendment

Native Windows evaluation was abandoned only after Python-native dependencies could not import under Smart App Control: the initial lm-eval import chain reached `sacrebleu 2.6.0 -> lxml.etree`, and the subsequent `sacrebleu==1.5.1` retry reached `evaluate -> datasets -> pyarrow.lib`. Smart App Control was never disabled. The earlier HellaSwag attempt was interrupted by display corruption and produced no completed result; the later native retry did not execute a benchmark request.

WSL2 Ubuntu 26.04 LTS is the isolated evaluation environment, with the public repository checked out at `c2d3a0a9aeef7ed23633529a5143c24d2961e87f` before this amendment. It uses Python `3.11.16`, `torch 2.13.0+cu132` (CUDA runtime `13.2`), `datasets 3.5.1`, `pyarrow 25.0.1`, `lm-eval 0.4.9.1`, and `sacrebleu 1.5.1`. `pip check`, all required imports, CUDA/BF16 visibility, checkpoint/tokenizer hashes, parameter recount, finite tensor checks, actual checkpoint adapter-vs-direct CPU FP32 scoring (absolute tolerance `1e-6`), causal/rolling tests, and the 86-test project suite passed. The WSL differential CUDA test produced no new DXG messages, CUDA error, or observed display corruption; no bitwise Windows/Linux BF16 equality is claimed or required.

The checkpoint, task definitions/templates, zero-shot setting, batch size, context, and scoring semantics are unchanged. No training and no benchmark evaluation ran in WSL during setup; benchmark execution remains separately gated.

## CPU FP32 execution amendment — authorized before official scoring

Research review rejected every local CUDA path for this official evaluation:

- a native-Windows real HellaSwag workload was interrupted by severe display
  corruption requiring forced shutdown;
- Windows Smart App Control blocks required native Python evaluation
  dependencies; and
- a real WSL HellaSwag workload generated fresh repeated
  `dxgkio_query_adapter_info: Ioctl failed: -22` records.

No valid official benchmark score has been produced or observed. The CPU
feasibility probe used only synthetic model token IDs for inference. It did
materialize frozen benchmark requests to measure their sizes, but did not
score any request, calculate a likelihood, correctness result, perplexity, or
benchmark metric. Its WSL artifact is
`artifacts/exp012-official-eval/cpu-feasibility.json` (SHA-256
`48802c559cea22fec42283ec87c3c62ef01cbae75d5aa0ed9e9425b895012360`).

WSL CPU FP32 is selected solely as an execution-stability fallback. Every
official evaluator process must start with `CUDA_VISIBLE_DEVICES=""` and
assert `torch.cuda.is_available() == False`; all model tensors must remain on
CPU. Evaluation now uses FP32 rather than CUDA BF16. FP32 is expected to be
at least as numerically faithful as BF16; this is an execution-device/
precision change, not a model, checkpoint, tokenizer, task, or scoring change.

The selected checkpoint SHA, exact 49,860,480 parameter count, frozen
tokenizer, zero-shot protocol, batch size 16, context 512, lm-eval 0.4.9.1,
upstream task definitions, TemplateLM causal likelihood semantics, and
separate WikiText-103 protocol remain mandatory. Checkpoint selection occurred
before any official result. The committed guarded launcher must own a single
task evaluator, persist PID/status/stdout/stderr, reject duplicates, and write
terminal state only after the child exits. A generic DXG log record alone is
recorded but is not a CPU-run stop condition when CUDA remains hidden, no CUDA
API is used, CPU scoring is healthy, and there is no display or system
instability.

No training is authorized. No benchmark score may change any frozen model or
evaluation decision.

## Official evaluation finalization — 2026-08-28

The guarded WSL CPU FP32 suite reached `state: succeeded` with return code
zero. It started at `2026-08-27T20:38:26.845712+00:00` and reached terminal
state at `2026-08-28T05:41:08.956990+00:00`: total guarded-suite duration
`32562.111278` seconds. The nested sequence ran for `32561.097048` seconds.
All five artifacts were rehashed after completion and each SHA-256 exactly
matches its entry in `status/sequence.status.json`:

| Task | Artifact SHA-256 | Evaluator wall seconds |
|---|---|---:|
| HellaSwag | `eeda0b3d58803e8fccc6ef61b4e287195eb70958c269d0c40b5c1685b9be5e50` | 22130.618813981 |
| ARC-Easy | `b4897908dc355352554c89250830a9490d906ed0dc7c1ffa286901ff64ce6ae6` | 776.1316821100045 |
| PIQA | `d2f126c5f2a9b3730d085c95863262b217219c1f344342c1e5502d4e759bb8e6` | 1248.1371511809994 |
| WinoGrande | `7041224d05d0660734f6ec8b376c37391c6abb363da7dc3ea5a620c5cc6d562a` | 337.74195021099877 |
| WikiText-103 | `4104c5e3f7fa3e1a93d24eb1d5fb88b7dd4de37b9fb1bd55e86700808cd9b1a8` | 8058.951546159995 |

Every artifact's embedded metadata was audited: checkpoint SHA-256
`cacb728b3963c10af8f4613149d8b879b0ef6e44558069726c621c1cb1bb981c`,
tokenizer SHA-256 `c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14`,
49,860,480 parameters, CPU FP32, `cuda_available: false`,
`cuda_visible_devices: ""`, zero-shot, batch size 16, context 512,
`lm-eval==0.4.9.1`, and amendment commit
`49bfe789fb8e6ebd23b00b5774f4f2e97ee1c464`. All required reported metrics
were finite.

| Task | Final raw metrics |
|---|---|
| HellaSwag | acc `0.273451503684525`; acc stderr `0.004448196648383006`; acc_norm `0.28759211312487554`; acc_norm stderr `0.004517148434180435` |
| ARC-Easy | acc `0.38552188552188554`; acc stderr `0.009987250004629016`; acc_norm `0.36447811447811446`; acc_norm stderr `0.00987572928248244` |
| PIQA | acc `0.6039173014145811`; acc stderr `0.011411089031912477`; acc_norm `0.6022850924918389`; acc_norm stderr `0.011419114133117227` |
| WinoGrande | acc `0.5035516969218626`; acc stderr `0.014052131146915853` |
| WikiText-103 held-out | perplexity `35.93897257521639`; BPB `1.4083853215598`; scored tokens `350948`; documents `2891`; mean NLL `3.581822293724097` |

The WikiText-103 raw artifact records the exact `mean_negative_log_likelihood`
field above; its summed negative log likelihood is retained in that artifact.
No rounded or derived value replaces the raw record.

### Interpretation boundary and prior interrupted attempts

The official benchmark values are final records, not training signals. They
must not select another checkpoint, alter the model, retune a setting, or
authorize another official benchmark pass.

The display-corruption attempt and terminal-closure interruption remain in the
preserved history directories as non-results. In particular, the terminal
closure evidence includes the `10042/10042` request-construction progress line
but no completed scoring pass, score, or HellaSwag JSON artifact. These attempts
did not inform selection or interpretation of the final result.
