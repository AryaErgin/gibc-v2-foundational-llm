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
python scripts/eval_exp012_official.py --task hellaswag --output artifacts/exp012-official-eval/lm_eval/hellaswag.json
python scripts/eval_exp012_official.py --task arc_easy --output artifacts/exp012-official-eval/lm_eval/arc_easy.json
python scripts/eval_exp012_official.py --task piqa --output artifacts/exp012-official-eval/lm_eval/piqa.json
python scripts/eval_exp012_official.py --task winogrande --output artifacts/exp012-official-eval/lm_eval/winogrande.json
python scripts/eval_exp012_wikitext103.py --output artifacts/exp012-official-eval/wikitext103.json
```

These commands execute only after the frozen pre-evaluation gate command passes. No benchmark result authorizes further training.

## EVALUATION-ENVIRONMENT-COMPATIBILITY amendment

Before any benchmark request, the first HellaSwag invocation failed while importing `lm_eval.api.metrics`: `sacrebleu 2.6.0` imported `lxml.etree`, whose native DLL was blocked by Windows Smart App Control. No lm-eval task request, dataset example, or benchmark metric executed, and no raw result artifact was written.

`lm-eval` remains exactly `0.4.9.1`. Its installed metadata permits `sacrebleu>=1.5.0`. The required packaged task definitions are unchanged accuracy-only multiple-choice tasks: HellaSwag `acc`/`acc_norm`, ARC-Easy `acc`/`acc_norm`, PIQA `acc`/`acc_norm`, and WinoGrande `acc`. They define no BLEU, chrF, TER, or other sacrebleu metric. The smallest compatible environment change is therefore to pin only `sacrebleu==1.5.1`, which has no `lxml` runtime dependency; no lm-eval source, task YAML/template, checkpoint, model, tokenizer, or protocol setting changes.

Before resuming evaluation, record the old/new package versions, run `pip check`, and require clean `import sacrebleu`, `import lm_eval`, and `from lm_eval import evaluator` smoke tests. Re-run all selected-checkpoint, parameter, tokenizer/config, adapter, causal-shift, context, and unit-test gates. If any import or gate fails, stop without relaxing Windows security or executing a benchmark request.
