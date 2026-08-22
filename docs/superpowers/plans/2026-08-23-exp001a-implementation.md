# EXP-001A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate the approved EXP-001 decoder-only baseline infrastructure without beginning the 100M-token training run.

**Architecture:** A small Python package separates model mathematics, tokenizer/data preparation, training/checkpointing, generation, and evaluation adaptation. Declarative YAML supplies every scientific value; preparation produces immutable manifests, while training writes ignored run artifacts and a small committed measured summary.

**Tech Stack:** Python 3.11.9; PyTorch 2.13.0+cu132 native SDPA; CUDA 13.2; RTX 5090 Laptop GPU BF16; Hugging Face `datasets`; `tokenizers`; `lm-eval`; PyYAML; pytest.

**Spec:** `docs/superpowers/specs/2026-08-23-exp001a-design.md`

## Global Constraints

- Use exactly 8,192 vocabulary entries, d_model 256, 8 layers, 8 heads, head dim 32, d_ff 1,024, GELU, pre-RMSNorm, RoPE, no linear bias, dropout 0, context 512, standard causal MHA, and tied embeddings.
- Fail if trainable parameter count is not exactly 8,392,960.
- Use `scaled_dot_product_attention(..., is_causal=True)`, BF16 CUDA autocast, AdamW β=(0.9,0.95), eps=1e-8, weight decay 0.1, peak LR 6e-4, cosine schedule, 100 warmup steps, clipping 1.0, and effective 32,768-token batches.
- No `torch.compile`, Triton, custom CUDA kernels, GQA/MQA, pretrained model weights/vocabularies, distributed training, or full 100M-token run.
- FineWeb remains the sole EXP-001 text source. Resolve official immutable data/benchmark revisions and exact package pins before materialization; record them in manifests and `DATA_SOURCES.md`.
- Never commit local instruction content, raw/derived benchmark content, data, caches, token shards, checkpoints, credentials, `.venv`, or large result artifacts.
- Tests are required for scientifically or operationally critical behavior only; documentation and trivial CLI forwarding are manually reviewed instead.

### Task 1: Reproducible package and fixed configuration

**Files:**
- Create: `pyproject.toml`, `configs/exp001.yaml`, `src/gibc_llm/__init__.py`, `src/gibc_llm/utils.py`, `tests/test_config.py`
- Modify: `.gitignore`, `README.md`

**Interfaces:**
- Produces `ExperimentConfig`, `load_config(path)`, `set_global_seed(seed)`, `sha256_file(path)`, `atomic_json_write(path, value)`, and `collect_environment()`.
- Consumes no model or dataset code.

- [ ] **Step 1: Write the failing configuration-invariant test.**

```python
def test_exp001_config_matches_approved_control() -> None:
    cfg = load_config(Path("configs/exp001.yaml"))
    assert (cfg.model.vocab_size, cfg.model.d_model, cfg.model.n_layers) == (8192, 256, 8)
    assert (cfg.training.effective_batch_tokens, cfg.training.warmup_steps) == (32768, 100)
    assert cfg.training.full_training_tokens == 100_000_000
```

- [ ] **Step 2: Run `pytest tests/test_config.py -q`; confirm it fails because the package/configuration does not exist.**
- [ ] **Step 3: Add the minimal YAML schema and utility implementation.** Parse nested dataclasses, reject unknown/missing keys, seed Python/NumPy/PyTorch/CUDA, and return masked machine/environment metadata; pin each installed dependency in `pyproject.toml` after installation succeeds in `.venv`.
- [ ] **Step 4: Run the configuration test and `python -m compileall src`; confirm both pass.**
- [ ] **Step 5: Manually inspect `.gitignore` coverage and README setup commands; do not add ceremonial CLI tests.**

### Task 2: Causal Transformer with exact accounting

**Files:**
- Create: `src/gibc_llm/model.py`, `scripts/count_parameters.py`, `tests/test_model.py`

**Interfaces:**
- Produces `RMSNorm(dim)`, `RotaryEmbedding(head_dim)`, `CausalSelfAttention(config)`, `TransformerBlock(config)`, `DecoderOnlyTransformer(config)`, and `parameter_breakdown(model)`.
- `DecoderOnlyTransformer.forward(input_ids)` returns logits `[B,T,8192]`; `loss(input_ids)` trains positions `0..T-2` against `1..T-1`.

- [ ] **Step 1: Write failing tests for parameter total/breakdown, tied storage, causal non-leakage, target shift, all critical tensor shapes, and parameter-free RoPE.**

```python
def test_parameter_count_is_the_approved_exact_total() -> None:
    model = DecoderOnlyTransformer(exp001_model_config())
    assert parameter_breakdown(model).total == 8_392_960

def test_future_tokens_cannot_change_prior_logits() -> None:
    a, b = fixed_prefix_sequences_that_diverge_after(5)
    model = DecoderOnlyTransformer(exp001_model_config()).eval()
    assert torch.allclose(model(a)[:, :6], model(b)[:, :6], atol=1e-5, rtol=1e-5)
```

- [ ] **Step 2: Run `pytest tests/test_model.py -q`; confirm failures are missing symbols, not test setup errors.**
- [ ] **Step 3: Implement RMSNorm with only a learned scale; sin/cos RoPE buffers/operations with no `Parameter`; Q/K/V/O and MLP bias disabled; pre-norm residual blocks; native causal SDPA; final RMSNorm; and `F.linear(hidden, token_embedding.weight)` for genuine sharing.**
- [ ] **Step 4: Run `pytest tests/test_model.py -q` and `python scripts/count_parameters.py --config configs/exp001.yaml`; require exactly 8,392,960 and output categories embedding, attention, MLP, norms, output-head additional, total.**
- [ ] **Step 5: Refactor only duplicated shape/validation helpers while retaining green tests.**

### Task 3: Deterministic tokenizer, split, packing, and decontamination

**Files:**
- Create: `src/gibc_llm/tokenizer.py`, `src/gibc_llm/data.py`, `tests/test_tokenizer.py`, `tests/test_data.py`, `tests/test_decontamination.py`, `scripts/prepare_exp001.py`
- Modify: `DATA_SOURCES.md`, `.gitignore`

**Interfaces:**
- Produces `Document`, `stable_document_id(text)`, `assign_split(id, validation_fraction, seed)`, `normalize_for_ngrams(text)`, `NgramContaminationFilter`, `train_tokenizer(train_texts, output_dir)`, `pack_documents(token_ids, context_length)`, and `prepare_exp001(config)`.
- Preparation returns a JSON-serializable manifest containing only hashes/counts/provenance/statistics, never benchmark item text.

- [ ] **Step 1: Write failing synthetic tests for repeatable train/validation assignment and serialized packed tokens, exactly 8,192 tokenizer vocabulary entries including declared special tokens, training-only tokenizer input, synthetic contaminated-document rejection, clean-document survival, and absence of raw benchmark text from diagnostic records.**

```python
def test_contamination_rejects_overlap_but_keeps_clean_text() -> None:
    index = NgramContaminationFilter.from_texts(["alpha beta gamma delta epsilon"])
    assert index.screen("prefix alpha beta gamma delta epsilon suffix").rejected
    assert not index.screen("unrelated orchard meteor library velvet").rejected
```

- [ ] **Step 2: Run the three test files; confirm expected missing-module failures.**
- [ ] **Step 3: Implement byte-level BPE from scratch with exactly two special tokens, `<|eod|>` and `<|pad|>`, and no unknown-token special token; initialize the byte alphabet so byte coverage does not require UNK. Calculate trainer vocabulary size so the serialized total is exactly 8,192. Save tokenizer JSON, SHA-256, fixed held-out-sample bytes/characters/tokens-per-word, and human-readable token examples.**
- [ ] **Step 4: Implement deterministic content-hash splitting, EOD insertion, contiguous 512-token packing, and normalized 13-gram screening. Build official benchmark n-gram inputs locally from pinned dataset revisions and emit only benchmark source identifiers, source hashes, document hashes, counts, and overlap metadata.**
- [ ] **Step 5: Resolve the official FineWeb repository/config/revision/license/field from its source metadata before fetching. Materialize only configured bounded train/tokenizer/validation amounts, use training partition only for tokenizer training, and write immutable manifests under ignored artifacts. Update `DATA_SOURCES.md` with the resolved facts.**
- [ ] **Step 6: Run the data tests and a bounded preparation command twice; require identical tokenizer and packed-data hashes across the two runs.**

### Task 4: BF16 training, validation, checkpoints, and learning diagnostics

**Files:**
- Create: `src/gibc_llm/train.py`, `tests/test_training.py`, `tests/test_checkpoint.py`, `scripts/train_exp001.py`
- Modify: `EXPERIMENT_LOG.md`, `experiments/EXP-001.md`

**Interfaces:**
- Produces `build_optimizer`, `CosineWithWarmup`, `evaluate`, `save_checkpoint`, `load_checkpoint`, `train`, and `RunState`.
- A checkpoint stores model/optimizer/scheduler state, step/tokens counters, config, Python/NumPy/PyTorch CPU/CUDA RNG states, and tokenizer/data identifiers.

- [ ] **Step 1: Write failing tests for next-token training loss, a fixed tiny-batch loss trajectory, checkpoint output equivalence, restored counters, restored RNG sequence, and a resumed optimizer step.**

```python
def test_checkpoint_round_trip_restores_outputs_and_rng(tmp_path: Path) -> None:
    before = model(ids).detach().clone()
    save_checkpoint(path, model, optimizer, scheduler, state, config)
    restored = load_checkpoint(path, fresh_model, fresh_optimizer, fresh_scheduler)
    assert torch.equal(before, fresh_model(ids))
    assert restored.state.tokens == state.tokens
```

- [ ] **Step 2: Run the training/checkpoint tests and confirm expected missing-import failures.**
- [ ] **Step 3: Implement one AdamW parameter group applying the approved 0.1 weight decay uniformly to every trainable parameter, cosine warmup schedule, BF16 autocast, finite gradient norm/clip, effective 32,768-token gradient accumulation, held-out average cross-entropy/perplexity, and JSONL metric records.**
- [ ] **Step 4: Implement atomic checkpoint save/load and device-neutral restore. Run CPU-friendly checkpoint/RNG tests and GPU tiny-overfit test; require materially downward loss rather than an arbitrary benchmark threshold.**
- [ ] **Step 5: Run a bounded CUDA microbatch profile over 8/16/32/64 sequences where feasible, select the largest stable candidate, derive accumulation to exactly 32,768 tokens, and record measured throughput/allocated+reserved memory.**

### Task 5: Generation and lm-evaluation-harness bridge

**Files:**
- Create: `src/gibc_llm/generation.py`, `src/gibc_llm/evaluation.py`, `scripts/generate.py`, `scripts/eval_exp001.py`, `tests/test_generation.py`, `tests/test_evaluation.py`
- Modify: `README.md`, `DATA_SOURCES.md`

**Interfaces:**
- Produces `generate(model, tokenizer, prompt, max_new_tokens, temperature, top_k, seed)`, `CustomCausalLM`, `loglikelihood(requests)`, and `loglikelihood_rolling(requests)`.
- Evaluation adapter loads only a local custom checkpoint/tokenizer and delegates task definitions to the pinned harness without template mutation.

- [ ] **Step 1: Write failing tests that greedy generation appends valid token IDs, seeded sampling is reproducible, log-likelihood scores the continuation rather than the context, rolling likelihood covers every predicted token exactly once, and the adapter never calls pretrained weight loading.**
- [ ] **Step 2: Run generation/evaluation tests and confirm expected missing-module failures.**
- [ ] **Step 3: Implement minimal cached/non-cached generation and adapter methods required by the installed pinned lm-eval interface. Use local token IDs/logits, explicit context truncation, and continuation masks.**
- [ ] **Step 4: Run synthetic adapter tests, then invoke bounded `limit` smoke checks for HellaSwag, ARC-Easy, PIQA, WinoGrande, and the WikiText-103 rolling-PPL path. Record task names, harness version, commands, few-shot/batch settings, templates as supplied, and any protocol ambiguity; do not report limited results as official scores.**

### Task 6: Smoke validation, documentation, and evidence review

**Files:**
- Modify: `ARCHITECTURE.md`, `DATA_SOURCES.md`, `EXPERIMENT_LOG.md`, `experiments/EXP-001.md`, `README.md`, `AI_ASSISTANCE.md`, `results/EXP-001A-summary.md`

**Interfaces:**
- Consumes actual run manifests, metrics JSONL, checkpoint metadata, parameter report, and test output.
- Produces a compact public summary with measured values and no raw benchmark/data content.

- [ ] **Step 1: Run full `pytest -q`, the parameter script, two deterministic preparation comparisons, tiny overfit, checkpoint/resume, generation smoke, bounded harness invocation, and a CUDA smoke run of no more than approximately two million tokens.**
- [ ] **Step 2: Extract measured first/final train and validation loss, PPL, microbatch/accumulation, throughput, allocated/reserved VRAM, wall time, checkpoint size, data counts, and FLOPs estimate `6 * 8,392,960 * measured_training_tokens`. Label FLOPs as an estimate.**
- [ ] **Step 3: Update documentation with equations and code mapping, source/revision/license/fields, artifact hashes, warnings, exact commands, all dependencies, and the explicit “in progress / infrastructure validated” EXP-001 state.**
- [ ] **Step 4: Self-review the plan/spec/result against every fixed scientific variable. Stop and request direction for a conflict; otherwise commit and push the validated implementation.**
