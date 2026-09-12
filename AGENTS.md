# Repository guidance — frozen submission phase

EXP-020 is the immutable GIBC V2 Track 01 competition model. Current evidence is in [RESULTS.md](RESULTS.md) and [the audit](docs/submission/EVIDENCE_AUDIT.md). Historical experiments/plans are records, not launch authorization.

- Do not train, resume, tune, rerun evaluation, or modify/overwrite frozen checkpoints, tokenizer, data, config or benchmark results.
- No benchmark requests or dataset materialization during packaging.
- Preserve negative results and historical chronology. Update current summaries without retrospectively changing old observations.
- Keep weights, data, caches, credentials and environments out of Git.
- The local repository is authoritative; no resets to remote state.
- Preserve `src/gibc_v2_foundational_llm.egg-info/` untracked.
- Documentation, recorded-data figures and offline checks are allowed. No commit, tag or push without authorization.
- Disclose AI assistance; never invent evidence or imply AI assistants constitute the scratch-trained model.
