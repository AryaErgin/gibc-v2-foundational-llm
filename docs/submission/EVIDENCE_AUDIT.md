# Submission evidence audit — 2026-09-08

Scope: read-only inspection and derived submission documentation after final-model and benchmark freeze. No training, forward inference, benchmark request, dataset loading, artifact rewrite, commit or push is part of this audit.

## Verified

The [machine-readable digest](../../results/exp020-submission-evidence.json) records source paths, byte sizes and independently recomputed SHA-256 values. It contains the exact final summary, public-safe metric/task metadata, data manifest, all validation milestones and 1,000-update training-loss means. It contains no benchmark prompts, answers or model weights.

- Final checkpoint hash/598,447,091 bytes, config, tokenizer, summary and manifest matched their frozen identities.
- Full 14,399,963,138-byte token stream and EXP-011/012 prefixes rehashed successfully.
- Benchmark index SHA matched; frozen General/Edu tensor files are byte-identical to EXP-012.
- CPU read-only checkpoint inspection counted 49,860,480 unique FP32 tensor elements, counting tied storage once, all finite. No model was instantiated or executed for this audit.
- All 219,726 training events are sequential, finite, and match tokens/cursor arithmetic.
- All five official result-file hashes match the completed sequence's recorded hashes; suite state is succeeded with return code zero.
- Final summary records WSL2, RTX 5090 Laptop GPU, Python 3.11.9, torch 2.13.0+cu132 and CUDA 13.2; the single-device launch is preserved in its original log script.
- Final training wall time is 136,589.44783848198 s = 37.9415132885 h. This includes pacing, but excludes data preparation, method exploration, aborted preparation and CPU official scoring.
- The terminal checkpoint is lowest-Combined among eligible late candidates.

## Selection and benchmark chronology

The preregistered rule exists in [exp020-final-preregistration.json](../../provenance/exp020-final-preregistration.json). The benchmark-free preflight binds the selected checkpoint/hash and reports no benchmark execution. Its recorded filesystem modification time is 2026-09-07 08:27:15 UTC; HellaSwag scoring began at 08:51:10 UTC. The sequence records selection before benchmark exposure. Filesystem timestamps are corroboration, not cryptographically signed independent time evidence.

Historical EXP-012 and other models were benchmarked earlier; public benchmark text was used to construct the exclusion index. Thus the claim is final EXP-020 selection before its official scoring, not total project-wide absence of benchmark exposure.

## Conflicts and limitations found before narrative edits

| Evidence issue | Treatment |
|---|---|
| README called EXP-012 final; current docs still promoted WSD or listed EXP-020 unlaunched | Update current summaries; retain historical result/preregistration files and date their status |
| Final data manifest's build command contains `...` | Preserve it; no invented literal replay command |
| Generic post-build verifier exit 143/log refers to aborted preparation | Do not cite it as success for rebuild-2; the current audit independently verifies rebuilt contents |
| Initial audit omitted the raw harness git_hash field | Corrected: all four original task JSONs record `3733279`, uniquely resolving to `37332797909df963ca7c77a945cea8752b60d481`; frozen artifacts remain unchanged |
| Contamination-source revisions and official harness dataset paths differ in scope | Preserve both. Exact snapshot equivalence is not established; official task definitions are retained verbatim in the digest |
| At the initial audit, no verified public EXP-020 weight-download endpoint | Resolved 2026-09-09: [public model and independent remote hash receipt](PUBLIC_RELEASE.md); original frozen evidence unchanged |
| Old WikiText harness metrics differ from official token PPL | Exclude EXP-001 WikiText from the direct token-PPL chart |
| Full campaign cost, energy and statistical significance not established | Report final-run measured wall time; no energy or SOTA claims |

## Source identities

- Training/preregistration runtime identity: `d88800733846c7a30e2044fa0a20f9e4a448f328`.
- Builder implementation: `79a61239bfa216e812fb93c81b082812dfeeca89`.
- Evaluator execution source, expanded from the raw artifacts' recorded short `3733279`: `37332797909df963ca7c77a945cea8752b60d481`.
- Submission work starts from that HEAD and remains uncommitted.

The audit does not repair or rerun the frozen evaluator. It documents the results and their practical provenance limits.
