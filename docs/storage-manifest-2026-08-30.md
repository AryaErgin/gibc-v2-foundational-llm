# Storage manifest — candidate classification (2026-08-30)

This is a directory-size inventory only. No artifact was hashed recursively,
compressed, moved, or deleted. Windows artifact root was approximately 75G;
the WSL repository was approximately 642M.

| Paths / class | Estimated size | Classification | Rationale |
|---|---:|---|---|
| `exp012-full-data` and immutable stream/manifest/tokenizer | 5.5G | `KEEP_CRITICAL` | Required frozen 2.4B stream for EXP-017A. |
| `exp012-full`, `exp012-eval-checkpoint`, terminal provenance | about 5.2G | `KEEP_CRITICAL` | Exact 2.4B cosine reference and validation lineage. |
| `exp013-wsd`, `exp013-seed43-wsd`, configs/logs/terminal checkpoints | about 4.6G | `KEEP_CRITICAL` | Promoted WSD evidence across seeds. |
| `exp017a-wsd-2p4b` Attempt 1 logs/telemetry/abort JSON | 52K | `KEEP_CRITICAL` | Technical-abort provenance; no checkpoint exists. |
| `exp015-fixed-examples` and schedules/configs/logs | 293M plus retained evidence | `KEEP_UNTIL_PAPER` | Fixed-example placement provenance. |
| EXP-014/015/016 configs, logs, manifests, terminal summaries | small subset of run dirs | `KEEP_UNTIL_PAPER` | Negative-result evidence must remain auditable. |
| historical completed full-run checkpoints not in active lineage | roughly 15G | `ARCHIVE` | Retain until a paper/submission retention audit. |
| failed-method binaries: `exp014-llr`, `exp015-a/b/c-*`, `exp016-control-wsl`, `exp016-magma-wsl` | roughly 12G | `DELETE_CANDIDATE` | Only after artifact-by-artifact provenance audit confirms configs/logs/hashes are retained. |
| smoke/preflight/retry checkpoints and duplicated publication/preflight directories | roughly 20–30G | `DELETE_CANDIDATE` | Only after exact duplication and recovery requirements are audited. |
| `diagnostic-cache*` | about 1G | `DELETE_CANDIDATE` | Diagnose contents and provenance before any removal. |

These are planning classes, not deletion authorization. The total candidate
estimate is deliberately broad (about 33–43G) because no recursive
file-level duplicate audit was performed. No destructive deletion occurred.
