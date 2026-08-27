# EXP-012 — Fresh 2.4B-token long-horizon calibration

Status: **EXP-012 COMPLETE**. This is a fresh seed-42 calibration under a 73,242-step cosine horizon, not an EXP-011 schedule reproduction or automatic final-model promotion. Official benchmarks were not run.

## Integrity and provenance

- Training source commit: `4b22f8d7a7eacbdd315cbb454a813203ae410c1d`.
- Recipe v3: 49,860,480 trainable parameters; loaded terminal checkpoint recount passed.
- Stored IDs / prediction tokens: `2,399,993,857` / `2,399,993,856`; final step/cursor: `73,242` / `4,687,488` sequential examples.
- Stream / manifest / tokenizer SHA-256: `27c1c8d06da579d443ee19017e12dd28a7c3fb8c6387cff76e9128c7c5fba82c` / `b19b508dd1d1928b8e3bbdf586547791dc3bd76af19f6e55b8c39465bd749ccf` / `c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14`.
- Strict loader independently rehashed and passed the exact raw-byte prefixes: EXP-004 300M `8e727fa2a2614751a1c34d7f9ac411dfebeb379a09f584ba4f7f418d1059cea1`; EXP-006 900M `07d635264c523ce78b437caef7b50d478a7368371af8f86ef655e4b91661e6af`; EXP-011 1.5B `092fc4a02f991b15fd8fcd2c209754e014485c74bea642c1a57270462141b671`.
- Target FineWeb/FineWeb-Edu prediction tokens: `1,599,995,904 / 799,997,952`; realized: `1,599,995,879 / 799,997,977`. The one-ID terminal offset is represented in stored contributions.
- The artifact was rebuilt from stream zero with one global canonical-content-SHA-256 dedup state across prefix and extension; 2,588,927 unique documents were selected. The contamination index was copied byte-for-byte from EXP-011 (`4b47a02d0bfa793809b02adcc251eb2f3560217e1ddcc0c595a78906386e7a1f`) and uses NFKC+casefold+tokenized normalized 13-gram SHA-256 matching. It detects indexed normalized overlap, not all lexical, semantic, or unknown-source contamination.

## Validation curve

| Step | Prediction tokens | General | Edu | Combined | LR recorded by run |
|---:|---:|---:|---:|---:|---:|
| 9,156 | 300,023,808 | 3.6363616884 | 3.3267031312 | 3.4815324098 | 0.0005798307370253698 |
| 18,312 | 600,047,616 | 3.4983913898 | 3.1783478260 | 3.3383696079 | 0.0005215206034459287 |
| 27,468 | 900,071,424 | 3.4249964058 | 3.1092402041 | 3.2671183050 | 0.00043396971415680507 |
| 36,624 | 1,200,095,232 | 3.3709251583 | 3.0435474217 | 3.2072362900 | 0.00033054506007076256 |
| 45,780 | 1,500,119,040 | 3.3114303648 | 2.9711615443 | 3.1412959546 | 0.0002270371879492215 |
| 54,936 | 1,800,142,848 | 3.2562543750 | 2.9141408801 | 3.0851976275 | 0.00013924935001819903 |
| 64,092 | 2,100,166,656 | 3.2110527456 | 2.8723290563 | 3.0416909009 | 0.00008058471375438371 |
| 73,242 | 2,399,993,856 | 3.1909595430 | 2.8499483168 | 3.0204539299 | 0.00006000000000000000 |

Final approximately-300M tranche: 2,100,166,656 to 2,399,993,856 prediction tokens (299,827,200). Combined gain was `0.0212369710`; General/Edu gains were `0.0200932026 / 0.0223807395`. Therefore the predeclared classification is **APPROACHING DIMINISHING RETURNS**. There was no terminal regression.

## Runtime and checkpoints

- Separate data preparation: `29,422.8518 s`; full model-training wall time: `24,362.3826 s`.
- Mean/final throughput: `99,429.40 / 90,443.73 tok/s`; peak allocated/reserved VRAM: `7,686,099,968 / 8,491,368,448` bytes.
- Approximate training compute: `717,989,073,943,265,280` FLOPs (`6 × trainable parameters × prediction tokens`). Hardware/software: Windows 10.0.26200, RTX 5090 Laptop GPU, Python 3.11.9, PyTorch 2.13.0+cu132, CUDA runtime 13.2, BF16.
- Checkpoint SHA-256: 300M `36a0dd8a26f6da6295050e2697c0587758f4e8aa6b27225b5f50a638ae1100ed`; 600M `4df6058cb5e6a7b8e2da43bbbfd7f990ef946ea239d2ef28ad5b2a4f703a813a`; 900M `e2cf4d0be42e1d3f8d79098c6d54e347cc7385c094d549deab27c6f163792ab9`; 1.2B `c5167526b94a5909be9db5228f0222545f7ac822a7ab4dca252eb0ec321155fa`; 1.5B `11c25812875c92c2629e84320b98b0e4cceb148684f26e3da99e29fd74834dcd`; 1.8B `5894fc34287c4ff2423dfff9e78a935923c574e581c305130df28e02166dedfe`; 2.1B `1d3ce78cf5f0779820aa9197ad10553c4636121cb0524fbb104a65d48b71dce2`; terminal `cacb728b3963c10af8f4613149d8b879b0ef6e44558069726c621c1cb1bb981c`.

All logged metrics, model tensors, and optimizer tensors were finite. Every checkpoint had matching run-state/data cursor, scheduler step count, and Python/NumPy/CPU/CUDA RNG state. The data download incurred one automatically retried transient `IncompleteRead`; it recovered without a restart or a changed control. The first terminal-status watcher had a PowerShell quoting error and produced no record; this evidence file supersedes it with the audited terminal status.

## Official CPU evaluation interruption — 2026-08-27

The first WSL CPU FP32 HellaSwag sequence was interrupted when its integrated terminal closed. The stale suite, sequence, and HellaSwag status/log evidence was copied with original timestamps and SHA-256 checksums to `artifacts/exp012-official-eval/history/interrupted-by-terminal-closure-20260827T202544Z` before recovery. Its `10042/10042` stderr progress line records request construction only; it is not a completed scoring pass. No `artifacts/exp012-official-eval/lm_eval/hellaswag.json` result artifact, HellaSwag score, or valid official result was produced. This is classified as `INTERRUPTED_BY_TERMINAL_CLOSURE`, not a failed model or evaluation result; it did not affect checkpoint selection or frozen evaluation semantics.
