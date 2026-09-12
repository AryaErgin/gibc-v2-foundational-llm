# Data provenance — frozen EXP-020

The final corpus is **7,199,981,568 prediction tokens** from a deterministic, non-cycled 2:1 FineWeb/FineWeb-Edu mixture. It was rebuilt from stream zero because EXP-012 did not persist the deduplication, cursor and mixture state required to prove safe extension.

| Source | Configuration | Frozen revision | Realized prediction tokens |
|---|---|---|---:|
| HuggingFaceFW/fineweb | sample-10BT | `9bb295ddab0e05d785b879661af7260fed5140fc` | 4,799,986,851 |
| HuggingFaceFW/fineweb-edu | default | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` | 2,399,994,717 |

Whole-document token-deficit balancing targets 4,799,987,712 / 2,399,993,856 predictions, with FineWeb winning deterministic ties. Realized contributions differ slightly because documents are selected whole and the final stream is truncated at the exact token boundary. One extra stored target ID gives **7,199,981,569 uint16 IDs / 14,399,963,138 bytes**.

## Screening, splitting and deduplication

Text is taken from the pinned sources' `text` fields. The upstream filtered/deduplicated datasets are further screened by the project. Split assignment uses SHA-256 of seed 42 and the canonical-content SHA-256, modulo 10,000; 200 buckets are reserved for validation. Canonical-content SHA-256 deduplication applies globally across both sources and the full rebuilt stream. EODs, source order and truncation follow the frozen builder.

The contamination index uses NFKC, casefold, normalized tokenized 13-grams, and exact SHA-256 membership. Its immutable SQLite identity is `4b47a02d0bfa793809b02adcc251eb2f3560217e1ddcc0c595a78906386e7a1f`. The final builder's RAM set contains the same exact hashes; RAM membership is an operational representation change, not an approximate filter.

| Recorded counter | FineWeb | FineWeb-Edu |
|---|---:|---:|
| Documents screened | 6,014,561 | 1,922,165 |
| Screening accepted | 6,006,043 | 1,918,768 |
| Contamination rejected | 8,518 | 3,397 |
| Validation exclusions | 120,113 | 38,476 |
| Intra-source duplicate skips | 8,333 | 14 |
| Cross-source duplicate skips | 2,873 | 4,034 |
| Documents contributed | 5,874,724 | 1,876,244 |

Screening acceptance occurs before subsequent split/dedup exclusions; “accepted” is not the same as “contributed.” The final unique selected document count is **7,750,968**.

## Frozen identities

| Artifact | SHA-256 |
|---|---|
| Final full stream | `94e39c09e7696a9668568802c37e6458799b47284efa566f74fb6792f571440e` |
| Final manifest | `aedbbb8dcfe47c5b0d7de2ee052fbeea93232db5a917d09e568c394fe06eacc7` |
| Tokenizer | `c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14` |
| First 1,500,020,737 IDs (EXP-011) | `092fc4a02f991b15fd8fcd2c209754e014485c74bea642c1a57270462141b671` |
| First 2,399,993,857 IDs (EXP-012) | `27c1c8d06da579d443ee19017e12dd28a7c3fb8c6387cff76e9128c7c5fba82c` |

The submission audit independently rehashed the entire stream, both prefixes, tokenizer and benchmark index. Frozen General and Edu tensors remain byte-identical to EXP-012; each contains 131,072 prediction tokens. Exact tensor/file hashes and source manifest are included in [the evidence digest](results/exp020-submission-evidence.json).

## Build history and limitations

The first data preparation attempt was aborted for systems performance before any model training. Recovery preserved scientific semantics through byte/decision-equivalence tests, native WSL scratch, bounded serialization and exact RAM membership. The original ≤12-hour operational gate failed; it was explicitly amended before the subsequent build. The completed manifest records 22,489.316490474 seconds of preparation (about 6.25 h), separately from training.

Builder implementation: `79a61239bfa216e812fb93c81b082812dfeeca89`. Final artifact directory: `artifacts/exp020-final-7p2b-data-rebuild-2` in the historical Windows artifact store. The manifest abbreviates the command using `...`; it is not a literal replay command. Full content hashes and the pinned builder/config are available, but no missing arguments or command timestamp are invented.

Benchmark source revisions used for screening are preserved in the manifest and [benchmark provenance](provenance/exp001-benchmark-revisions.json). They are not automatically evidence of the exact dataset snapshots resolved by the later harness. For example, the recorded official PIQA task path is `baber/piqa`, while screening cites historical `piqa` provenance. The digest retains this distinction; byte equivalence between those dataset snapshots has not been established by this packaging audit.

The screen detects indexed normalized n-gram overlap. It does not prove zero semantic, paraphrased, short-text or unknown-source overlap. Benchmark texts used for exclusion were not training examples or signals for final checkpoint selection.

FineWeb, FineWeb-Edu, their upstream filtering and annotation, Hugging Face tooling, and the benchmark authors are credited in [SOURCE_LEDGER.md](SOURCE_LEDGER.md) and [CODE_ATTRIBUTION.md](CODE_ATTRIBUTION.md). The repository code license does not override upstream data terms; the [scoped license audit](docs/submission/LICENSE_AUDIT.md) records upstream terms and unresolved PIQA/WikiText limits. Public weight delivery does not establish blanket clearance of underlying webpage or benchmark rights; no corpus or benchmark text is redistributed.
