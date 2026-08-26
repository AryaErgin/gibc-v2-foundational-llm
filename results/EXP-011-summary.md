# EXP-011 — Recipe v3 Long-Horizon Calibration

EXP-011 completed the single authorized fresh seed-42 45,777-step run of Near-Cap Recipe v3 (49,860,480 trainable parameters) through 1,500,020,736 prediction tokens. It is a long-horizon calibration, not automatic final-checkpoint promotion; official benchmarks have not run.

## Provenance and integrity

- Training-source commits: `cfe78547eecb37079a3da102a3f5e6b02f725017` for the verified 0→900M phase; `59db721ad1ed35446a36db1a3807a9ab0ffbdd5a` for the verified 900M→1.5B continuation after its boundary-gate correction.
- Full stream / manifest / tokenizer SHA-256: `092fc4a02f991b15fd8fcd2c209754e014485c74bea642c1a57270462141b671` / `b2ed5e461d753beb581c0d88668371c16abc63c6c9a67673f453a46f27d9feeb` / `c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14`.
- Stored IDs / prediction tokens: `1,500,020,737` / `1,500,020,736`.
- Raw-byte prefix evidence: EXP-004 300M `8e727fa2a2614751a1c34d7f9ac411dfebeb379a09f584ba4f7f418d1059cea1`; EXP-006 900M `07d635264c523ce78b437caef7b50d478a7368371af8f86ef655e4b91661e6af`; both matched exactly.
- The 900M checkpoint restored model, optimizer (83 state entries / 2 groups), CPU/CUDA/Python/NumPy RNG, scheduler step 27,468, and cursor 1,757,952. The final checkpoint loaded at step 45,777 / cursor 2,929,728 with scheduler LR `6e-5`; all logged numerical metrics and final model tensors were finite.

## Validation curve

| Prediction tokens | General | Edu | Combined | LR from run state |
|---:|---:|---:|---:|---:|
| 300,023,808 | 3.6302440166 | 3.3181973100 | 3.4742206633 | 0.0005492980000078377 |
| 600,047,616 | 3.4739109874 | 3.1538517475 | 3.3138813674 | 0.000414472387127598 |
| 900,071,424 | 3.3710331023 | 3.0461599529 | 3.2085965276 | 0.00024724035994133734 |
| 1,200,095,232 | 3.2905968130 | 2.9606119394 | 3.1256043762 | 0.00011175768037925403 |
| 1,500,020,736 | 3.2471743524 | 2.9129971564 | 3.0800857544 | 0.00006000000000000000 |

Combined gains (earlier minus later): 300→600M `0.1603392959`; 600→900M `0.1052848399`; 900M→1.2B `0.0829921514`; 1.2B→1.5B `0.0455186218` nats. The final tranche is **MEANINGFULLY DATA-LIMITED** under the predeclared 0.025–<0.05 band. General and Edu both improved in that tranche (`0.0434224606` and `0.0476147830`); neither shows disproportionate saturation.

## Runtime and checkpoints

- 0→900M training wall time: `8,971.3878 s`; 900M→1.5B continuation: `5,875.3530 s`; total model-training wall time: `14,846.7408 s`.
- Separate data-preparation wall time: `23,989.5951 s`.
- Continuation mean/final throughput: `102,624.21 / 105,443.83 tok/s`. Overall peak allocated/reserved VRAM across phases: `7,687,993,856 / 8,491,368,448` bytes.
- Approximate training compute: `4.4875052344147968e17` FLOPs, using `6 × trainable parameters × prediction tokens`.
- Checkpoint SHA-256: 900M `e3e3aa36f37c40baeef52308cf9ee2802d3d005c6f247736edd6c25100670896`; 1.2B `1ca9f93187590c42a111454185b273d5a9ca082f52473c6ce1f1ed557256d87c`; 1.5B `c1e65c7de00f100e0ecc4a9d0d5db148bde827fb0b22891b8f69a9b9a37928c0`.

Data Recipe v1 remains globally canonical-content-SHA-256 deduplicated across the inherited prefix and extension. The deterministic 2:1 target was FineWeb/FineWeb-Edu `1,000,013,824 / 500,006,912`; realized prediction-token contributions were `1,000,014,372 / 500,006,364`. Screening reused the EXP-006 NFKC+casefold+tokenized normalized 13-gram SHA-256 index (`4b47a02d0bfa793809b02adcc251eb2f3560217e1ddcc0c595a78906386e7a1f`). This screen detects indexed normalized n-gram overlaps only; it does not prove absence of all contamination.
