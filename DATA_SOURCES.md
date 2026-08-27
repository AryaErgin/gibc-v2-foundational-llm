# Data Sources

## Data Recipe v1

The selected near-cap experiments use a deterministic 2:1 FineWeb / FineWeb-Edu mixture: `HuggingFaceFW/fineweb` `sample-10BT`, revision `9bb295ddab0e05d785b879661af7260fed5140fc`; and `HuggingFaceFW/fineweb-edu` `default`, revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`. Text comes from each dataset's `text` field. Deterministic split assignment is `sha256(seed:canonical_content_sha256) modulo 10000`, seed 42, with buckets below 200 reserved for validation.

EXP-011's non-cycled full artifact contains 1,500,020,737 stored uint16 IDs and 1,500,020,736 prediction tokens. Its stream SHA-256 is `092fc4a02f991b15fd8fcd2c209754e014485c74bea642c1a57270462141b671`; its manifest SHA-256 is `b2ed5e461d753beb581c0d88668371c16abc63c6c9a67673f453a46f27d9feeb`; its frozen 8,192-entry tokenizer SHA-256 is `c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14`.

Target prediction-token contributions were FineWeb `1,000,013,824` and FineWeb-Edu `500,006,912`. The verified realized contributions were `1,000,014,372` and `500,006,364`; the one-token stored-ID offset belongs to FineWeb's terminal target. Mixing is deterministic token-deficit-balanced whole-document selection, with FineWeb winning ties.

## Deduplication and contamination controls

Deduplication is global `canonical_content_sha256` whole-document deduplication across both sources and across the inherited prefix plus extension, not a separately reset extension pass. The EXP-011 manifest records 1,620,761 unique documents; within-source duplicate skips FineWeb/FineWeb-Edu `236/3`; and cross-source skips `91/358`.

Contamination screening uses NFKC + casefold + tokenized normalized 13-gram SHA-256 overlap against the immutable benchmark index reused from EXP-006, SHA-256 `4b47a02d0bfa793809b02adcc251eb2f3560217e1ddcc0c595a78906386e7a1f`. The manifest records accepted/rejected/scanned counts and pinned HellaSwag, ARC-Easy, PIQA, WinoGrande, and WikiText-103 source revisions/splits. This finite normalized n-gram screen is evidence against detected overlap, not proof of zero lexical, semantic, or unknown-source contamination.

## Prefix provenance

The full artifact's first 300,023,809 stored IDs / 300,023,808 predictions exactly match EXP-004 (`8e727fa2a2614751a1c34d7f9ac411dfebeb379a09f584ba4f7f418d1059cea1`). Its first 900,071,425 stored IDs / 900,071,424 predictions exactly match EXP-006 (`07d635264c523ce78b437caef7b50d478a7368371af8f86ef655e4b91661e6af`).

## EXP-012 extension record

EXP-012 rebuilt the stream from zero because the prior builder had no serializable global-dedup state. Its one global canonical-content-SHA-256 state covers the complete 2.4B artifact, including all inherited prefixes. The artifact has 2,399,993,857 stored uint16 IDs / 2,399,993,856 prediction tokens; stream / manifest / tokenizer SHA-256 are `27c1c8d06da579d443ee19017e12dd28a7c3fb8c6387cff76e9128c7c5fba82c` / `b19b508dd1d1928b8e3bbdf586547791dc3bd76af19f6e55b8c39465bd749ccf` / `c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14`.

Target FineWeb/FineWeb-Edu prediction-token contributions were `1,599,995,904 / 799,997,952`; realized contributions were `1,599,995,879 / 799,997,977`. The artifact independently passed exact EXP-004 300M, EXP-006 900M, and EXP-011 1.5B prefix checks; the EXP-011 prefix SHA-256 is `092fc4a02f991b15fd8fcd2c209754e014485c74bea642c1a57270462141b671`.
