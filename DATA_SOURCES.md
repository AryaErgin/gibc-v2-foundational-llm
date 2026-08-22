# Data Sources

FineWeb source: `HuggingFaceFW/fineweb`, official `sample-10BT` config, revision `9bb295ddab0e05d785b879661af7260fed5140fc`, field `text`, ODC-BY license. SHA-256 content IDs and `sha256(seed:id) % 10000 < 200` give deterministic 2% validation/98% training assignment (seed 42).

Tokenizer training used only accepted train documents (about 100 MiB), byte-level BPE from scratch, exactly 8192 entries, special token `<|endoftext|>`, SHA-256 `3adc8a1d02f9e9d28d08eb8761206732877eb25c65aab2f80f34d1b6c1a1f122`. Held-out stats: 3.6307 bytes/token, 3.6307 characters/token, 1.4905 tokens/word.

Decontamination normalizes NFKC/casefold/word-punctuation tokens, hashes normalized 13-grams, and rejects overlap against local-only HellaSwag, ARC-Easy, PIQA, WinoGrande, and WikiText-103 validation/test indexes. 34,302 documents scanned; 34 accepted-document matches rejected. Diagnostics contain hashes/counts only. No overlap detected by this procedure does not prove zero contamination.
