# 3:30 evidence-first video storyboard

**Runtime: exactly 210 seconds.** Human recording/editing still required. This is a demonstration of the model's measured development and reproducibility—not fabricated live generation. Use saved figures and read-only commands only. No training, model inference, benchmark call or dataset loading.

Capture at 1920×1080; show PNGs full-screen, crop away machine account names where unnecessary, use ≥24px captions. Keep narration calm, approximately 125–140 words/minute; use the remaining time in each segment for readable pauses.

| Time | Narration | Screen / action | On-screen caption |
|---|---|---|---|
| 0:00–0:20 | “Can a language model trained from scratch on one laptop GPU provide a credible competition submission? Our final model has exactly 49,860,480 trainable parameters. We trained it on 7.2 billion prediction tokens, then froze it before its official benchmark scores.” | README first screen; slowly highlight exact parameters, tokens and runtime | 49,860,480 parameters · from scratch · EXP-020 frozen |
| 0:20–0:45 | “Here are the results. WikiText-103 token perplexity is 31.783. HellaSwag and ARC-Easy improve against our earlier 2.4-billion-token reference. PIQA is roughly stable. WinoGrande regresses. We report the whole profile, not only the wins. These are single-run comparisons, not a claim of universal reasoning improvement.” | progression.png; zoom right-hand results interpretation | Better HellaSwag / ARC · stable PIQA · lower WinoGrande |
| 0:45–1:10 | “The architecture is deliberately conventional: nine Transformer blocks, width 640, twenty attention heads, SwiGLU, RoPE and tied embeddings. The tokenizer was also trained from scratch. Ordinary AdamW and one cosine schedule cover all 219,726 optimizer updates. No pretrained weights or later fine-tuning supply the model.” | pipeline.png, focus MODEL and TRAIN; brief ARCHITECTURE parameter table | Standard components, exact parameter accounting |
| 1:10–1:35 | “The corpus uses two parts FineWeb to one part FineWeb-Edu with deterministic whole-document mixing. Exact-document deduplication selected about 7.75 million unique documents. Normalized thirteen-gram screening excludes benchmark overlap. Rebuilt historical prefixes match their recorded hashes. This is strong provenance, not a guarantee against every semantic overlap.” | pipeline DATA; DATA_SOURCES prefix hashes and counts; no benchmark text shown | Exact dedup + exclusion screen + historical byte-prefix proof |
| 1:35–2:00 | “The final run took 37.94 hours on one RTX 5090 Laptop GPU, including deliberate thermal pacing. Peak reserved memory was 8.68 decimal gigabytes. Active and paced throughput are reported separately. This curve is internal General and Edu validation—not official benchmark evaluation. The terminal checkpoint was selected under the frozen rule.” | trajectory.png, then summary wall/throughput table | 37.94h final run only · 0.300s pacing included |
| 2:00–2:25 | “Our most useful innovation was experimental discipline. A replicated WSD proxy win failed its longer-horizon gate. QK-Norm improved validation but not enough for promotion. Cautious Weight Decay's intermediate advantage reversed by its endpoint. We retained those failures and did not retune after seeing the result. The final model uses none of those interventions.” | decisions.png; pause on QK and CWD rows | Preregistered gates · negative results retained · no rescue tuning |
| 2:25–2:50 | “Checkpoint selection used only internal validation. After freeze, the official suite ran CPU FP32 and zero-shot; WikiText used the separate held-out rolling evaluator. Historical models had already been benchmarked, so we make a precise claim: this final checkpoint was chosen before its own official scoring. Benchmarks cannot change it now.” | EVIDENCE_AUDIT selection chronology; RESULTS protocol | Selection → freeze → official reporting, not benchmark tuning |
| 2:50–3:15 | “The repository is organized for a quick read and a deeper audit. This offline command checks the evidence digest, frozen config identity, local links and figure provenance without running a model or benchmark. Exact artifact hashes, runtime, source revisions, metric definitions and limits are linked beside the headline results.” | Run `python scripts/verify_submission_package.py`; show PASS, then hash table | Read-only audit · no GPU · no benchmark requests |
| 3:15–3:30 | “AI systems materially assisted research, learning, code and documentation; the neural model itself was trained from scratch. Human review retained scientific authority. Our submission is a measured model, an honest account of its limits, and an experiment trail that can be audited.” | AI_ASSISTANCE, finish README title + repository URL only if human verifies live release | AI-assisted development ≠ pretrained submitted model |

## Exact safe screens/commands

Run from the authoritative source checkout, not the Windows artifact store:

```bash
git rev-parse HEAD
git status --short
python scripts/verify_submission_package.py
python -m json.tool results/exp020-submission-evidence.json
```

The JSON viewer only displays stored metadata; navigate to summary/official_results/sources. Do not display credentials, private logs or benchmark prompts. Open these saved files: `docs/assets/exp020/{trajectory,progression,decisions,pipeline}.png`.

Do not show training/evaluation commands being executed, simulated model answers, or a fabricated live inference demo. The end card can show [the verified public model URL](https://huggingface.co/AryaErgin/gibc-v2-exp020); the final published source revision still requires human confirmation. The script does not imply video production or publication is already complete.
