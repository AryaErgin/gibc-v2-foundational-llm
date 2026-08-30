# EXP-016 Magma - finalized 2026-08-30

## Decision

**Reject Magma.** The preregistered fresh seed-42 Magma arm was worse than its
contemporaneous ordinary-AdamW control by `0.09226645529270172` combined NLL
(`+2.738047133223155%`). It failed the capability threshold and both domain
guards. Efficiency passed but cannot rescue capability failure. Recipe-v3 + WSD
remains promoted. No tuning, seed-43, benchmark, or follow-up is authorized.

## Frozen design

Both arms used commit `6d573de36939b352eae305224aa3ddd9bd61f0a0`, fresh seed
42, WSL Python 3.11 / torch 2.13.0+cu132, RTX 5090 Laptop, Recipe-v3
(49,860,480 parameters), BF16, AdamW beta1=0.9 beta2=0.95 eps=1e-8, weight
decay 0.1, clip 1.0, WSD, ctx512, physical batch 32, accumulation 2, and
9,156 updates / 300,023,808 tokens. Schedule A SHA256:
`39c509f59489d125904be61e7e3094e0e87af5ee7ead46afe6742cac35185eb2`; stream:
`8e727fa2a2614751a1c34d7f9ac411dfebeb379a09f584ba4f7f418d1059cea1`.

Magma alone used the preregistered wrapper: p=0.5, tau=2.0, EMA smoothing 0.9,
dedicated seed-42 CUDA generator, and 63 attention/SwiGLU matrices covering
44,605,440 parameters. Embedding/output, RMSNorm, and all other tensors stayed
dense AdamW; the full AdamW delta including decoupled weight decay was masked.

## Terminal results

| Arm | General | Edu | Combined | PPL | Mean tok/s | Final tok/s | Wall seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Control | 3.526235818862915 | 3.2133454084396362 | 3.3697906136512756 | 29.072439048585412 | 84214.58618744368 | 85653.14276260643 | 3593.926469404 |
| Magma | 3.616050601005554 | 3.3080635368824005 | 3.4620570689439774 | 31.882493592417106 | 77959.75480972537 | 78649.54789924077 | 3880.89827551 |

Magma-Control: General `+0.08981478214263916`, Edu `+0.09471812844276428`,
Combined `+0.09226645529270172`; relative combined `+2.738047133223155%`.
Throughput `-7.427254185867982%`; wall time `+7.984910335508033%`.
Capability FAIL; General guard FAIL; Edu guard FAIL; efficiency PASS.

## Integrity and diagnostics

Both runs completed 9,156 updates, 300,023,808 tokens, cursor 585,984, final
General/Edu validation, and terminal checkpoints; all losses were finite.
Neither resumed, crashed, invoked a benchmark, nor had CUDA/thermal aborts.
Control ran 2026-08-30T15:52:30Z-16:52:30Z; peak GPU 70 C / 102.70 W and peak
allocated/reserved 7,876,458,496 / 8,680,112,128 bytes. Its terminal SHA256:
`9c22e978e96cd5d67a97cda3f755278b29cb42d2e6a309f62231222cc71e7cb2`.
Magma ran 2026-08-30T17:11:19Z-18:16:06Z; peak GPU 71 C / 169.57 W with the
same peak memory. Terminal SHA256:
`2e5dbb361340ffd46eb3da518bbf33be8b1b076cbab89d90d834aaa0f7f1f9bd`.

The first temporary gate expected `end`/`edu_end`; production correctly labels
step 9,156 `milestone`/`edu_milestone` at the 3,052-step boundary. The
label-only predicate was corrected and rerun against the unchanged Control;
this was an orchestration defect, not a run/artifact anomaly.

Magma recorded 576,828 mask draws (=9156x63). Exact scalar replay from the
isolated generator matched the checkpointed terminal state: survival
`0.5002791126644338`, per-block min/median/mean/max
`0.4861293141109655`/`0.5007645259938838`/`0.5002791126644337`/`0.5112494539100043`.
Final alignment EMA min/median/mean/max:
`0.5518477728471778`/`0.5537446737289429`/`0.5535492455911168`/`0.5543828435290787`;
invalid values: 0.
