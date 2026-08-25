# Project Plan

The project follows a baseline-first experimental sequence: infrastructure validation, baseline measurement, controlled interventions, final training, evaluation, and submission documentation.

Current stage: EXP-009 review complete. Near-Cap Architecture Recipe v3 remains the 49,860,480-parameter SwiGLU allocation; its selected schedule remains peak/min 6e-4/6e-5 after the controlled calibration tie. Native Windows, OMEN Performance mode, and AC power remain frozen as the standard environment; the production path remains 32 sequences x 2 accumulation / 32,768 prediction tokens per update. Do not begin official benchmarks or any >300M follow-up until research review.

EXP-010 is prepared only as a predeclared SwiGLU depth/width allocation test. Existing governance does not authorize its 60-update preflight or its 300M candidate run. No official benchmark or any additional training may start without new research authorization.

EXP-010 subsequently completed under research authorization and retained Recipe v3 by its predeclared engineering tiebreak. Conditional EXP-011 authorization selects Recipe v3 only: fresh seed-42 initialization, a 45,777-step cosine schedule from step zero, and a hard ceiling of 1,500,020,736 prediction tokens. No official benchmark is authorized.

EXP-011 preflight and fresh-process resume are complete and valid. The current authorized phase is fresh Recipe v3 training through the immutable verified EXP-006 900M prefix; continuation beyond step 27,468 requires a separately prepared 1.5B stream whose raw 900M prefix matches exactly.
