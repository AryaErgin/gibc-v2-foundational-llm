# Project Plan

The project follows a baseline-first experimental sequence: infrastructure validation, baseline measurement, controlled interventions, final training, evaluation, and submission documentation.

Current stage: EXP-009 review complete. Near-Cap Architecture Recipe v3 remains the 49,860,480-parameter SwiGLU allocation; its selected schedule remains peak/min 6e-4/6e-5 after the controlled calibration tie. Native Windows, OMEN Performance mode, and AC power remain frozen as the standard environment; the production path remains 32 sequences x 2 accumulation / 32,768 prediction tokens per update. Do not begin official benchmarks or any >300M follow-up until research review.
