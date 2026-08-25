# Project Plan

The project follows a baseline-first experimental sequence: infrastructure validation, baseline measurement, controlled interventions, final training, evaluation, and submission documentation.

Current stage: EXP-009 learning-rate calibration. Near-Cap Architecture Recipe v3 is frozen as the 49,860,480-parameter SwiGLU allocation. Native Windows, OMEN Performance mode, and AC power remain frozen as the standard environment; the production path remains 32 sequences x 2 accumulation / 32,768 prediction tokens per update. Only the declared 4e-4/4e-5 and 8e-4/8e-5 schedule amplitudes may vary from the existing 6e-4/6e-5 EXP-008A control. No official benchmark or training beyond the two authorized 300M candidates may start before review.
