# EXP-001 Architecture

Decoder-only causal Transformer: vocab 8192, d_model 256, 8 blocks, 8 heads of 32 dimensions, d_ff 1024, exact GELU, pre-RMSNorm, no bias/dropout, tied embedding/output, and context 512. RMSNorm is `x * rsqrt(mean(x^2)+1e-5) * scale`; each scale is the only norm parameter. RoPE uses theta 10000, all 32 head dimensions, adjacent-pair rotation, and no scaling/trainable parameters. SDPA uses `is_causal=True`.

Parameter mapping in `src/gibc_llm/model.py`: embedding `8192*256=2,097,152`; attention `8*4*256*256=2,097,152`; MLP `8*(256*1024+1024*256)=4,194,304`; norm scales `8*2*256+256=4,352`; tied output additional `0`; total **8,392,960**. Initialization is Normal(0,0.02) for embeddings/linears and 1 for norm scales under seed 42.
