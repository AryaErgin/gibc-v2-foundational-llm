# EXP-001 Architecture

Decoder-only causal Transformer: vocab 8192, d_model 256, 8 blocks, 8 heads of 32 dimensions, d_ff 1024, exact GELU, pre-RMSNorm, no bias/dropout, tied embedding/output, and context 512. RMSNorm is `x * rsqrt(mean(x^2)+1e-5) * scale`; each scale is the only norm parameter. RoPE uses theta 10000, all 32 head dimensions, adjacent-pair rotation, and no scaling/trainable parameters. SDPA uses `is_causal=True`.

Parameter mapping in `src/gibc_llm/model.py`: embedding `8192*256=2,097,152`; attention `8*4*256*256=2,097,152`; MLP `8*(256*1024+1024*256)=4,194,304`; norm scales `8*2*256+256=4,352`; tied output additional `0`; total **8,392,960**. Initialization is Normal(0,0.02) for embeddings/linears and 1 for norm scales under seed 42.

## Near-Cap Architecture Recipe v2

EXP-007 freezes the production near-cap recipe as EXP-007B: vocabulary 8,192; d_model 640; 9 decoder blocks; 20 heads with head dimension and rotary dimension 32; d_ff 2,560; exact GELU; pre-RMSNorm eps 1e-5; standard causal SDPA; RoPE theta 10,000; tied input/output embeddings; no linear bias; dropout zero; context 512; initialization Normal(0,0.02); seed 42. The exact trainable parameter count is **49,491,840**, including all norm scales and the tied output treatment.

EXP-007A's final combined validation loss was numerically lower, but B-A was only +0.0056269169 nats, inside the predeclared 0.02-nat engineering-tie region. Recipe v2 therefore selects EXP-007B using its higher throughput and lower allocated-memory measurements, not a claim of lower validation loss.
