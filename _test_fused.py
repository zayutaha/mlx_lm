import sys, os
sys.path.insert(0, '.')
import mlx.core as mx
from mlx_lm.models.turboquant_cache import TurboQuantKVCache
from mlx_lm.models.turboquant_fused import fused_decode_attention
from mlx_lm.models.turboquant_packing import packed_dim
from mlx_lm.models.turboquant_kernels import packed_dequantize

bits = 3
head_dim = 64
n_kv = 4
n_q = 24
n_repeats = n_q // n_kv
pack_dim = packed_dim(head_dim, bits)
n_tokens = 100

cache = TurboQuantKVCache(bits=bits, seed=42)
B = 1
keys = mx.random.normal(shape=(B, n_kv, n_tokens, head_dim))
values = mx.random.normal(shape=(B, n_kv, n_tokens, head_dim))
k_fp16, v_fp16 = cache.update_and_fetch(keys, values)

# Get packed state
state = cache.state
k_packed, k_norms, v_packed, v_norms = state
print(f'K packed: {k_packed.shape}')

# Get correct signs and centroids from cache internals
k_signs = cache._k_q.signs
v_signs = cache._v_q.signs
centroids = cache._k_q.centroids  # same for both
print(f'Centroids: {centroids.tolist()}')

query = mx.random.normal(shape=(B, n_q, 1, head_dim))

# Reference: dequant using cache's own dequantize() method  
k_deq_ref, v_deq_ref = cache.dequantize()
k_deq_ref = k_deq_ref.reshape(1, n_kv, n_tokens, head_dim)
v_deq_ref = v_deq_ref.reshape(1, n_kv, n_tokens, head_dim)
scale = head_dim ** -0.5
ref = mx.fast.scaled_dot_product_attention(query, k_deq_ref, v_deq_ref, scale=scale, mask=None)

# Fused: need to use CACHE's centroids + signs, batch identical
# The fused kernel uses the same signs for K and V, but cache uses different ones
# For now, test with K signs (or better, fix the kernel to accept separate K/V signs)
# Actually, the centroids are the same for both - the difference is in signs
# Let's test with the V signs to see the impact
out_k = fused_decode_attention(
    query, k_packed, k_norms, v_packed, v_norms,
    centroids, k_signs, head_dim, pack_dim, n_repeats, n_kv, bits
)

mx.eval(ref, out_k)
diff = mx.abs(ref - out_k).max().item()
cos_sim = mx.sum(ref * out_k) / (mx.linalg.norm(ref) * mx.linalg.norm(out_k) + 1e-10)
print(f'Using K signs - Max diff: {diff:.6f}, Cos sim: {cos_sim:.6f}')

# Now try with the actual packed dequant to isolate the kernel vs dequant issue
k_deq = packed_dequantize(k_packed.reshape(-1, pack_dim), k_norms.reshape(-1), centroids, k_signs, head_dim, bits)
v_deq = packed_dequantize(v_packed.reshape(-1, pack_dim), v_norms.reshape(-1), centroids, v_signs, head_dim, bits)
k_deq = k_deq.reshape(1, n_kv, n_tokens, head_dim)
v_deq = v_deq.reshape(1, n_kv, n_tokens, head_dim)
ref2 = mx.fast.scaled_dot_product_attention(query, k_deq, v_deq, scale=scale, mask=None)

mx.eval(ref2, out_k)
diff2 = mx.abs(ref2 - out_k).max().item()
cos2 = mx.sum(ref2 * out_k) / (mx.linalg.norm(ref2) * mx.linalg.norm(out_k) + 1e-10)
print(f'vs packed_dequantize - Max diff: {diff2:.6f}, Cos sim: {cos2:.6f}')
if diff2 < 0.05:
    print('PASS: kernel matches packed_dequantize')
else:
    print('FAIL: kernel differs from packed_dequantize')
    print(f'Ref[:4]: {ref2.flatten()[:4]}')
    print(f'Out[:4]: {out_k.flatten()[:4]}')
