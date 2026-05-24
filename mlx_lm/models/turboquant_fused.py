"""Fused decode attention: reads from TurboQuant packed storage directly."""

import math
from functools import lru_cache
from typing import Optional

import mlx.core as mx


def _metal_available() -> bool:
    return hasattr(mx, "metal") and mx.metal.is_available()


@lru_cache(maxsize=None)
def _fused_decode_single_kernel(bits: int):
    """Fused decode: one threadgroup per q_head, head_dim threads, single pass."""
    if not _metal_available() or bits <= 0:
        return None

    val_mask = (1 << bits) - 1

    source = f"""
        auto lid = thread_position_in_threadgroup.x;
        auto gid = threadgroup_position_in_grid.x;

        auto head_dim = dims[0];
        auto packed_dim = dims[1];
        auto n_repeats = dims[2];
        auto seq_len = dims[3];
        auto n_kv_heads = dims[4];

        auto q_head = gid;
        auto kv_head = q_head / n_repeats;

        // Bit offset for this element in packed storage
        auto bo = lid * {bits};
        auto word_idx = bo / 32;
        auto offset = bo % 32;
        bool spills = (offset + {bits}) > 32;
        constexpr uint val_mask = {val_mask}u;

        // Load query element
        float q_val = (lid < head_dim) ? queries[q_head * head_dim + lid] : 0.0f;

        // Online softmax state
        float max_score = -INFINITY;
        float sum_exp = 0.0f;
        float acc = 0.0f;

        // Shared for WHT butterfly + cross-SIMD reduction
        threadgroup float shared[1024];
        threadgroup float tg_partial[32];

        for (uint t = 0; t < seq_len; t++) {{
            // Dequantize key
            auto k_base = (kv_head * seq_len + t) * packed_dim;
            uint kw = k_packed[k_base + word_idx];
            uint kidx = (kw >> offset) & val_mask;
            if (spills) {{
                kidx = ((kw >> offset) | (k_packed[k_base + word_idx + 1] << ({bits} - offset))) & val_mask;
            }}
            float k_val = (lid < head_dim) ? centroids[kidx] : 0.0f;

            // WHT butterfly in shared memory
            shared[lid] = k_val;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = 1; stride < head_dim; stride <<= 1) {{
                auto pair = lid ^ stride;
                float a = shared[min(lid, pair)];
                float b = shared[max(lid, pair)];
                shared[lid] = (lid < pair) ? (a + b) : (a - b);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}
            float kn = k_norms[kv_head * seq_len + t];
            float k_deq = (lid < head_dim) ? shared[lid] * (1.0f / sqrt((float)head_dim)) * signs[lid] * kn : 0.0f;

            // Cross-SIMD score reduction
            auto sg_id = lid / 32;
            auto sg_lid = lid % 32;
            float ps = simd_sum(q_val * k_deq);
            if (sg_lid == 0) tg_partial[sg_id] = ps;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float score = 0.0f;
            if (sg_id == 0) {{
                auto num_sg = (head_dim + 31) / 32;
                for (int i = sg_lid; i < num_sg; i += 32)
                    score += tg_partial[i];
                score = simd_sum(score);
                tg_partial[0] = score;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            score = tg_partial[0];

            // Online softmax
            float new_max = max(max_score, score);
            float factor = exp(max_score - new_max);
            float exp_score = exp(score - new_max);
            max_score = new_max;
            sum_exp = sum_exp * factor + exp_score;

            // Dequantize value
            uint vw = v_packed[k_base + word_idx];
            uint vidx = (vw >> offset) & val_mask;
            if (spills) {{
                vidx = ((vw >> offset) | (v_packed[k_base + word_idx + 1] << ({bits} - offset))) & val_mask;
            }}
            float v_val = (lid < head_dim) ? centroids[vidx] * v_norms[kv_head * seq_len + t] : 0.0f;

            acc = acc * factor + exp_score * v_val;
        }}

        if (lid < head_dim) {{
            out[q_head * head_dim + lid] = acc / max(sum_exp, 1e-10f);
        }}
    """
    return mx.fast.metal_kernel(
        name=f"tq_fused_single_{bits}",
        input_names=["queries", "k_packed", "k_norms", "v_packed", "v_norms",
                      "centroids", "signs", "dims"],
        output_names=["out"],
        source=source,
    )


@lru_cache(maxsize=None)
def _fused_decode_batched_kernel(bits: int):
    """Batched: one threadgroup per kv_head, dequant key once for all repeats."""
    if not _metal_available() or bits <= 0:
        return None

    val_mask = (1 << bits) - 1

    source = f"""
        auto lid = thread_position_in_threadgroup.x;
        auto gid = threadgroup_position_in_grid.x;

        auto head_dim = dims[0];
        auto packed_dim = dims[1];
        auto n_repeats = dims[2];
        auto seq_len = dims[3];
        auto n_kv_heads = dims[4];
        auto kv_head = gid;

        auto bo = lid * {bits};
        auto word_idx = bo / 32;
        auto offset = bo % 32;
        bool spills = (offset + {bits}) > 32;
        constexpr uint val_mask = {val_mask}u;

        // Load all query replicas
        float q_vals[32];
        for (uint r = 0; r < n_repeats; r++) {{
            q_vals[r] = (lid < head_dim) ? queries[(kv_head * n_repeats + r) * head_dim + lid] : 0.0f;
        }}

        float max_scores[32];
        float sum_exps[32];
        float accs[32];
        for (uint r = 0; r < n_repeats; r++) {{
            max_scores[r] = -INFINITY;
            sum_exps[r] = 0.0f;
            accs[r] = 0.0f;
        }}

        threadgroup float shared[1024];
        threadgroup float tg_partial[32];

        for (uint t = 0; t < seq_len; t++) {{
            // Dequantize key ONCE
            auto k_base = (kv_head * seq_len + t) * packed_dim;
            uint kw = k_packed[k_base + word_idx];
            uint kidx = (kw >> offset) & val_mask;
            if (spills) {{
                kidx = ((kw >> offset) | (k_packed[k_base + word_idx + 1] << ({bits} - offset))) & val_mask;
            }}
            float k_val = (lid < head_dim) ? centroids[kidx] : 0.0f;

            shared[lid] = k_val;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = 1; stride < head_dim; stride <<= 1) {{
                auto pair = lid ^ stride;
                float a = shared[min(lid, pair)];
                float b = shared[max(lid, pair)];
                shared[lid] = (lid < pair) ? (a + b) : (a - b);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}
            float kn = k_norms[kv_head * seq_len + t];
            float k_deq = (lid < head_dim) ? shared[lid] * (1.0f / sqrt((float)head_dim)) * signs[lid] * kn : 0.0f;

            // Dequantize value ONCE
            uint vw = v_packed[k_base + word_idx];
            uint vidx = (vw >> offset) & val_mask;
            if (spills) {{
                vidx = ((vw >> offset) | (v_packed[k_base + word_idx + 1] << ({bits} - offset))) & val_mask;
            }}
            float v_val = (lid < head_dim) ? centroids[vidx] * v_norms[kv_head * seq_len + t] : 0.0f;

            // Cross-SIMD reduce, then per-repeat online softmax
            auto sg_id = lid / 32;
            auto sg_lid = lid % 32;

            for (uint r = 0; r < n_repeats; r++) {{
                float ps = simd_sum(q_vals[r] * k_deq);
                if (sg_lid == 0) tg_partial[sg_id] = ps;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float score = 0.0f;
                if (sg_id == 0) {{
                    auto num_sg = (head_dim + 31) / 32;
                    for (int i = sg_lid; i < num_sg; i += 32)
                        score += tg_partial[i];
                    score = simd_sum(score);
                    tg_partial[0] = score;
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
                score = tg_partial[0];

                float new_max = max(max_scores[r], score);
                float factor = exp(max_scores[r] - new_max);
                float exp_score = exp(score - new_max);
                max_scores[r] = new_max;
                sum_exps[r] = sum_exps[r] * factor + exp_score;
                accs[r] = accs[r] * factor + exp_score * v_val;
            }}
        }}

        if (lid < head_dim) {{
            for (uint r = 0; r < n_repeats; r++) {{
                out[(kv_head * n_repeats + r) * head_dim + lid] = accs[r] / max(sum_exps[r], 1e-10f);
            }}
        }}
    """
    return mx.fast.metal_kernel(
        name=f"tq_fused_batched_{bits}",
        input_names=["queries", "k_packed", "k_norms", "v_packed", "v_norms",
                      "centroids", "signs", "dims"],
        output_names=["out"],
        source=source,
    )


def fused_decode_attention(
    queries: mx.array,
    k_packed: mx.array,
    k_norms: mx.array,
    v_packed: mx.array,
    v_norms: mx.array,
    centroids: mx.array,
    signs: mx.array,
    head_dim: int,
    packed_dim: int,
    n_repeats: int,
    n_kv_heads: int,
    bits: int,
) -> Optional[mx.array]:
    """Fused decode attention from packed TurboQuant state.

    Args:
        queries: (1, n_q_heads, 1, head_dim) float32, already RoPE'd
        k_packed: (1, n_kv_heads, seq_len, packed_dim) uint32
        k_norms: (1, n_kv_heads, seq_len) float32
        v_packed/v_norms: same as K
        centroids: (1<<bits,) float32 codebook
        signs: (head_dim,) float32 RHT signs
        head_dim: head dimension
        packed_dim: packed uint32 dimension
        n_repeats: GQA repeats (n_q_heads // n_kv_heads)
        n_kv_heads: number of KV heads
        bits: quantization bits

    Returns:
        (1, n_q_heads, 1, head_dim) or None if Metal unavailable
    """
    B, n_q_heads, S, D = queries.shape
    if B != 1 or S != 1:
        return None

    seq_len = k_norms.shape[-1]
    dims = mx.array([head_dim, packed_dim, n_repeats, seq_len, n_kv_heads], dtype=mx.uint32)

    # Try batched kernel (GQA, dequant key once)
    if n_repeats > 1 and n_repeats <= 32:
        kernel = _fused_decode_batched_kernel(bits)
        if kernel is not None:
            out = kernel(
                inputs=[queries.reshape(n_q_heads, D),
                        k_packed.reshape(n_kv_heads, seq_len, packed_dim),
                        k_norms.reshape(n_kv_heads, seq_len),
                        v_packed.reshape(n_kv_heads, seq_len, packed_dim),
                        v_norms.reshape(n_kv_heads, seq_len),
                        centroids, signs, dims],
                grid=(n_kv_heads, 1, 1),
                threadgroup=(head_dim, 1, 1),
                output_shapes=[(n_q_heads, head_dim)],
                output_dtypes=[mx.float32],
            )
            return out[0].reshape(B, n_q_heads, 1, D)

    # Fallback to single kernel
    kernel = _fused_decode_single_kernel(bits)
    if kernel is None:
        return None

    out = kernel(
        inputs=[queries.reshape(n_q_heads, D),
                k_packed.reshape(n_kv_heads, seq_len, packed_dim),
                k_norms.reshape(n_kv_heads, seq_len),
                v_packed.reshape(n_kv_heads, seq_len, packed_dim),
                v_norms.reshape(n_kv_heads, seq_len),
                centroids, signs, dims],
        grid=(n_q_heads, 1, 1),
        threadgroup=(head_dim, 1, 1),
        output_shapes=[(n_q_heads, head_dim)],
        output_dtypes=[mx.float32],
    )
    return out[0].reshape(B, n_q_heads, 1, D)
