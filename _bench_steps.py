"""Benchmark: 256 vs 512 step size at 64k and 128k context. TurboQuant 3-bit."""
import gc, time, sys
from pathlib import Path
import mlx.core as mx
from mlx_lm.utils import load_model
from mlx_lm.generate import generate_step
from mlx_lm.models import cache

MODEL = Path.home() / '.omlx/models/Qwwwon'
CTX_SIZES = [65536, 131072]
STEP_SIZES = [256, 512]
GEN_TOKENS = 3  # fewer gen tokens to save time

print(f"{'Context':>8} | {'Step':>5} | {'tok/s':>7} | {'Time':>8} | {'Peak':>8} | {'Overhead':>9}")
print("-" * 57)

for ctx in CTX_SIZES:
    model, config = load_model(MODEL, lazy=False)
    baseline = mx.get_active_memory() / 1e9

    tokens = list(range(ctx))
    prompt = mx.array(tokens, mx.uint32)

    for step in STEP_SIZES:
        c = cache.make_prompt_cache(model, turbo_kv_bits=3)
        gc.collect(); mx.clear_cache(); mx.reset_peak_memory()

        t0 = time.perf_counter()
        for tok, lp in generate_step(prompt, model, max_tokens=GEN_TOKENS,
                                      prompt_cache=c, prefill_step_size=step):
            pass
        elapsed = time.perf_counter() - t0

        peak_gb = mx.get_peak_memory() / 1e9
        tok_s = ctx / elapsed
        overhead = peak_gb - baseline

        print(f"{ctx:>8} | {step:>5} | {tok_s:>7.0f} | {elapsed:>8.1f}s | {peak_gb:>8.2f}GB | {overhead:>+8.2f}GB")
        sys.stdout.flush()
        del c; gc.collect()

    del model, prompt, tokens; gc.collect()
