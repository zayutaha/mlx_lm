import time, sys, os
sys.path.insert(0, '.')
from mlx_lm.utils import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.generate import generate_step
import mlx.core as mx

model_path = os.path.expanduser('~/.omlx/models/Qwwwon')
print('Loading model...')
model, tokenizer = load(model_path)

text = "The theory of quantum mechanics is a fundamental theory in physics " * 100
ids = tokenizer.encode(text, return_tensors='mlx')[0]
print(f'Prompt length: {len(ids)}')

def run_bench(label, **kwargs):
    cache = make_prompt_cache(model, **kwargs)
    t0 = time.time()
    pret = ids
    t1 = 0
    for i, (tok, _) in enumerate(generate_step(pret, model, prompt_cache=cache)):
        if i == 0:
            t1 = time.time()
        if i >= 9:
            break
    t2 = time.time()
    prefill = t1 - t0
    decode = t2 - t1
    print(f'{label}: prefill={prefill:.3f}s, decode={decode:.3f}s, tok/s={10/(prefill+decode):.1f}')
    return prefill, decode

base_prefill, base_decode = run_bench('BASELINE')
tq_prefill, tq_decode = run_bench('TURBO 3-bit', turbo_kv_bits=3)

print(f'\nPeak GPU: {mx.metal.get_peak_memory()/1e9:.2f} GB')
print(f'Active: {mx.metal.get_active_memory()/1e9:.2f} GB')
