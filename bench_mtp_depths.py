import time, os
import mlx.core as mx
from mlx_lm.utils import load
from mlx_lm.generate import stream_generate

model_path = os.path.expanduser('~/.omlx/models/Qwwwon')
model, tokenizer = load(model_path)

prompts = {
    "code": "def fibonacci(n):",
    "technical": "Quantum entanglement is a phenomenon where",
    "creative": "The stars shone brightly over the"
}

def bench(prompt, k):
    ids = tokenizer.encode(prompt, return_tensors='mlx')[0]
    # Warmup
    for _ in stream_generate(model, tokenizer, ids, max_tokens=10, mtp=True, num_draft_tokens=k): pass
    
    t0 = time.time()
    count = 0
    for resp in stream_generate(model, tokenizer, ids, max_tokens=50, mtp=True, num_draft_tokens=k):
        count += 1
    t1 = time.time()
    return count / (t1 - t0)

for p_name, p_text in prompts.items():
    for k in [1, 2, 3]:
        speed = bench(p_text, k)
        print(f"Prompt: {p_name}, k={k}, Speed: {speed:.2f} tok/s")
