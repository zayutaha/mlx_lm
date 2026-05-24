import sys, os
import math
sys.path.insert(0, '.')
import mlx.core as mx
from mlx_lm.utils import load
from mlx_lm.models.cache import make_prompt_cache

# Load model and prepare MTP environment
model_path = os.path.expanduser('~/.omlx/models/Qwwwon')
model, tokenizer = load(model_path)
input_ids = mx.array([[1, 2, 3]], dtype=mx.uint32)

# Correct cache usage
cache = make_prompt_cache(model)
mtp_cache = model.make_mtp_cache()

# Generate main token
logits, hidden = model(input_ids, cache=cache, return_hidden=True)
next_token = mx.argmax(logits[:, -1, :], axis=-1)

# Generate draft token with MTP head
# Note: hidden here is (B, L, H), mtp_forward needs (B, 1, H)
mtp_logits = model.mtp_forward(hidden[:, -1:, :], next_token[:, None], mtp_cache)
draft_token = mx.argmax(mtp_logits, axis=-1)

print(f"Main token: {next_token.item()}")
print(f"Draft token: {draft_token.item()}")
