import re
path = 'mlx_lm/generate.py'
with open(path, 'r') as f:
    content = f.read()

# Fix the mess
pattern = r'if mtp and hasattr\(model, "mtp_forward"\):.*?else:\s+logits = logits\[:, -1, :\]'
new_content = re.sub(pattern, 'logits = logits[:, -1, :]', content, flags=re.DOTALL)
with open(path, 'w') as f:
    f.write(new_content)
