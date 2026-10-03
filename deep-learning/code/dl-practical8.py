# One width-64 GPT block predicts Shakespeare characters from a 256-byte context.
# Set STUDENT_ID, then run this file or paste it into one notebook cell.
import ast
import hashlib
import json
import math
import random
import re
import zipfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

STUDENT_ID = "studentID"
DATA_DIR = Path("data")

# %% Model and training settings

CONFIG = dict(task="gpt", steps=2000, batch_size=32,
              learning_rate=0.003, weight_decay=0.0)


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3*width, bias=False)
        self.output = nn.Linear(width, width)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [a.reshape(batch, length, self.heads, width//self.heads)
                   .transpose(1, 2) for a in (q, k, v)]
        h = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.output(h.transpose(1, 2).reshape(batch, length, width))


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attention = Attention(width, heads)
        self.mlp = nn.Sequential(nn.Linear(width, 4*width), nn.GELU(),
                                 nn.Linear(4*width, width))

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class Model(nn.Module):
    def __init__(self, width=64, heads=2, layers=1):
        super().__init__()
        self.token = nn.Embedding(256, width)
        self.position = nn.Embedding(256, width)
        self.blocks = nn.Sequential(*(Block(width, heads) for _ in range(layers)))
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 256)

    def forward(self, tokens):
        length = tokens.size(1)
        position = torch.arange(length, device=tokens.device)
        h = self.token(tokens) + self.position(position)
        return self.head(self.norm(self.blocks(h)))


def build_model():
    return Model()

MAX_PARAMETERS, MAX_STEPS, MAX_BATCH = (1000000, 10000, 64)

# %% Data, evaluation and submission
def program_source():
    if "__file__" in globals():
        source = Path(__file__).read_text()
    else:
        source = get_ipython().history_manager.input_hist_raw[-1].rstrip() + "\n"
    tree = ast.parse(source)
    names = {getattr(node, "name", "") for node in tree.body}
    if not {"build_model", "run", "save_submission"} <= names:
        raise ValueError("Paste the complete program into one notebook cell and run that cell.")
    return source


def model_size(model):
    parameters = sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters())
    buffers = sum(b.numel()*(2 if b.is_complex() else 1) for b in model.buffers())
    if parameters + buffers > MAX_PARAMETERS:
        raise ValueError(f"Model has {parameters + buffers} values; limit is {MAX_PARAMETERS}.")
    return parameters, buffers


def save_submission(folder, source, history, metrics, size, seed, split):
    stem = "lab-08"
    code = folder / (stem + "-code.py")
    code.write_text(source)
    names = [f"batch-{i:02d}.txt" for i in range(1, 11)]
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    updates = {"model": CONFIG["steps"]}
    result = dict(format=6, lab=8, seed=seed, config=dict(CONFIG),
                  parameters=size[0], buffer_elements=size[1], updates=updates,
                  protocol="comp3801-gpt-context256-v1-" + split, code_sha256=digest(code),
                  metrics=metrics, loss=history,
                  samples_sha256={name: digest(folder/"samples"/name) for name in names})
    record = folder / (stem + "-results.json")
    record.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    archive = folder.with_name(folder.name + ".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as pack:
        for path in [code, record] + [folder/"samples"/name for name in names]:
            pack.write(path, path.relative_to(folder))
    print(json.dumps(metrics, indent=2))
    print("Submission:", archive.resolve())
    return result


def show_results(history, folder):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 2.5))
    curves = np.array(history)
    plt.plot(curves[:, 0], curves[:, 1])
    plt.xlabel("Update"); plt.ylabel("Loss"); plt.tight_layout(); plt.show()
    print((folder/"samples/batch-01.txt").read_text(encoding="utf-8", errors="replace"))


def download(url, path, expected):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        from urllib.request import urlretrieve
        temporary = path.with_suffix(".download")
        try:
            urlretrieve(url, temporary)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected:
                raise ValueError("Download checksum differs: " + str(path))
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Cached file checksum differs: " + str(path))
    return path

def text_data(split="train"):
    path = download("https://raw.githubusercontent.com/cwkx/research-data/main/datasets/shakespeare.txt",
                    DATA_DIR/"tinyshakespeare.txt",
                    "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed")
    data = path.read_bytes()
    a, b = int(.8*len(data)), int(.9*len(data))
    parts = {"train": data[:a], "validation": data[a:b], "test": data[b:]}
    return torch.tensor(list(parts[split]), dtype=torch.long)

def text_batch(tokens, batch, context=256, generator=None):
    if len(tokens) <= context:
        raise ValueError('text split is too short')
    starts = torch.randint(len(tokens)-context, (batch,), device=tokens.device, generator=generator)
    positions = torch.arange(context+1, device=tokens.device)
    windows = tokens[starts[:, None] + positions]
    return windows[:, :-1], windows[:, 1:]

@torch.no_grad()
def language_metrics(model, tokens, device='cpu', batch=32, context=256):
    """Evaluate next-byte loss in non-overlapping windows, counting each target once."""
    model.eval()
    x_windows, y_windows = [], []
    total, count = 0.0, 0
    for start in range(0, len(tokens)-1, context):
        end = min(start+context, len(tokens)-1)
        x_windows.append(tokens[start:end])
        y_windows.append(tokens[start+1:end+1])
        # Group windows by length so the shorter final window is included in the score.
        if len(x_windows) == batch or end == len(tokens)-1:
            groups = {}
            for x, y in zip(x_windows, y_windows):
                groups.setdefault(len(x), []).append((x, y))
            for length, group in groups.items():
                x = torch.stack([p[0] for p in group]).to(device)
                y = torch.stack([p[1] for p in group]).to(device)
                logits = model(x)
                total += float(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction='sum'))
                count += y.numel()
            x_windows, y_windows = [], []
    if not count:
        raise ValueError('at least two tokens required')
    nll = total/count
    return {'nll_nats_per_byte': nll, 'bits_per_byte': nll/math.log(2),
            'perplexity': math.exp(nll), 'target_bytes': count, 'context': context}

# %% Training
def train(model, device):
    tokens = text_data().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"],
                                  weight_decay=CONFIG["weight_decay"])
    history = []
    model.train()
    for step in range(1, CONFIG["steps"] + 1):
        x, y = text_batch(tokens, CONFIG["batch_size"], context=256)
        x, y = x.to(device), y.to(device)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        if step == 1 or step % max(1, CONFIG["steps"]//40) == 0 or step == CONFIG["steps"]:
            history.append([step, round(loss.item(), 6)])
        if step == 1 or step % max(1, CONFIG["steps"]//10) == 0 or step == CONFIG["steps"]:
            print(f"{step} / {CONFIG['steps']}: loss {loss.item():.4f}", flush=True)
    return history

@torch.no_grad()
def generate(model, tokens, length=512, temperature=.8, generator=None):
    model.eval()
    for _ in range(length-tokens.size(1)):
        logits = model(tokens[:, -256:])[:, -1] / temperature
        next_token = torch.multinomial(logits.softmax(-1), 1, generator=generator)
        tokens = torch.cat([tokens, next_token], dim=1)
    return tokens


@torch.no_grad()
def text_score(model, folder, device, split):
    model.eval()
    scores = language_metrics(model, text_data(split), device, batch=32, context=256)
    metrics = {key: scores[key] for key in ("nll_nats_per_byte", "target_bytes")}
    rng = torch.Generator(device=device).manual_seed(921)
    prefixes = [b"ROMEO:\n ", b"JULIET:\n", b"First Ci", b"KING:\n  "]*2
    for batch in range(10):
        tokens = torch.tensor([list(prefix) for prefix in prefixes], device=device)
        tokens = generate(model, tokens, length=512, temperature=.8, generator=rng)
        samples = [f"=== Sample {i+1:02d} ===\n".encode() + bytes(row.tolist())
                   for i, row in enumerate(tokens)]
        (folder/f"batch-{batch+1:02d}.txt").write_bytes(b"\n\n".join(samples) + b"\n")
    return metrics

# %% Run the experiment
def run(out=None, device=None, seed=3801, split="validation", show=True):
    source = program_source()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", STUDENT_ID):
        raise ValueError("Use letters, digits, underscore or hyphen for STUDENT_ID.")
    if CONFIG["task"] != "gpt":
        raise ValueError("Keep task='gpt' in CONFIG.")
    if not 1 <= CONFIG["steps"] <= MAX_STEPS or not 1 <= CONFIG["batch_size"] <= MAX_BATCH:
        raise ValueError("Training steps or batch size exceeds the practical limit.")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    folder = Path(out or f"lab-08-{STUDENT_ID}")
    (folder/"samples").mkdir(parents=True, exist_ok=True)
    model = build_model().to(device)
    size = model_size(model)
    history = train(model, device)
    metrics = text_score(model, folder/"samples", device, split)
    result = save_submission(folder, source, history, metrics, size, seed, split)
    if show:
        show_results(history, folder)
    return result


if __name__ == "__main__":
    run()
