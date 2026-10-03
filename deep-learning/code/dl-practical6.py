# Small RGB U-Net for CIFAR-10 cars, using time-conditioned residual blocks for DDPM or image flow.
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
from torch_fidelity import calculate_metrics  # pip install torch-fidelity
from torch.nn import functional as F

STUDENT_ID = "studentID"
DATA_DIR = Path("data")

# %% Model and training settings

CONFIG = dict(task="ddpm", steps=1000, batch_size=64,
              learning_rate=0.001, weight_decay=0.0)


class Residual(nn.Module):
    def __init__(self, width, condition=64):
        super().__init__()
        self.norm1, self.norm2 = nn.GroupNorm(4, width), nn.GroupNorm(4, width)
        self.conv1 = nn.Conv2d(width, width, 3, padding=1)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)
        self.condition = nn.Linear(condition, 2 * width)

    def forward(self, x, condition):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.condition(condition).chunk(2, -1)
        h = self.norm2(h) * (1 + scale[:, :, None, None])
        h = h + shift[:, :, None, None]
        return x + self.conv2(F.silu(h))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.time = nn.Sequential(nn.Linear(17, 64), nn.SiLU(), nn.Linear(64, 64))
        self.label = nn.Embedding(1, 64)
        self.input = nn.Conv2d(3, 24, 3, padding=1)
        self.first = Residual(24)
        self.down = nn.Conv2d(24, 48, 4, 2, 1)
        self.middle = Residual(48)
        self.up = nn.ConvTranspose2d(48, 24, 4, 2, 1)
        self.merge = nn.Conv2d(48, 24, 1)
        self.last = Residual(24)
        self.out = nn.Conv2d(24, 3, 3, padding=1)

    def forward(self, x, t, y=None):
        frequencies = 2 ** torch.arange(8, device=t.device)
        angles = t[:, None] * frequencies[None] * math.pi
        e = self.time(torch.cat([t[:, None], angles.sin(), angles.cos()], -1))
        if y is not None:
            e = e + self.label(y)
        h = self.first(self.input(x), e)
        low = self.middle(self.down(h), e)
        up = self.up(low)
        return self.out(self.last(self.merge(torch.cat([h, up], 1)), e))


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
    stem = "lab-06"
    code = folder / (stem + "-code.py")
    code.write_text(source)
    names = [f"batch-{i:02d}.png" for i in range(1, 11)]
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    updates = {"model": CONFIG["steps"]}
    result = dict(format=6, lab=6, seed=seed, config=dict(CONFIG),
                  parameters=size[0], buffer_elements=size[1], updates=updates,
                  protocol="comp3801-cifar10-cars-kid1000-v1-" + split, code_sha256=digest(code),
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
    from PIL import Image
    plt.figure(figsize=(6, 6))
    plt.imshow(Image.open(folder/"samples/batch-01.png"))
    plt.axis("off"); plt.tight_layout(); plt.show()


def image_data(split="train"):
    from torchvision.datasets import CIFAR10
    dataset = CIFAR10(root=DATA_DIR, train=split != "test", download=True)
    index = torch.where(torch.tensor(dataset.targets) == 1)[0]
    if split != "test":
        order = torch.randperm(len(index), generator=torch.Generator().manual_seed(3802))
        index = index[order[:4000] if split == "train" else order[4000:]]
    images = torch.from_numpy(dataset.data).permute(0, 3, 1, 2)[index].float()/127.5 - 1
    return images, torch.zeros(len(images), dtype=torch.long)


def image_bytes(images):
    return ((images.clamp(-1, 1) + 1)*127.5 + .5).floor().to(torch.uint8)


class Images(torch.utils.data.Dataset):
    """Give the KID library one uint8 RGB image at a time."""
    def __init__(self, pixels): self.pixels = pixels
    def __len__(self): return len(self.pixels)
    def __getitem__(self, index): return self.pixels[index]


@torch.no_grad()
def image_score(model, folder, device, split):
    from PIL import Image
    model.eval()
    rng = torch.Generator(device=device).manual_seed(921)
    fake = torch.cat([generate(model, 50, device, rng).clamp(-1, 1).cpu() for _ in range(20)])
    real, _ = image_data(split)
    pixels = image_bytes(fake)
    # One full set of 1,000 images, using the same pixels as the saved grids.
    metrics = calculate_metrics(
        input1=Images(image_bytes(real)), input2=Images(pixels),
        cuda=torch.device(device).type == "cuda", batch_size=32,
        kid=True, kid_subset_size=1000, kid_subsets=1,
        feature_extractor="inception-v3-compat", feature_layer_kid="2048",
        rng_seed=921, save_cpu_ram=True, verbose=False)
    score = metrics["kernel_inception_distance_mean"]
    for i, batch in enumerate(pixels.split(100), 1):
        grid = batch.reshape(10, 10, 3, 32, 32).permute(0, 3, 1, 4, 2).reshape(320, 320, 3)
        Image.fromarray(grid.numpy()).save(folder/f"batch-{i:02d}.png")
    print(f"KID x 1000: {1000*score:.3f}")
    return {"kid": score, "samples": 1000}


def diffusion_schedule(steps=100, device='cpu'):
    # The cosine schedule approaches pure noise within 100 diffusion steps.
    grid = torch.linspace(0, 1, steps+1, device=device, dtype=torch.float64)
    cumulative = torch.cos((grid+.008)/1.008 * torch.pi/2).square()
    cumulative = cumulative/cumulative[0]
    betas = (1-cumulative[1:]/cumulative[:-1]).clamp(max=.999).float()
    alphas = 1-betas
    bars = torch.cumprod(alphas, dim=0)
    previous = torch.cat([torch.ones(1, device=device), bars[:-1]])
    variance = betas*(1-previous)/(1-bars)
    return betas, alphas, bars, variance

def add_noise(clean, epsilon, t, bars):
    a = bars[t].reshape(-1, 1, 1, 1)
    return a.sqrt()*clean + (1-a).sqrt()*epsilon

def diffusion_loss(model, clean, bars, labels=None):
    t = torch.randint(len(bars), (len(clean),), device=clean.device)
    epsilon = torch.randn_like(clean)
    noisy = add_noise(clean, epsilon, t, bars)
    # Integer index 0 denotes mathematical diffusion step 1.
    prediction = model(noisy, (t.float()+1)/len(bars), labels) if labels is not None else model(noisy, (t.float()+1)/len(bars))
    return F.mse_loss(prediction, epsilon)

def flow_path(noise, data, t):
    t = t.reshape(-1, 1, 1, 1)
    return (1-t)*noise+t*data, data-noise

def flow_loss(model, data, labels):
    noise = torch.randn_like(data)
    t = torch.rand(len(data), device=data.device)
    x, target = flow_path(noise, data, t)
    return F.mse_loss(model(x, t, labels), target)

@torch.no_grad()
def sample_ddpm(model, count, schedule, device='cpu', generator=None, labels=None, shape=(3, 32, 32)):
    betas, alphas, bars, variance = schedule
    x = torch.randn(count, *shape, device=device, generator=generator)
    for i in reversed(range(len(betas))):
        t = torch.full((count,), (i+1)/len(betas), device=device)
        epsilon = model(x, t, labels) if labels is not None else model(x, t)
        # Predict clean x0, clip to the known data support, then use q(x_{t-1}|x_t,x0).
        clean = ((x-(1-bars[i]).sqrt()*epsilon)/bars[i].sqrt()).clamp(-1, 1)
        prev = bars[i-1] if i else torch.ones((), device=device)
        mean = (prev.sqrt()*betas[i]/(1-bars[i]))*clean
        mean = mean + (alphas[i].sqrt()*(1-prev)/(1-bars[i]))*x
        if i:
            noise = torch.randn(x.shape, device=device, generator=generator)
            x = mean + variance[i].sqrt()*noise
        else:
            x = mean
    return x.clamp(-1, 1)

@torch.no_grad()
def sample_flow(model, labels, steps=25, device='cpu', generator=None, shape=(3, 32, 32)):
    if steps < 1:
        raise ValueError('sampling steps must be positive')
    x = torch.randn(len(labels), *shape, device=device, generator=generator)
    dt = 1/steps
    for i in range(steps):
        t = torch.full((len(labels),), i/steps, device=device)
        x = x + dt*model(x, t, labels)
    return x.clamp(-1, 1)

# %% Training
def train(model, device):
    images, labels = image_data()
    optimiser = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"],
                                  weight_decay=CONFIG["weight_decay"])
    schedule = diffusion_schedule(device=device)
    history = []
    model.train()
    for step in range(1, CONFIG["steps"] + 1):
        index = torch.randint(len(images), (CONFIG["batch_size"],))
        x, y = images[index].to(device), labels[index].to(device)
        loss = diffusion_loss(model, x, schedule[2], y) if CONFIG["task"] == "ddpm" else flow_loss(model, x, y)
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
def generate(model, count, device, generator):
    labels = torch.zeros(count, dtype=torch.long, device=device)
    if CONFIG["task"] == "ddpm":
        return sample_ddpm(model, count, diffusion_schedule(device=device), device,
                           generator, labels, shape=(3, 32, 32))
    return sample_flow(model, labels, 25, device, generator, shape=(3, 32, 32))

# %% Run the experiment
def run(out=None, device=None, seed=3801, split="validation", show=True):
    source = program_source()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", STUDENT_ID):
        raise ValueError("Use letters, digits, underscore or hyphen for STUDENT_ID.")
    if CONFIG["task"] not in ("ddpm", "flow"):
        raise ValueError("Set task to 'ddpm' or 'flow'.")
    if not 1 <= CONFIG["steps"] <= MAX_STEPS or not 1 <= CONFIG["batch_size"] <= MAX_BATCH:
        raise ValueError("Training steps or batch size exceeds the practical limit.")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    folder = Path(out or f"lab-06-{STUDENT_ID}")
    (folder/"samples").mkdir(parents=True, exist_ok=True)
    model = build_model().to(device)
    size = model_size(model)
    history = train(model, device)
    metrics = image_score(model, folder/"samples", device, split)
    result = save_submission(folder, source, history, metrics, size, seed, split)
    if show:
        show_results(history, folder)
    return result


if __name__ == "__main__":
    run()
