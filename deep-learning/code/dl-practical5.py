# A two-layer CIFAR-10 horse GAN, adapted from the MNIST example using 8x8 features and RGB output.
# Set STUDENT_ID, then run this file or paste it into one notebook cell.
import ast
import hashlib
import json
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

CONFIG = dict(task="dcgan", steps=2000, batch_size=64,
              learning_rate=0.001, weight_decay=0.0)


class Generator(nn.Module):
    def __init__(self, latent=16, width=16):
        super().__init__()
        self.latent = latent
        self.project = nn.Linear(latent, width*2*8*8)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(width*2, width, 4, 2, 1, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(),
            nn.ConvTranspose2d(width, 3, 4, 2, 1), nn.Tanh())

    def forward(self, z):
        return self.net(self.project(z).reshape(z.size(0), -1, 8, 8))


class Discriminator(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, width, 4, 2, 1), nn.LeakyReLU(.2),
                                 nn.Conv2d(width, width*2, 4, 2, 1), nn.LeakyReLU(.2),
                                 nn.Flatten(), nn.Linear(width*2*8*8, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class Model(nn.Module):
    def __init__(self, generator_width=16, discriminator_width=16, latent=16):
        super().__init__()
        self.generator = Generator(latent=latent, width=generator_width)
        self.discriminator = Discriminator(width=discriminator_width)


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
    stem = "lab-05"
    code = folder / (stem + "-code.py")
    code.write_text(source)
    names = [f"batch-{i:02d}.png" for i in range(1, 11)]
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    updates = {"generator": CONFIG["steps"], "discriminator": CONFIG["steps"]}
    result = dict(format=6, lab=5, seed=seed, config=dict(CONFIG),
                  parameters=size[0], buffer_elements=size[1], updates=updates,
                  protocol="comp3801-cifar10-horses-kid1000-v1-" + split, code_sha256=digest(code),
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
    plt.plot(curves[:, 0], curves[:, 1], label="Generator")
    plt.plot(curves[:, 0], curves[:, 2], label="Discriminator")
    plt.legend()
    plt.xlabel("GAN iteration"); plt.ylabel("Loss"); plt.tight_layout(); plt.show()
    from PIL import Image
    plt.figure(figsize=(6, 6))
    plt.imshow(Image.open(folder/"samples/batch-01.png"))
    plt.axis("off"); plt.tight_layout(); plt.show()


def image_data(split="train"):
    from torchvision.datasets import CIFAR10
    dataset = CIFAR10(root=DATA_DIR, train=split != "test", download=True)
    index = torch.where(torch.tensor(dataset.targets) == 7)[0]
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


def discriminator_loss(real_logits, fake_logits):
    return (F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
            + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits)))

def generator_loss(fake_logits):
    return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))

# %% Training
def train(model, device):
    images, _ = image_data()
    opt_g = torch.optim.Adam(model.generator.parameters(), lr=CONFIG["learning_rate"],
                             betas=(.5, .999), weight_decay=CONFIG["weight_decay"])
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=CONFIG["learning_rate"],
                             betas=(.5, .999), weight_decay=CONFIG["weight_decay"])
    history = []
    model.train()
    for step in range(1, CONFIG["steps"] + 1):
        real = images[torch.randint(len(images), (CONFIG["batch_size"],))].to(device)
        noise = torch.randn(len(real), model.generator.latent, device=device)
        fake = model.generator(noise)
        opt_d.zero_grad(set_to_none=True)
        d_loss = discriminator_loss(model.discriminator(real), model.discriminator(fake.detach()))
        d_loss.backward(); opt_d.step()
        for parameter in model.discriminator.parameters(): parameter.requires_grad_(False)
        opt_g.zero_grad(set_to_none=True)
        g_loss = generator_loss(model.discriminator(fake))
        g_loss.backward(); opt_g.step()
        for parameter in model.discriminator.parameters(): parameter.requires_grad_(True)
        if step == 1 or step % max(1, CONFIG["steps"]//40) == 0 or step == CONFIG["steps"]:
            history.append([step, round(g_loss.item(), 6), round(d_loss.item(), 6)])
        if step == 1 or step % max(1, CONFIG["steps"]//10) == 0 or step == CONFIG["steps"]:
            print(f"{step} / {CONFIG['steps']}: generator loss {g_loss.item():.4f}, "
                  f"discriminator loss {d_loss.item():.4f}", flush=True)
    return history


@torch.no_grad()
def generate(model, count, device, generator):
    return model.generator(torch.randn(count, model.generator.latent, device=device, generator=generator))

# %% Run the experiment
def run(out=None, device=None, seed=3801, split="validation", show=True):
    source = program_source()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", STUDENT_ID):
        raise ValueError("Use letters, digits, underscore or hyphen for STUDENT_ID.")
    if CONFIG["task"] != "dcgan":
        raise ValueError("Keep task='dcgan' in CONFIG.")
    if not 1 <= CONFIG["steps"] <= MAX_STEPS or not 1 <= CONFIG["batch_size"] <= MAX_BATCH:
        raise ValueError("Training steps or batch size exceeds the practical limit.")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    if torch.device(device).type == "cuda":
        torch.backends.cudnn.benchmark = True
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    folder = Path(out or f"lab-05-{STUDENT_ID}")
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
