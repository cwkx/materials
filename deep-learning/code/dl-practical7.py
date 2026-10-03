# Conditional flow matching generates two voices saying digits one to nine.
# Set STUDENT_ID, then run this file or paste it into one notebook cell.
import wave
import ast
import hashlib
import json
import random
import re
import zipfile
from urllib.request import urlretrieve
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

STUDENT_ID = "studentID"
DATA_DIR = Path(__file__).parent if "__file__" in globals() else Path(".")

# %% Model and training settings

CONFIG = dict(task="audio_flow", steps=3000, batch_size=32,
              learning_rate=.001, weight_decay=.01)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Conv2d(3, 16, 3, padding=1)
        self.label = nn.Embedding(10, 16)
        self.voice = nn.Embedding(3, 16)
        self.time = nn.Linear(1, 16)
        self.down1 = nn.Sequential(nn.Conv2d(16, 32, 4, 2, 1), nn.GroupNorm(4, 32))
        self.down2 = nn.Sequential(nn.Conv2d(32, 48, 4, 2, 1), nn.GroupNorm(4, 48))
        self.down3 = nn.Sequential(nn.Conv2d(48, 64, 4, 2, 1), nn.GroupNorm(4, 64))
        self.condition = nn.Linear(16, 64)
        self.middle = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(4, 64))
        self.up3 = nn.Sequential(nn.Conv2d(64+48, 48, 3, padding=1), nn.GroupNorm(4, 48))
        self.up1 = nn.Sequential(nn.Conv2d(48+32, 32, 3, padding=1), nn.GroupNorm(4, 32))
        self.up2 = nn.Sequential(nn.Conv2d(32+16, 16, 3, padding=1), nn.GroupNorm(4, 16))
        self.output = nn.Conv2d(16, 1, 3, padding=1)

    def voice_embedding(self, voice):
        amount = voice[:, None].to(self.voice.weight.dtype).clamp(0, 1)
        blend = torch.lerp(self.voice.weight[0], self.voice.weight[1], amount)
        return torch.where((voice == 2)[:, None], self.voice.weight[2], blend)

    def forward(self, x, t, y, voice):
        f = torch.linspace(-1, 1, x.size(2), device=x.device)
        frequency = f[None, None, :, None].expand_as(x)
        s = torch.linspace(-1, 1, x.size(3), device=x.device)
        position = s[None, None, None, :].expand_as(x)
        embedding = self.time(t[:, None])+self.label(y)+self.voice_embedding(voice)
        a = self.input(torch.cat((x, frequency, position), 1))
        a = F.silu(a+embedding[:, :, None, None])
        b = F.silu(self.down1(a))
        c = F.silu(self.down2(b))
        h = F.silu(self.down3(c)+self.condition(embedding)[:, :, None, None])
        h = F.silu(self.middle(h))
        h = F.interpolate(h, size=c.shape[-2:], mode="nearest")
        h = F.silu(self.up3(torch.cat((h, c), 1)))
        h = F.interpolate(h, size=b.shape[-2:], mode="nearest")
        h = F.silu(self.up1(torch.cat((h,b),1)))
        h = F.interpolate(h, size=a.shape[-2:], mode="nearest")
        return self.output(F.silu(self.up2(torch.cat((h,a),1))))

def build_model():
    return Model()

MAX_PARAMETERS, MAX_STEPS, MAX_BATCH = 1000000, 10000, 64

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
    stem = "lab-07"
    code = folder / (stem + "-code.py")
    code.write_text(source)
    names = [f"sample-{digit}.wav" for digit in range(1, 10)]
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    updates = {"model": CONFIG["steps"]}
    result = dict(format=6, lab=7, seed=seed, config=dict(CONFIG),
                  parameters=size[0], buffer_elements=size[1], updates=updates,
                  protocol="comp3801-audiomnist-voices-kad-v1-" + split, code_sha256=digest(code),
                  metrics=metrics, loss=history,
                  samples_sha256={name: digest(folder/"samples"/name) for name in names})
    record = folder / (stem + "-results.json")
    record.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    archive = folder.with_name(folder.name + ".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as pack:
        for path in [code, record] + [folder/"samples"/name for name in names]:
            pack.write(path, path.relative_to(folder))
    print(json.dumps(metrics, indent=2))
    print("KAD x 1000:", round(metrics["kad"]*1000, 3))
    print("Submission:", archive.resolve())
    return result


DATA_SHA256 = '0a5b11b61cf95d0b581b70c5c1bfbdef396912bfbbb80c39584f922dbb2e6bf5'

DATA_URL = "https://raw.githubusercontent.com/cwkx/research-data/main/datasets/audio.npz"

def audio_data(split="train"):
    path = DATA_DIR / "audio.npz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(DATA_URL, path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != DATA_SHA256:
        raise ValueError("Use audio.npz from https://github.com/cwkx/research-data/tree/main/datasets")
    data = np.load(path)
    keep = data["split"] == {"train": 0, "validation": 1, "test": 2}[split]
    x = torch.from_numpy(data["spectrograms"][keep].astype(np.float32))[:, None]
    y = torch.from_numpy(data["labels"][keep].astype(np.int64))
    voices = torch.from_numpy(data["voices"][keep].astype(np.int64))
    waves = torch.from_numpy(data["waveforms"][keep].astype(np.float32))/32767
    return x, y, voices, waves


def flow_loss(model, clean, labels, voices):
    labels, voices = labels.clone(), voices.clone()
    hidden = torch.rand(len(labels), device=clean.device) < .1
    labels[hidden], voices[hidden] = 9, 2
    noise = torch.randn_like(clean)
    t = torch.rand(len(clean), device=clean.device)
    time = t[:, None, None, None]
    state = (1-time)*noise + time*clean
    return F.mse_loss(model(state, t, labels, voices), clean-noise)


@torch.no_grad()
def sample(model, labels, voices, start):
    model.eval()
    x = start.clone()
    for k in range(64):
        t = torch.full((len(x),), k/64, device=x.device)
        conditional = model(x, t, labels, voices)
        unconditional = model(x, t, torch.full_like(labels, 9), torch.full_like(voices, 2))
        x = x + (unconditional + 1.5*(conditional-unconditional))/64
    return x[:, 0]


@torch.no_grad()
def to_audio(spectrograms, generator):
    # Reconstruct 8 kHz audio from log magnitudes with Griffin-Lim.
    magnitude = (spectrograms*2.5-5).exp().clamp(0, 100)
    magnitude = F.pad(magnitude, (0, 0, 0, 1))
    window = torch.hann_window(256, device=spectrograms.device)
    angles = 2*torch.pi*torch.rand(magnitude.shape, device=magnitude.device, generator=generator)
    z = magnitude*torch.exp(1j*angles)
    previous = 0
    for _ in range(64):
        waves = torch.istft(z, 256, 128, window=window, length=8064)
        rebuilt = torch.stft(waves, 256, 128, window=window, return_complex=True)
        z = rebuilt - (.99/1.99)*previous
        previous = rebuilt
        z = magnitude*z/(z.abs()+1e-8)
    return torch.istft(z, 256, 128, window=window, length=8064)


def save_audio(path, waves):
    # Each word is followed by 0.2 seconds of silence in the saved WAV.
    pcm = (waves.cpu().clamp(-1, 1)*32767).round().to(torch.int16)
    joined = F.pad(pcm, (0, 1600)).flatten().numpy().astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1); output.setsampwidth(2); output.setframerate(8000)
        output.writeframes(joined.tobytes())
    return pcm.float()/32767


@torch.no_grad()
def audio_features(encoder, waves, device):
    import torchaudio
    features = []
    for batch in waves.split(8):
        batch = torchaudio.functional.resample(batch.to(device), 8000, 16000)
        hidden = encoder.extract_features(batch)[0][-1]
        features.append(hidden.mean(1).cpu())
    return torch.cat(features)


def kernel_distance(real, generated):
    # Unbiased squared MMD in fixed speech features; small negative estimates are valid.
    real, generated = real.double(), generated.double()
    bandwidth = torch.pdist(real).median().square().clamp_min(1e-12)
    kernel = lambda a, b: torch.exp(-torch.cdist(a, b).square()/(2*bandwidth))
    rr, gg, rg = kernel(real, real), kernel(generated, generated), kernel(real, generated)
    off_diagonal = lambda k: (k.sum()-k.diag().sum())/(len(k)*(len(k)-1))
    return float(off_diagonal(rr) + off_diagonal(gg) - 2*rg.mean())


@torch.no_grad()
def audio_score(model, folder, device, split):
    import torchaudio
    _, digits, voices, real = audio_data(split)
    generator = torch.Generator(device=device).manual_seed(921)
    endpoints = []
    for digit in range(9):
        start = torch.randn(20, 1, 128, 64, device=device, generator=generator)
        labels = torch.full((40,), digit, device=device, dtype=torch.long)
        voice = torch.tensor([0., 1.], device=device).repeat_interleave(20)
        generated = sample(model, labels, voice, start.repeat(2, 1, 1, 1))
        waves = to_audio(generated, generator).cpu()
        pcm = (waves.clamp(-1, 1)*32767).round()/32767
        endpoints.append(pcm.reshape(2, 20, 8064))
        # One female-to-male transition in each submitted preview.
        labels = torch.full((5,), digit, device=device, dtype=torch.long)
        voice = torch.linspace(0, 1, 5, device=device)
        preview = sample(model, labels, voice, start[:1].repeat(5, 1, 1, 1))
        save_audio(folder/f"sample-{digit+1}.wav", to_audio(preview, generator))
    # Frozen evaluator weights are downloaded once and cached by PyTorch.
    encoder = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model().to(device).eval()
    real_features = audio_features(encoder, real, device)
    scores = [[], []]
    for digit, pair in enumerate(endpoints):
        for voice in (0, 1):
            reference = real_features[(digits == digit) & (voices == voice)]
            features = audio_features(encoder, pair[voice], device)
            scores[voice].append(kernel_distance(reference, features))
    female, male = map(np.mean, scores)
    return dict(kad=float((female+male)/2), kad_female=float(female),
                kad_male=float(male), samples=360)


# %% Training
def train(model, device):
    images, labels, voices, _ = audio_data()
    images, labels, voices = images.to(device), labels.to(device), voices.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"],
                                  weight_decay=CONFIG["weight_decay"])
    history = []
    model.train()
    for step in range(1, CONFIG["steps"]+1):
        ids = torch.randint(len(images), (CONFIG["batch_size"],), device=device)
        loss = flow_loss(model, images[ids], labels[ids], voices[ids])
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimiser.step()
        if step == 1 or step % max(1, CONFIG["steps"]//40) == 0 or step == CONFIG["steps"]:
            history.append([step, round(loss.item(), 6)])
        if step == 1 or step % max(1, CONFIG["steps"]//10) == 0 or step == CONFIG["steps"]:
            print(f"{step} / {CONFIG['steps']}: loss {loss.item():.4f}", flush=True)
    return history


def show_results(history, folder):
    import matplotlib.pyplot as plt
    values = np.array(history)
    plt.plot(values[:, 0], values[:, 1])
    plt.xlabel("Update"); plt.ylabel("Flow matching loss"); plt.show()
    if "get_ipython" in globals():
        from IPython.display import Audio, display
        for name in ("sample-1.wav", "sample-7.wav"):
            with wave.open(str(folder/"samples"/name)) as f:
                pcm = np.frombuffer(f.readframes(5*9664), dtype="<i2")
            display(Audio(pcm.astype(np.float32)/32767, rate=8000, normalize=False))


# %% Run the experiment
def run(out=None, device=None, seed=3801, split="validation", show=True):
    source = program_source()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", STUDENT_ID):
        raise ValueError("Use letters, digits, underscore or hyphen for STUDENT_ID.")
    if CONFIG["task"] != "audio_flow":
        raise ValueError("Keep task='audio_flow' in CONFIG.")
    if not 1 <= CONFIG["steps"] <= MAX_STEPS or not 1 <= CONFIG["batch_size"] <= MAX_BATCH:
        raise ValueError("Training steps or batch size exceeds the practical limit.")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    folder = Path(out or f"lab-07-{STUDENT_ID}")
    (folder/"samples").mkdir(parents=True, exist_ok=True)
    model = build_model().to(device)
    size = model_size(model)
    history = train(model, device)
    metrics = audio_score(model, folder/"samples", device, split)
    result = save_submission(folder, source, history, metrics, size, seed, split)
    if show:
        show_results(history, folder)
    return result


if __name__ == "__main__":
    run()
