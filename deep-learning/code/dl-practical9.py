# A sine hidden layer followed by a ReLU hidden layer fits each coordinate signal.
# Set STUDENT_ID and run. The dataset is downloaded from cwkx/research-data.
# Requires numpy, torch, pillow, scipy and imageio-ffmpeg.
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import time
from urllib.request import urlretrieve
import zipfile

import numpy as np
import torch
from torch import nn

STUDENT_ID = "studentID"
DATA_DIR = Path(__file__).parent if "__file__" in globals() else Path(".")
DATA_URL = "https://raw.githubusercontent.com/cwkx/research-data/main/datasets/multimodal.npz"
DATA_SHA256 = "d83c908f100d76b458608e75347eaaa33e2cf882d09a06350bdece51e15c4d20"
CONFIG = dict(task="implicit", steps=2000, batch_size=4096,
              learning_rate=0.001, weight_decay=0.0, storage_dtype="float32")
MAX_MODEL_BYTES = 1024**2
NAMES = ("parrot", "crow", "bach", "cat")
SAMPLE_NAMES = ("parrot.png", "crow.png", "bach.wav", "cat.mp4")


class Model(nn.Module):
    def __init__(self, inputs=2, outputs=3, width=256):
        super().__init__()
        # Inputs: 1 time coordinate for audio, 2 for images, 3 for video.
        self.fc1 = nn.Linear(inputs, width)
        self.fc2 = nn.Linear(width, width)
        # Outputs: 1 amplitude for mono audio, or 3 RGB values for images/video.
        self.fc3 = nn.Linear(width, outputs)
        # Start sine units at a range of frequency scales, then learn their weights.
        with torch.no_grad():
            self.fc1.weight *= torch.logspace(0, 4, width)[:, None]

    def forward(self, x):
        x = torch.sin(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def build_model(inputs=2, outputs=3):
    return Model(inputs, outputs)


def coordinates(index, shape):
    """Convert flat array indices to (x, y), (t,) or (x, y, t)."""
    axes = []
    for size in reversed(shape):
        axes.append(2*((index % size).float()+0.5)/size-1)
        index = index // size
    return torch.stack(axes, dim=1)


def train_signal(model, values, shape, seed):
    model.train()
    generator = torch.Generator(device=values.device).manual_seed(seed+1)
    optimiser = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"],
                                 weight_decay=CONFIG["weight_decay"])
    history = []
    for step in range(1, CONFIG["steps"]+1):
        index = torch.randint(len(values), (CONFIG["batch_size"],),
                              device=values.device, generator=generator)
        target = values[index].float()
        if values.dtype == torch.uint8:
            target = target/127.5-1
        loss = (model(coordinates(index, shape))-target).square().mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        if step == 1 or step % max(1, CONFIG["steps"]//40) == 0 or step == CONFIG["steps"]:
            history.append([step, loss.item()])
        if step == 1 or step % max(1, CONFIG["steps"]//10) == 0 or step == CONFIG["steps"]:
            print(f"{step} / {CONFIG['steps']}: MSE {loss.item():.6f}", flush=True)
    return history


# %% Save model state and reconstruct the complete signals
@torch.no_grad()
def save_model(model, path, precision):
    arrays = {}
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        value = value.detach().cpu()
        if value.is_floating_point():
            value = value.to(getattr(torch, precision))
        arrays[name] = value.numpy()
        if not np.isfinite(arrays[name]).all():
            raise ValueError("Non-finite model state: "+name)
    np.savez(path, **arrays)
    size = Path(path).stat().st_size
    if size > MAX_MODEL_BYTES:
        raise ValueError(f"Model uses {size:,} bytes; the limit is {MAX_MODEL_BYTES:,}.")
    return size


@torch.no_grad()
def reconstruct(model, values, shape, path, rate=44100, fps=(25, 1), batch_size=32768):
    model.eval()
    path = Path(path)
    video = path.suffix == ".mp4"
    output = None if video else np.empty((len(values), values.shape[1]), dtype=np.float32)
    writer = None
    if video:
        import imageio_ffmpeg
        writer = imageio_ffmpeg.write_frames(str(path), (shape[2], shape[1]),
            fps=fps[0]/fps[1], codec="libx264", pix_fmt_out="yuv420p",
            quality=None, macro_block_size=1,
            output_params=["-crf", "18", "-maxrate", "2500k", "-bufsize", "5000k",
                           "-movflags", "+faststart"])
        writer.send(None)
    error, energy = 0.0, 0.0
    try:
        for first in range(0, len(values), batch_size):
            last = min(first+batch_size, len(values))
            index = torch.arange(first, last, device=values.device)
            target = values[first:last].float()
            if values.dtype == torch.uint8:
                target = target/127.5-1
            prediction = model(coordinates(index, shape)).float().clamp(-1, 1)
            if prediction.shape != target.shape or not torch.isfinite(prediction).all():
                raise ValueError("Return one finite prediction per target value.")
            error += (prediction-target).square().double().sum().item()
            energy += target.square().double().sum().item()
            if video:
                # Stream RGB batches; FFmpeg assembles them into complete frames.
                writer.send(((prediction+1)*127.5).round().byte().cpu().numpy().tobytes())
            else:
                output[first:last] = prediction.cpu().numpy()
    finally:
        if writer is not None:
            writer.close()
    if video and path.stat().st_size >= 5_000_000:
        raise ValueError("Keep the compressed video preview below 5 MB.")
    if path.suffix == ".png":
        from PIL import Image
        pixels = np.rint((output.reshape(*shape, -1)+1)*127.5).astype(np.uint8)
        Image.fromarray(pixels).save(path)
    elif path.suffix == ".wav":
        from scipy.io import wavfile
        wavfile.write(path, rate, output[:, 0] if output.shape[1] == 1 else output)
    mse = error/values.numel()
    return dict(mse=mse, psnr_db=10*math.log10(4/max(mse, 1e-12)),
                snr_db=10*math.log10(max(energy, 1e-12)/max(error, 1e-12)))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def program_source():
    if "__file__" in globals():
        source = Path(__file__).read_text()
    else:
        source = get_ipython().history_manager.input_hist_raw[-1].rstrip()+"\n"
    names = {getattr(node, "name", "") for node in ast.parse(source).body}
    if not {"build_model", "run"} <= names:
        raise ValueError("Run the complete program in one notebook cell.")
    return source


# %% Run four independent fits and write the submission ZIP
def run(out=None, device=None, seed=3801, split="reconstruction", show=True):
    source = program_source()
    if split != "reconstruction" or CONFIG["task"] != "implicit":
        raise ValueError("Use the full-signal reconstruction task.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", STUDENT_ID):
        raise ValueError("Use letters, digits, underscore or hyphen for STUDENT_ID.")
    if CONFIG["storage_dtype"] not in ("float16", "float32"):
        raise ValueError("Store floating-point state as float16 or float32.")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    path = Path(DATA_DIR)/"multimodal.npz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(DATA_URL, path)
    if digest(path) != DATA_SHA256:
        raise ValueError("Use multimodal.npz from the research-data repository.")
    folder = Path(out or f"lab-09-{STUDENT_ID}")
    for subfolder in ("samples", "models"):
        (folder/subfolder).mkdir(parents=True, exist_ok=True)
    losses, metrics, representations = {}, {}, {}
    with np.load(path, allow_pickle=False) as data:
        for name, sample in zip(NAMES, SAMPLE_NAMES):
            array = data[name]
            shape, channels = array.shape[:-1], array.shape[-1]
            values = torch.as_tensor(array.reshape(-1, channels), device=device)
            torch.manual_seed(seed)
            model = build_model(len(shape), channels).to(device)
            model_path = folder/"models"/(name+".npz")
            save_model(model, model_path, CONFIG["storage_dtype"])
            print(f"Fitting {name}: {shape}, {channels} channel(s)", flush=True)
            start = time.perf_counter()
            losses[name] = train_signal(model, values, shape, seed)
            size = save_model(model, model_path, CONFIG["storage_dtype"])
            with np.load(model_path, allow_pickle=False) as saved, torch.no_grad():
                for key, value in list(model.named_parameters()) + list(model.named_buffers()):
                    value.copy_(torch.as_tensor(saved[key], device=device, dtype=value.dtype))
            score = reconstruct(model, values, shape, folder/"samples"/sample,
                                int(data["bach_sample_rate"]), tuple(data["cat_fps"]))
            metrics[name+"_psnr_db"] = score["psnr_db"]
            if name == "bach":
                metrics["bach_snr_db"] = score["snr_db"]
            representations[name] = dict(shape=list(shape), channels=channels,
                parameters=sum(p.numel() for p in model.parameters()),
                buffer_elements=sum(b.numel() for b in model.buffers()),
                model_bytes=size, model_sha256=digest(model_path), mse=score["mse"],
                seconds=time.perf_counter()-start)
            print(f"{name}: {score['psnr_db']:.2f} dB, {size:,} bytes", flush=True)
            del model, values
    metrics["mean_psnr_db"] = sum(metrics[n+"_psnr_db"] for n in NAMES)/4
    code, record = folder/"lab-09-code.py", folder/"lab-09-results.json"
    code.write_text(source)
    result = dict(format=7, lab=9, seed=seed, config=CONFIG, loss=losses, metrics=metrics,
        protocol="dl-multimodal-v1-reconstruction", code_sha256=digest(code), data_sha256=DATA_SHA256,
        parameters=max(r["parameters"] for r in representations.values()),
        buffer_elements=max(r["buffer_elements"] for r in representations.values()),
        updates={n: CONFIG["steps"] for n in NAMES}, representations=representations,
        samples_sha256={n: digest(folder/"samples"/n) for n in SAMPLE_NAMES})
    record.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    paths = [code, record] + [folder/"samples"/n for n in SAMPLE_NAMES]
    paths += [folder/"models"/(n+".npz") for n in NAMES]
    archive = folder.with_name(folder.name+".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as pack:
        for path in paths:
            pack.write(path, path.relative_to(folder))
    print(json.dumps(metrics, indent=2), "\nSubmission:", archive.resolve())
    return result


if __name__ == "__main__":
    run()
