# Code for dl-practical3.pdf. Run the # %% sections in order in VS Code or IPython.
# Follow Practical 1 for environment setup; datasets download on first use.
# %%
import math
import torch
from torch import nn
from torch.nn import functional as F
torch.set_num_threads(2)

# %%
import math
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def plot_rows(rows, labels):
    fig, axes = plt.subplots(len(rows), 1, figsize=(9, 1.05*len(rows)), squeeze=False)
    for ax, row, label in zip(axes[:, 0], rows, labels):
        image = make_grid(row.detach().cpu().clamp(0, 1), nrow=len(row), padding=2, pad_value=1)
        ax.imshow(image.permute(1, 2, 0), interpolation='nearest')
        ax.set(xticks=[], yticks=[])
        ax.set_ylabel(label, rotation=0, ha='right', va='center', fontsize=10, labelpad=10)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.subplots_adjust(left=.20, right=.995, top=.99, bottom=.01, hspace=.08)
    return fig

def plot_marginals(values):
    values = values.detach().cpu()
    lo, hi = values.min().item(), values.max().item()
    bins = torch.linspace(lo, hi, 51).numpy()
    grid = torch.linspace(lo, hi, 300)
    fig, axes = plt.subplots(1, values.shape[1], figsize=(9, 2.6),
                             sharex=True, sharey=True, layout='constrained')
    for j, ax in enumerate(axes):
        v = values[:, j]
        normal = torch.distributions.Normal(v.mean(), v.std())
        ax.hist(v, bins=bins, density=True, alpha=.55, label='Encoded images')
        ax.plot(grid, normal.log_prob(grid).exp(), color='darkorange', label='Fitted Gaussian')
        ax.set(xlabel=f'$z_{j+1}$', ylabel='Density' if j == 0 else '', xlim=(lo, hi))
        ax.set_title(rf'$\mu={v.mean():.2f},\ \sigma={v.std():.2f}$', fontsize=9)
    axes[0].legend(fontsize=7, loc='upper left')
    return fig

@torch.no_grad()
def evaluate(model, images, device='cuda'):
    if isinstance(model, nn.Module):
        model.eval()
    rgb_error = luminance_error = 0.
    for target in images.split(128):
        target = target.to(device)
        predicted = model(grey(target)).clamp(0, 1)
        rgb_error += F.mse_loss(predicted, target, reduction='sum').item()
        luminance_error += F.mse_loss(grey(predicted), grey(target), reduction='sum').item()
    mse = rgb_error/images.numel()
    return {'mse': mse, 'psnr_db': -10*math.log10(max(mse, 1e-12)),
            'luminance_mse': luminance_error/(len(images)*images.shape[-2]*images.shape[-1])}

# %%
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt

# %%
def data(root='data', split='train'):
    if split not in ('train', 'validation'):
        raise ValueError('choose train or validation')
    dataset = CIFAR10(root, train=True, download=True)
    order = torch.randperm(50000, generator=torch.Generator().manual_seed(3801))
    index = order[:40000] if split == 'train' else order[40000:]
    rgb = torch.tensor(dataset.data[index], dtype=torch.float32).permute(0, 3, 1, 2)/255
    return rgb, torch.tensor(dataset.targets)[index]

# %%
images, labels = data()
validation, categories = data(split="validation")

# %%
def grey(rgb):
    weights = rgb.new_tensor([.299, .587, .114])[None, :, None, None]
    return (rgb*weights).sum(1, keepdim=True)

# %%
mono = grey(images[:8])
print(images.shape, mono.shape, mono.flatten(1).shape)
plot_rows([mono.expand(-1, 3, -1, -1), images[:8]],
          ["Greyscale", "RGB target"])
plt.show()

# %%
class Colouriser(nn.Module):
    def __init__(self, latent=64, skip=False, width=256):
        super().__init__()
        self.skip = skip
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(1024, width), nn.ReLU(),
                                     nn.Linear(width, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, width), nn.ReLU(),
                                     nn.Linear(width, 3072), nn.Unflatten(1, (3, 32, 32)))
        nn.init.zeros_(self.decoder[2].weight)
        nn.init.zeros_(self.decoder[2].bias)

    def decode(self, z, mono=None):
        rgb = self.decoder(z)
        return rgb + mono if self.skip else rgb

    def forward(self, mono):
        return self.decode(self.encoder(mono), mono)

# %%
model = Colouriser(latent=4)
z = model.encoder(mono)
rgb = model.decode(z)
print(z.shape, rgb.shape)
print(sum(p.numel() for p in model.parameters()))

# %%
def train(model, images, steps=6000, batch=64, device='cuda', learning_rate=.001, seed=3802):
    model.to(device).train()
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    batches = torch.Generator().manual_seed(seed)
    losses = []
    for step in range(steps):
        index = torch.randint(len(images), (batch,), generator=batches)
        target = images[index].to(device)
        loss = F.mse_loss(model(grey(target)), target)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        losses.append(loss.item())
    return losses

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
models, histories = {}, {}
for d in (4, 64):
    torch.manual_seed(3801)
    name = f"{d} values"
    net = Colouriser(latent=d)
    histories[name] = train(net, images, device=device)
    models[name] = net.eval()

# %%
for d in (4, 64):
    torch.manual_seed(3801)
    name = f"{d} values + skip"
    net = Colouriser(latent=d, skip=True)
    histories[name] = train(net, images, device=device)
    models[name] = net.eval()
with torch.no_grad():
    target = validation[:8].to(device)
    c = grey(target)
    rows = [c.expand(-1, 3, -1, -1)]
    rows += [net(c) for net in models.values()]
    rows += [target]
plot_rows(rows, ["Greyscale"] + list(models) + ["Target"])
plt.show()

# %%
for name, net in models.items():
    print(name, evaluate(net, validation, device))
for name, history in histories.items():
    plt.semilogy(history, label=name)
plt.xlabel("Update"); plt.ylabel("Training RGB MSE")
plt.legend(); plt.show()

# %%
plain = models["64 values"]
skipped = models["64 values + skip"]
with torch.no_grad():
    c = grey(validation[[1, 2]].to(device))
    a = torch.linspace(0, 1, 9, device=device)[:, None]
    z = plain.encoder(c)
    s = skipped.encoder(c)
    mixed_z = (1-a)*z[:1] + a*z[1:]
    mixed_s = (1-a)*s[:1] + a*s[1:]
    b = a[:, :, None, None]
    mixed_grey = (1-b)*c[:1] + b*c[1:]
    rows = [plain.decode(mixed_z),
            skipped.decode(mixed_s, mixed_grey),
            skipped.decode(mixed_s, c[:1]),
            skipped.decode(mixed_s, c[1:])]
plot_rows(rows, ["Latent only", "Blend both paths",
                       "Keep first grey", "Keep last grey"])
plt.show()

# %%
plain4 = models["4 values"]
with torch.no_grad():
    z = plain4.encoder(grey(images[:4096]).to(device)).cpu()
print(z.mean(0), z.std(0))
plot_marginals(z)
plt.show()

# %%
def gaussian_samples(values, count=8, seed=3803):
    mean = values.mean(0)
    centred = values - mean
    covariance = centred.T @ centred / (len(values)-1)
    eye = torch.eye(values.shape[1], device=values.device, dtype=values.dtype)
    scale = torch.linalg.cholesky(covariance + 1e-5*eye)
    rng = torch.Generator(device=values.device).manual_seed(seed)
    noise = torch.randn(count, values.shape[1], generator=rng,
                        device=values.device, dtype=values.dtype)
    return mean + noise @ scale.T

# %%
rows, names = [], []
with torch.no_grad():
    for d in (4, 64):
        net = models[f"{d} values"]
        z = net.encoder(grey(images[:4096]).to(device)).cpu()
        sampled = gaussian_samples(z).to(device)
        rows += [net(grey(validation[:8]).to(device)),
                 net.decode(sampled)]
        names += [f"{d}: encoded", f"{d}: Gaussian"]
plot_rows(rows, names)
plt.show()

# %%
skipped4 = models["4 values + skip"]
with torch.no_grad():
    c_train = grey(images[:4096])
    z = skipped4.encoder(c_train.to(device)).cpu()
    joint = torch.cat([z, c_train.flatten(1)], dim=1)
    sampled = gaussian_samples(joint)
    new_z = sampled[:, :4].to(device)
    new_grey = sampled[:, 4:].reshape(8, 1, 32, 32).to(device)
    real_grey = grey(validation[:8]).to(device)
    rows = [real_grey, skipped4.decode(new_z, real_grey),
            new_grey, skipped4.decode(new_z, new_grey)]
plot_rows(rows, ["Real grey", "Real grey + random z",
                       "Gaussian grey", "Both paths sampled"])
plt.show()
