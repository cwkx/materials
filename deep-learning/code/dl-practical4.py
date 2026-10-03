# Code for dl-practical4.pdf. Run the # %% sections in order in VS Code or IPython.
# Follow Practical 1 for environment setup; datasets download on first use.
# %%
import math
import torch
from torch import nn
from torch.nn import functional as F
torch.set_num_threads(2)

# %%
from pathlib import Path
import math
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat, reduce
from skimage import data as pictures
import matplotlib.pyplot as plt
coffee=torch.tensor(pictures.coffee(),dtype=torch.float32)/255
coffee_batch=repeat(coffee,'h w c -> b c h w',b=10)
mosaic=rearrange(coffee_batch,'(r k) c h w -> (r h) (k w) c',r=2)
print(coffee.shape,coffee_batch.shape,mosaic.shape)

# %%
class ResidualBlock(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv2d(channels,channels,3,1,1),nn.ReLU(),
                                  nn.Conv2d(channels,channels,3,1,1))
    def forward(self,x):
        return F.relu(x+self.layers(x))

class Stage(nn.Module):
    def __init__(self,inputs,outputs):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv2d(inputs,outputs,4,2,1),nn.ReLU(),
                                  ResidualBlock(outputs),ResidualBlock(outputs))
    def forward(self,x):return self.layers(x)

class Colouriser(nn.Module):
    def __init__(self, feature_skip=False):
        super().__init__()
        self.feature_skip=feature_skip
        self.encoder=nn.Sequential(Stage(1,16),Stage(16,32))
        self.up=nn.Sequential(nn.Upsample(scale_factor=2,mode='nearest'),
                              nn.Conv2d(32,16,3,1,1))
        self.decoder=nn.Sequential(ResidualBlock(16),nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Conv2d(16,3,3,1,1))
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)
    def forward(self,x):
        h=self.encoder[0](x)
        z=self.encoder[1](h)
        up=self.up(z)
        if self.feature_skip:up=up+h
        return x.expand(-1,3,-1,-1)+self.decoder(up)

# %%
def data(root='data', split='train'):
    if split not in ('train', 'validation'):
        raise ValueError('choose train or validation')
    dataset = CIFAR10(root, train=True, download=True)
    order = torch.randperm(50000, generator=torch.Generator().manual_seed(3801))
    index = order[:40000] if split == 'train' else order[40000:]
    rgb = torch.tensor(dataset.data[index], dtype=torch.float32).permute(0, 3, 1, 2)/255
    return rgb, torch.tensor(dataset.targets)[index]

def grey(rgb):
    weights = rgb.new_tensor([.299, .587, .114])[None, :, None, None]
    return (rgb*weights).sum(1, keepdim=True)

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

@torch.no_grad()
def examples(model, images, out, device='cuda'):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model.eval()
    target = images[:8].to(device)
    mono = grey(target)
    fig = plot_rows([mono.expand(-1, 3, -1, -1), model(mono), target],
                    ['Greyscale', 'Prediction', 'Target'])
    fig.savefig(out/'colourisation.png', dpi=160)
    plt.close(fig)

# %%
device = "cuda"
images, labels = data()
validation, categories = data(split="validation")
torch.manual_seed(3801)
model = Colouriser()
history = train(model, images, steps=6000, device=device, learning_rate=.001)
print(evaluate(model, validation, device))
print("parameters", sum(p.numel() for p in model.parameters()))
examples(model, validation, "work/lab-04", device)
photo = F.interpolate(coffee_batch[:1], (128, 192), mode="area").to(device)
with torch.no_grad():
    output = model(grey(photo))
plt.imshow(output[0].clamp(0, 1).permute(1, 2, 0).cpu()); plt.axis("off"); plt.show()
