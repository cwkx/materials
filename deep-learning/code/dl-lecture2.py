# %% Tensors
# Run each section in IPython, or use %run -i code/dl-lecture2.py.
# %% Values, axes and shapes
import os
import torch
torch.manual_seed(3801)
torch.set_num_threads(2)
device = os.environ.get('DL_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')

scalar = torch.tensor(3.)
vector = torch.arange(6.)
matrix = torch.tensor([[1., 2., 3.], [2., 4., 6.]])
print(scalar.shape, vector.shape, matrix.shape)
print(matrix.ndim, torch.linalg.matrix_rank(matrix).item())  # 2 axes, matrix rank 1
print(matrix.numel(), matrix.dtype, matrix.device)

# %% Elementwise multiplication and matrix multiplication
print(matrix * matrix)              # (2, 3), elementwise
print(matrix @ matrix.T)            # (2, 2), row dot products
row = torch.tensor([10., 20., 30.])
print(matrix + row)                 # (2, 3) + (3,)
column = torch.tensor([[100.], [200.]])
print(matrix + column)              # (2, 3) + (2, 1)

# %% Indexing and reductions: predict the shape before running each line
images = torch.rand(4, 3, 8, 8)     # batch, channels, height, width
print(images[0].shape, images[:1].shape)
print(images[:, 0].shape, images[:, :1].shape)
mean = images.mean(dim=(2, 3), keepdim=True)
print(mean.shape, (images - mean).shape)
one_grey_image = torch.rand(1, 1, 8, 8)
print(one_grey_image.squeeze().shape)       # loses the batch axis too
print(one_grey_image.squeeze(1).shape)      # remove only the channel axis

# %% A view can share storage
values = torch.arange(6.)
table = values.view(2, 3)
table[0, 0] = -1
print(values)                       # the first value changed
independent = table.clone()
independent[0, 0] = 99
print(table[0, 0])                  # still -1
print(table.view(-1).shape, images.flatten(start_dim=1).shape)

# %% Transposing changes strides; view needs a compatible layout
transposed = table.T
print(table.stride(), transposed.stride(), transposed.is_contiguous())
try:
    transposed.view(-1)
except RuntimeError:
    print('This transpose cannot be flattened with view.')
print(transposed.reshape(-1))       # reshape may copy
print(transposed.contiguous().view(-1))

# %% Axis order: use permute for CHW to HWC
chw = torch.arange(3 * 2 * 2).view(3, 2, 2)
hwc = chw.permute(1, 2, 0)
wrong = chw.reshape(2, 2, 3)
print(hwc[0, 0], wrong[0, 0])        # [0, 4, 8] versus [0, 1, 2]

# %% Types and devices
labels = torch.tensor([0, 2, 1])     # int64 class indices
pixels = torch.tensor([0, 127, 255], dtype=torch.uint8)
print(labels.dtype, (pixels.float() / 255).dtype)
on_device = images.to(device)
print(on_device.device, on_device.cpu().shape)


# %% Gradients
# %% Autograd records operations on tensors that require gradients
import torch
x = torch.zeros(16, 16, requires_grad=True)
z = (x + 2) * 3
loss = z.mean()
print(x.is_leaf, z.is_leaf, x.requires_grad)
print(z.grad_fn, loss.item(), x.grad)
loss.backward()
print(x.grad[0, 0].item())           # 3 / 256

# %% Backward accumulates into .grad; the next forward builds a new graph
((x + 2) * 3).mean().backward()
print(x.grad[0, 0].item())           # 6 / 256
x.grad = None
((x + 2) * 3).mean().backward()
print(x.grad[0, 0].item())           # 3 / 256 again

# %% Derivative of a scalar loss
w = torch.tensor([1., 2., 3.], requires_grad=True)
loss = w.square().mean()
loss.backward()
print(loss.item(), w.grad)          # mean(w**2), then 2*w/3
print(w.detach())                   # backward has not updated w

# %% Logging and plotting without keeping the training graph
logged_loss = loss.item()           # a Python number for a loss history
array = w.detach().cpu().numpy()    # values for Matplotlib
with torch.no_grad():
    prediction = 2 * w
print(prediction.requires_grad)    # False

# %% Optional: detach changes the derivative, not the numerical value
u = torch.tensor(2., requires_grad=True)
full = u * u
print(torch.autograd.grad(full, u)[0].item())        # 4
stopped = u * u.detach()
print(torch.autograd.grad(stopped, u)[0].item())     # 2
# detach shares storage. Use detach().clone() for an independent snapshot.


# %% Weights
# %% A linear layer stores weights and a bias
import copy
import torch
from torch import nn
from torch.nn import functional as F

x = torch.tensor([[1.], [2.], [3.]])
target = 2 * x + 1
model = nn.Linear(1, 1)
with torch.no_grad():
    model.weight.zero_()
    model.bias.zero_()
initial = copy.deepcopy(model)
print([(name, tuple(p.shape)) for name, p in model.named_parameters()])
print(sum(p.numel() for p in model.parameters()))    # 2

# %% y = x W^T + b. The input and the prediction are both (3, 1).
prediction = model(x)
print(torch.allclose(prediction, x @ model.weight.T + model.bias))
loss = F.mse_loss(prediction, target)
model.zero_grad(set_to_none=True)
loss.backward()
print(loss.item())                  # 83 / 3
print(model.weight.grad, model.bias.grad)  # -68 / 3, -10
print(model.weight, model.bias)     # both still zero

# %% One manual SGD update. Keep the update outside the recorded graph.
learning_rate = 0.1
with torch.no_grad():
    for parameter in model.parameters():
        parameter.sub_(learning_rate * parameter.grad)
after_manual = torch.cat([p.detach().flatten() for p in model.parameters()])
print(after_manual)                # [2.2667, 1.0000]
print(F.mse_loss(model(x), target).item())  # approximately 0.331852

# %% The optimiser performs the same update from the same initial weights
optimised = copy.deepcopy(initial)
optimiser = torch.optim.SGD(optimised.parameters(), lr=learning_rate)
optimiser.zero_grad(set_to_none=True)
loss = F.mse_loss(optimised(x), target)
loss.backward()
optimiser.step()
after_sgd = torch.cat([p.detach().flatten() for p in optimised.parameters()])
print(torch.allclose(after_manual, after_sgd))      # True

# %% Inspect an MLP without training it
network = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
for name, parameter in network.named_parameters():
    print(name, tuple(parameter.shape), parameter.requires_grad)
print(sum(p.numel() for p in network.parameters()))  # 58
print(network(torch.randn(5, 4)).shape)             # (5, 2)


# %% Classifier
# %% A small subset of the same MNIST data used in Practical 1
import os
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(3801)
torch.set_num_threads(2)
device = os.environ.get('DL_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
out = Path('work/lecture02')
out.mkdir(parents=True, exist_ok=True)
dataset = MNIST('data', train=True, transform=ToTensor(), download=True)
order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(3801))
train_set = Subset(dataset, order[:5000].tolist())
validation_set = Subset(dataset, order[50000:51000].tolist())
loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)
train_check = DataLoader(train_set, batch_size=128, shuffle=False, num_workers=0)
validation = DataLoader(validation_set, batch_size=128, shuffle=False, num_workers=0)
images, labels = next(iter(loader))
print(images.shape, labels.shape, images.dtype, labels.dtype)
# Both subsets come from the official training set. The official test set stays unused.

# %% A module registers its submodules and their parameters
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(28 * 28, 64)
        self.output = nn.Linear(64, 10)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.output(F.relu(self.hidden(x)))

model = Classifier().to(device)
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
print('Parameters:', sum(p.numel() for p in model.parameters()))  # 50,890
logits = model(images.to(device))
print(logits.shape, F.cross_entropy(logits, labels.to(device)).item())
# Cross entropy accepts raw logits and integer class indices. No softmax here.

# %% Evaluation includes the short final batch
@torch.no_grad()
def evaluate(model, batches):
    model.eval()
    total_loss, correct, count = 0., 0, 0
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction='sum').item()
        correct += (logits.argmax(dim=1) == y).sum().item()
        count += len(y)
    return total_loss / count, correct / count

# %% Save the current curves; refresh training.png in the browser
history = []
def record(epoch):
    train_loss, train_accuracy = evaluate(model, train_check)
    val_loss, val_accuracy = evaluate(model, validation)
    history.append([epoch, train_loss, val_loss, train_accuracy, val_accuracy])
    print(f'{epoch}: train CE {train_loss:.3f}, validation CE {val_loss:.3f}, '
          f'validation accuracy {val_accuracy:.1%}')
    h = torch.tensor(history).numpy()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), layout='constrained')
    for column, axis, label in [(1, axes[0], 'Cross entropy (nats)'),
                                 (3, axes[1], 'Accuracy')]:
        axis.plot(h[:, 0], h[:, column], 'o-', label='Training subset')
        axis.plot(h[:, 0], h[:, column + 1], 'o-', label='Validation subset')
        axis.set(xlabel='Epoch', ylabel=label)
        axis.legend()
    fig.savefig(out / 'training.png', dpi=160)
    plt.close(fig)

# %% The training loop
record(0)
for epoch in range(1, 7):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimiser.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimiser.step()
    record(epoch)

# %% First validation images, in order, including mistakes
model.eval()
with torch.no_grad():
    x, y = next(iter(validation))
    predicted = model(x[:12].to(device)).argmax(dim=1).cpu()
fig, axes = plt.subplots(2, 6, figsize=(9, 3.4), layout='constrained')
for i, axis in enumerate(axes.flat):
    axis.imshow(x[i, 0], cmap='gray', vmin=0, vmax=1)
    axis.set_title(f'{predicted[i].item()} / {y[i].item()}', fontsize=11)
    axis.axis('off')
fig.suptitle('Prediction / target on the first 12 validation images')
fig.savefig(out / 'predictions.png', dpi=160)
plt.close(fig)
(out / 'classifier-results.json').write_text(json.dumps({
    'seed': 3801, 'device': device, 'torch': str(torch.__version__),
    'parameters': sum(p.numel() for p in model.parameters()),
    'training_examples': len(train_set), 'validation_examples': len(validation_set),
    'history_columns': ['epoch', 'train_ce', 'validation_ce', 'train_accuracy', 'validation_accuracy'],
    'history': history}, indent=2) + '\n')


# %% Images
# %% A CIFAR-10 image as a tensor, using the Practical 3 split
import os
from pathlib import Path
import torch
from torch import nn
from torchvision.datasets import CIFAR10
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(3801)
torch.set_num_threads(2)
device = os.environ.get('DL_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
out = Path('work/lecture02')
out.mkdir(parents=True, exist_ok=True)
dataset = CIFAR10('data', train=True, download=True)
order = torch.randperm(50000, generator=torch.Generator().manual_seed(3801))
rgb = torch.tensor(dataset.data[order[:8]], dtype=torch.float32)
rgb = rgb.permute(0, 3, 1, 2).to(device) / 255
print(rgb.shape)                    # (8, 3, 32, 32)

# %% Broadcasting the luminance weights over a batch and over all pixels
weights = rgb.new_tensor([0.299, 0.587, 0.114])[None, :, None, None]
grey = (rgb * weights).sum(dim=1, keepdim=True)
print(weights.shape, grey.shape)   # (1, 3, 1, 1), (8, 1, 32, 32)

# %% Save the inputs and targets for the colourisation task
fig, axes = plt.subplots(2, 8, figsize=(10, 3), layout='constrained')
for i in range(8):
    axes[0, i].imshow(rgb[i].permute(1, 2, 0).detach().cpu().numpy())
    axes[1, i].imshow(grey[i, 0].cpu(), cmap='gray', vmin=0, vmax=1)
    for axis in axes[:, i]:
        axis.set(xticks=[], yticks=[])
axes[0, 0].set_ylabel('RGB')
axes[1, 0].set_ylabel('Grey')
fig.savefig(out / 'images.png', dpi=160)
plt.close(fig)

# %% Flatten, encode to 64 values, and decode back to an RGB image
flat = grey.flatten(1)
encoder = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, 64)).to(device)
decoder = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 3072),
                        nn.Unflatten(1, (3, 32, 32))).to(device)
z = encoder(flat)
prediction = decoder(z)
print(flat.shape, z.shape, prediction.shape)
print(encoder[0].weight.shape, decoder[2].weight.shape)
assert prediction.shape == rgb.shape
loss = (prediction - rgb).square().mean()
loss.backward()
print(loss.item(), encoder[0].weight.grad.norm().item())
# We have computed gradients. Practical 3 adds the optimiser updates,
# compares bottleneck sizes, and introduces the greyscale skip path.
