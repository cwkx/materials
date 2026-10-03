# Code for dl-practical1.pdf. Run the # %% sections in order in VS Code or IPython.
# Follow Practical 1 for environment setup; datasets download on first use.
# %%
import math
import torch
from torch import nn
from torch.nn import functional as F
torch.set_num_threads(2)

# %%
import sys
import torch
import torchvision

print(sys.executable)
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.tensor([1., 2., 3.], device=device)
print(x, x.device, (x * x).sum().item())

# %%
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

# %%
def load_data():
    dataset = MNIST("data", train=True, download=True)
    order = torch.randperm(60000, generator=torch.Generator().manual_seed(3801))
    images = dataset.data[order].unsqueeze(1).float()/127.5 - 1
    labels = dataset.targets[order]
    return images[:50000], labels[:50000], images[50000:], labels[50000:]

# %%
images, labels, validation_images, validation_labels = load_data()
print(images.shape, labels.shape)
plt.imshow(images[0, 0], cmap="gray", vmin=-1, vmax=1)
plt.title(f"Label: {labels[0].item()}")
plt.axis("off")
plt.show()

# %%
def make_model(width=16):
    return nn.Sequential(nn.Flatten(), nn.Linear(784, width),
                         nn.ReLU(), nn.Linear(width, 10))

# %%
torch.manual_seed(3801)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = make_model().to(device)
x, y = images[:64].to(device), labels[:64].to(device)
logits = model(x)
print(logits.shape)
print(logits.argmax(dim=1)[:10], y[:10])
print(sum(p.numel() for p in model.parameters()))

# %%
loss = F.cross_entropy(logits, y)
probabilities = logits.softmax(dim=1)
print(loss.item(), probabilities[0], probabilities[0].sum().item())

# %%
def train(model, images, labels, device, steps=150, learning_rate=0.003):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    model.train()
    for step in range(steps):
        index = torch.randint(len(images), (64,))
        x, y = images[index].to(device), labels[index].to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(loss.item())
        if (step + 1) % 25 == 0:
            print(step + 1, round(loss.item(), 3))
    return history

# %%
history = train(model, images, labels, device)
plt.plot(range(1, len(history)+1), history)
plt.xlabel("Gradient update")
plt.ylabel("Batch cross-entropy")
plt.show()

# %%
@torch.no_grad()
def accuracy(model, images, labels, device):
    model.eval()
    correct = 0
    for x, y in zip(images.split(256), labels.split(256)):
        correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
    return correct / len(images)

# %%
score = accuracy(model, validation_images, validation_labels, device)
print(f"Validation accuracy: {100*score:.2f}%")
