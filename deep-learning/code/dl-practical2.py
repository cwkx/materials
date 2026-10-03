# Code for dl-practical2.pdf. Run the # %% sections in order in VS Code or IPython.
# Follow Practical 1 for environment setup; datasets download on first use.
# %%
import math
import torch
from torch import nn
from torch.nn import functional as F
torch.set_num_threads(2)

# %%
"""Plotting for Practical 2; the model and updates appear in the teaching cells."""
from pathlib import Path
from io import BytesIO
import base64
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

COLOURS = ['#315e8d', '#ffffff', '#c16c36']
CMAP = LinearSegmentedColormap.from_list('spirals', COLOURS)
AXIS = torch.linspace(-1.1, 1.1, 220)
GX, GY = torch.meshgrid(AXIS, AXIS, indexing='xy')
GRID = torch.stack((GX.flatten(), GY.flatten()), dim=1)


@torch.no_grad()
def snapshot(model, X, y, step):
    return dict(step=step, score=model(GRID).reshape(GX.shape).numpy().copy(),
                loss=(model(X)-y).square().mean().item())


def _prediction(ax, score, X, y):
    ax.imshow(np.clip(score, -1, 1), origin='lower', extent=(-1.1, 1.1, -1.1, 1.1),
              cmap=CMAP, vmin=-1, vmax=1, alpha=0.32, interpolation='bilinear')
    ax.scatter(*np.asarray(X).T, c=np.asarray(y).ravel(), cmap=CMAP,
               vmin=-1, vmax=1, s=9, linewidths=0, zorder=3)
    ax.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), xticks=[-1, 0, 1], yticks=[-1, 0, 1],
           xlabel='$x_1$', ylabel='$x_2$', aspect='equal')
    for spine in ax.spines.values():
        spine.set_color('#b5b5b5')


def show_progress(frames, X, y):
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), layout='constrained')
    for ax, step in zip(axes, (0, 1500, 6000)):
        frame = min(frames, key=lambda f: abs(f['step'] - step))
        _prediction(ax, frame['score'], X, y)
        ax.set_title(f"{frame['step']:,} updates\nMSE = {frame['loss']:.3f}", fontsize=11)
    plt.show()
    return fig


def replay(frames, X, y, path='spirals.html'):
    """Write a self-contained training animation with a loss curve and slider."""
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9), layout='constrained')
    steps = [f['step'] for f in frames]
    losses = [f['loss'] for f in frames]
    pictures = []
    for index, frame in enumerate(frames):
        for ax in axes:
            ax.clear()
        _prediction(axes[0], frame['score'], X, y)
        axes[0].set_title(f"Predictions after {frame['step']:,} updates")
        axes[1].semilogy(steps, losses, color='#dddddd', lw=1.5)
        axes[1].semilogy(steps[:index+1], losses[:index+1], color='#222222', lw=1.5)
        axes[1].plot(frame['step'], frame['loss'], 'o', color='#222222', ms=5)
        axes[1].set(xlabel='Update', ylabel='Mean squared error', title='Training loss',
                    box_aspect=1)
        axes[1].margins(x=0.03)
        axes[1].spines[['top', 'right']].set_visible(False)
        stream = BytesIO()
        fig.savefig(stream, format='png', dpi=95)
        pictures.append('data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode())
    plt.close(fig)
    html = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Learning two spirals</title>
<style>body{font:17px system-ui;max-width:920px;margin:24px auto;padding:0 16px;color:#222}
img{width:100%;height:auto}button{font:inherit;padding:6px 14px}input{flex:1;min-width:80px}
.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap}p{line-height:1.5}</style>
<h1>Learning two spirals</h1>
<p>Dots show the training data, with orange labels +1 and blue labels -1. As training
changes the weights, the background shows the network's predictions and the curve tracks
their mean squared error. Background colours fade to white near output zero and reach
full strength at -1 and +1; values beyond this range use the same end colours.</p>
<img id="frame" alt="Predictions on two spirals beside the training loss curve">
<div class="controls"><button id="play">Play</button><button id="restart">Restart</button>
<label for="seek">Update</label><input id="seek" type="range" min="0" value="0">
<output id="step"></output></div>
<p>Pause and move the slider to compare stages of training. After changing the model,
rerun the training cell and reload this page to view its predictions.</p>
<script>const pictures=PICTURES,steps=STEPS;
const frame=document.getElementById('frame'), seek=document.getElementById('seek'),
play=document.getElementById('play'), step=document.getElementById('step');
let timer=null;seek.max=pictures.length-1;
function draw(i){seek.value=i;frame.src=pictures[i];step.value=steps[i];}
function pause(){clearInterval(timer);timer=null;play.textContent='Play';}
play.onclick=()=>{if(timer){pause();return;}if(+seek.value===pictures.length-1)draw(0);
play.textContent='Pause';timer=setInterval(()=>{const i=+seek.value+1;
if(i>=pictures.length){pause();return;}draw(i);},180);};
seek.oninput=()=>{pause();draw(+seek.value);};
document.getElementById('restart').onclick=()=>{pause();draw(0);};draw(0);
</script></html>'''
    html = html.replace('PICTURES', json.dumps(pictures)).replace('STEPS', json.dumps(steps))
    Path(path).write_text(html, encoding='utf-8')
    print(f'Open {Path(path).resolve()} in a browser to replay training.')
    return Path(path)


@torch.no_grad()
def show_weights(model, X, y, path):
    """Plot the loss over two output weights, with the hidden layers and bias fixed."""
    H = model[:-1](X)
    b = model[-1].bias
    path = torch.stack(path).squeeze(-1)
    optimum = torch.linalg.lstsq(H, y - b).solution[:, 0]
    bounds = torch.cat((path, optimum[None]))
    span = (bounds.max(0).values - bounds.min(0).values).max().clamp_min(0.5)
    centre = (bounds.max(0).values + bounds.min(0).values) / 2
    a = torch.linspace(-0.75, 0.75, 220) * span
    wx, wy = torch.meshgrid(a + centre[0], a + centre[1], indexing='xy')
    W = torch.stack((wx.flatten(), wy.flatten()))
    # Expand mean((H @ w + b - y)**2) as a quadratic in the two output weights.
    residual = b - y
    losses = ((W * ((H.T @ H / len(X)) @ W)).sum(0)
              + 2 * (residual.T @ H / len(X)) @ W + residual.square().mean())
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.7), layout='constrained')
    contours = axes[0].contour(wx, wy, losses.reshape(wx.shape),
                              levels=[0.01, 0.03, 0.1, 0.3, 1, 2],
                              colors='#aaaaaa', linewidths=0.8)
    axes[0].clabel(contours, fontsize=8, fmt='%g')
    axes[0].plot(*path.T, '.-', color='#222222', lw=1.2, ms=4)
    axes[0].scatter(*path[0], s=40, facecolor='white', edgecolor='#222222', zorder=5)
    axes[0].annotate('Start', path[0], xytext=(-12, -19), textcoords='offset points', fontsize=10)
    axes[0].annotate(f'{len(path)-1} updates', path[-1], xytext=(-5, 20),
                     textcoords='offset points', ha='right', fontsize=10)
    axes[0].annotate('', xy=path[1], xytext=path[0],
                     arrowprops={'arrowstyle': '->', 'color': '#222222', 'lw': 1.5})
    axes[0].set(xlabel='Output weight $w_1$', ylabel='Output weight $w_2$',
                title='Loss as two weights change', aspect='equal')
    for spine in axes[0].spines.values():
        spine.set_color('#b5b5b5')
    score = model[:-1](GRID) @ path[-1] + b
    _prediction(axes[1], score.reshape(GX.shape), X, y)
    axes[1].set_title('Resulting predictions')
    print(f'MSE after {len(path)-1} output-weight updates: {(H @ path[-1:, :].T + b - y).square().mean():.5f}')
    plt.show()
    return fig

# %%
import torch
from torch import nn
# Run the plotting-functions cell first in dl-practical2.py.

torch.set_num_threads(2)
torch.manual_seed(42)

def spirals(n):
    t = torch.linspace(0, 1, n)
    angle = 2.5 * torch.pi * t
    radius = 0.15 + 0.8 * t
    arm = torch.stack((radius * angle.cos(), radius * angle.sin()), dim=1)
    X = torch.cat((arm, -arm)) + 0.01 * torch.randn(2 * n, 2)
    y = torch.cat((torch.ones(n, 1), -torch.ones(n, 1)))
    return X, y

X, y = spirals(128)
X_test, y_test = spirals(512)

# %%
torch.manual_seed(7)
width = 32
model = nn.Sequential(
    nn.Linear(2, width), nn.Tanh(),
    nn.Linear(width, width), nn.Tanh(),
    nn.Linear(width, 2), nn.Tanh(),
    nn.Linear(2, 1),
)
print(sum(p.numel() for p in model.parameters()), "parameters")

# %%
score = model(X)
loss = (score - y).square().mean()
loss.backward()
print(model[-1].weight.grad)
model.zero_grad()

# %%
frames = [snapshot(model, X, y, step=0)]
for step in range(1, 6001):
    loss = (model(X) - y).square().mean()
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            p -= 0.1 * p.grad
            p.grad.zero_()
    if step % 150 == 0:
        frames.append(snapshot(model, X, y, step))
show_progress(frames, X, y)
replay(frames, X, y)

# %%
with torch.no_grad():
    prediction = model(X_test)
    accuracy = ((prediction > 0) == (y_test > 0)).float().mean()
print(f"Test accuracy: {accuracy:.1%}")

# %%
H = model[:-1](X).detach()
b = model[-1].bias.detach()
w = torch.zeros(2, 1, requires_grad=True)
path = [w.detach().clone()]
for step in range(20):
    loss = (H @ w + b - y).square().mean()
    loss.backward()
    with torch.no_grad():
        w -= 0.2 * w.grad
        w.grad.zero_()
    path.append(w.detach().clone())
show_weights(model, X, y, path)
