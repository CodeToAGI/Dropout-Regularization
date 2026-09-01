"""
CodeToAGI — Deep Learning Series EP16 Challenge
See BatchNorm Speed Up Training Yourself

Task:
1) Take your EP15 10-layer MLP with He initialization.
2) Train it on MNIST for 30 epochs. Record loss curve and final accuracy.
3) Add nn.BatchNorm1d after every Linear layer (before activation).
4) Train again — same epochs, same learning rate.
5) Compare loss curves: how many epochs to reach the same accuracy?
6) Try removing model.eval() at test time — see what breaks.
7) Post your epoch-to-accuracy comparison in the comments.

Run:
    python ep16_batchnorm_comparison.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE   = 128
EPOCHS       = 30
LR           = 1e-3          # same LR for both models (BatchNorm usually allows higher)
SEED         = 42
HIDDEN_SIZES = [512, 256, 256, 128, 128, 64, 64, 32, 32]  # 9 hidden → 10 layers total
INPUT_DIM    = 784
NUM_CLASSES  = 10
OUTPUT_DIR   = Path("ep16_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"Device: {DEVICE}")
print(f"Epochs: {EPOCHS} | Batch size: {BATCH_SIZE} | LR: {LR}")
print(f"Architecture: {INPUT_DIM} → {' → '.join(map(str, HIDDEN_SIZES))} → {NUM_CLASSES}")
print("-" * 60)


# ── Data ──────────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_ds = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)


# ── Models ────────────────────────────────────────────────────────────────────
def he_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class MLP(nn.Module):
    """10-layer MLP with optional BatchNorm after every Linear (before ReLU)."""
    def __init__(self, use_bn: bool = False):
        super().__init__()
        self.use_bn = use_bn
        layers = []
        in_dim = INPUT_DIM
        for h in HIDDEN_SIZES:
            layers.append(nn.Linear(in_dim, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            in_dim = h
        layers.append(nn.Linear(in_dim, NUM_CLASSES))
        self.net = nn.Sequential(*layers)
        self.apply(he_init)

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


# ── Train / Eval helpers ──────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += x.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, use_eval_mode: bool = True):
    if use_eval_mode:
        model.eval()
    else:
        # Intentionally leave in train mode to demonstrate the bug
        model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += x.size(0)
    return total_loss / total, correct / total


def train_model(use_bn: bool, tag: str):
    print(f"\n{'='*60}")
    print(f"Training: {tag}")
    print(f"{'='*60}")
    model = MLP(use_bn=use_bn).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc = 0.0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        te_loss, te_acc = evaluate(model, test_loader, criterion, use_eval_mode=True)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)
        best_acc = max(best_acc, te_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{EPOCHS} | "
                  f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
                  f"test_loss={te_loss:.4f} acc={te_acc:.4f}")

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s | Best test acc: {best_acc:.4f}")

    # Demonstrate the silent bug: forgetting model.eval()
    if use_bn:
        print("\n--- Bug demo: evaluate WITHOUT model.eval() ---")
        buggy_losses, buggy_accs = [], []
        for _ in range(3):
            bl, ba = evaluate(model, test_loader, criterion, use_eval_mode=False)
            buggy_losses.append(bl)
            buggy_accs.append(ba)
            print(f"  Run: loss={bl:.4f}  acc={ba:.4f}")
        print("→ Results fluctuate because BatchNorm uses per-batch stats in train mode.")
        print("→ Always call model.eval() before inference / validation.")

    return model, history, best_acc


# ── Run both experiments ──────────────────────────────────────────────────────
model_no_bn, hist_no_bn, best_no_bn = train_model(use_bn=False, tag="He init ONLY (no BatchNorm)")
model_bn,    hist_bn,    best_bn    = train_model(use_bn=True,  tag="He init + BatchNorm1d")


# ── Find epochs to reach target accuracy ──────────────────────────────────────
TARGET_ACC = 0.97  # 97% as mentioned in the episode

def epochs_to_acc(history, target=TARGET_ACC):
    for i, acc in enumerate(history["test_acc"], 1):
        if acc >= target:
            return i
    return None

ep_no = epochs_to_acc(hist_no_bn)
ep_bn = epochs_to_acc(hist_bn)

print("\n" + "=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)
print(f"Target accuracy: {TARGET_ACC*100:.0f}%")
print(f"Without BatchNorm → reached in {ep_no or 'N/A'} epochs  (best {best_no_bn:.4f})")
print(f"With    BatchNorm → reached in {ep_bn or 'N/A'} epochs  (best {best_bn:.4f})")
if ep_no and ep_bn:
    speedup = ep_no / ep_bn
    print(f"Speed-up factor ≈ {speedup:.1f}× fewer epochs")
print("=" * 60)


# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Loss
axes[0].plot(hist_no_bn["test_loss"], label="No BatchNorm", color="#ff5555", linewidth=2)
axes[0].plot(hist_bn["test_loss"],    label="With BatchNorm", color="#50fa7b", linewidth=2)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Test Loss")
axes[0].set_title("Test Loss — With vs Without BatchNorm")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(hist_no_bn["test_acc"], label="No BatchNorm", color="#ff5555", linewidth=2)
axes[1].plot(hist_bn["test_acc"],    label="With BatchNorm", color="#50fa7b", linewidth=2)
axes[1].axhline(TARGET_ACC, color="#f1fa8c", linestyle="--", linewidth=1.5, label=f"{TARGET_ACC*100:.0f}% target")
if ep_no:
    axes[1].axvline(ep_no, color="#ff5555", linestyle=":", alpha=0.7)
if ep_bn:
    axes[1].axvline(ep_bn, color="#50fa7b", linestyle=":", alpha=0.7)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Test Accuracy")
axes[1].set_title("Test Accuracy — With vs Without BatchNorm")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = OUTPUT_DIR / "ep16_batchnorm_comparison.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved → {plot_path}")

# Also save histories for later analysis
np.savez(
    OUTPUT_DIR / "ep16_histories.npz",
    no_bn_loss=hist_no_bn["test_loss"],
    no_bn_acc=hist_no_bn["test_acc"],
    bn_loss=hist_bn["test_loss"],
    bn_acc=hist_bn["test_acc"],
)
print(f"Histories saved → {OUTPUT_DIR / 'ep16_histories.npz'}")
print("\nDone. Post your epoch-to-accuracy numbers in the YouTube comments!")
