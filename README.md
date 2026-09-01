CodeToAGI — Deep Learning Series

Episode 17: Dropout & Regularization Explained — Preventing Overfitting in Deep Nets

Module 4 — Training Deep Networks
Presenter: Mahaz Abbasi · AI Engineer
Series: 72 episodes · From neurons to GPT



What this episode covers





Overfitting in deep networks — why depth makes it worse



Dropout — randomly killing neurons to force robust learning



How dropout works at train time vs inference (inverted dropout scaling)



L1 regularization — sparsity and feature selection



L2 regularization / weight decay — the most common default



L1 vs L2 — when to use which



Early stopping — the simplest regularizer of all



PyTorch code: nn.Dropout, weight_decay in AdamW, early-stopping loop



Challenge

Fix an Overfit Model — Three Ways





Take your EP16 MLP (with BatchNorm) and train it on MNIST for 50 epochs.



Intentionally overfit: use a very small training subset (e.g. 1000 samples).



Record train vs validation accuracy curve — confirm the gap.



Add nn.Dropout(0.4) after each ReLU. Retrain. Record curves.



Add weight_decay=1e-4 to your Adam optimizer (prefer AdamW). Retrain.



Implement early stopping: stop when val loss hasn’t improved for 5 epochs.



Post your three val accuracy comparisons in the comments.

Solution file: ep17_regularization_comparison.py

python ep17_regularization_comparison.py

The script:





Creates a small MNIST subset (easy to overfit)



Trains baseline, +Dropout, +Weight Decay (AdamW), +Early Stopping, and a full combo



Plots train/val accuracy and validation loss for all runs



Restores best weights automatically



Prints a clear comparison table



Key takeaways







Technique



What it does



When to use





Dropout



Randomly zeros neurons → forces redundant features



Still overfitting after weight decay





L1



Pushes weights exactly to zero (sparsity)



High-dim sparse inputs / feature selection





L2 / Weight Decay



Shrinks all weights, keeps them small



Default for almost every model





Early Stopping



Stops when val loss plateaus, restores best



Always — free and complementary

Default modern recipe:
AdamW(weight_decay=1e-4) + nn.Dropout(0.3–0.5) + early stopping (patience 5–20)



PyTorch quick reference

import torch.nn as nn
import torch.optim as optim

# 1) Dropout in the model (disabled automatically in eval mode)
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(p=0.4),          # ← after activation
    nn.Linear(256, 10),
)

# 2) Correct weight decay → use AdamW
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# 3) Early stopping skeleton
best_val_loss = float("inf")
patience_cnt  = 0
PATIENCE      = 5
best_state    = None

for epoch in range(100):
    model.train()
    train_one_epoch(...)

    model.eval()
    val_loss = evaluate(...)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        patience_cnt = 0
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            break

model.load_state_dict(best_state)   # restore best



Resources





Dropout paper: Srivastava et al., 2014



AdamW: Loshchilov & Hutter, 2019



Full series code: github.com/CodeToAGI/deep-learning-series



Previous: EP16 — Batch Normalization



Next: EP18 — Learning Rate Schedules & Warmup



License

Code and materials for educational use. Feel free to fork, experiment, and share your results.
