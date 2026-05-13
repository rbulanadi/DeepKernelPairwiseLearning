# Deep Kernel Pairwise Learning

by Ralph Bulanadi, Yongtao Liu
Center for Nanophase Materials Sciences, Oak Ridge National Laboratory

## Features
- Tie/Equal Support: Users can indicate when two options are equally good
- Confidence Weighting: Express certainty about each preference (weak/medium/strong)
- UCB and EUBO strategies for active learning


## Core Capabilities
- Deep Kernel Learning: Neural network feature extraction + Gaussian Process
- Pairwise Comparisons: Learn from relative preferences instead of absolute ratings
- Active Learning: Intelligently select next comparisons to maximize information gain
- High-Dimensional Inputs: Handle image patches, spectra, or any high-D data
- Uncertainty Quantification: Bayesian uncertainty estimates for exploration


The system learns to predict a scalar target value (e.g., preference score) from high-dimensional inputs (e.g., image patches) using only pairwise comparison feedback.

## Key Concepts
1. Pairwise Comparisons
Instead of asking "Rate this on a scale of 1-10", we ask "Which is better, A or B?"
Benefits:
- More natural for humans
- No need to calibrate absolute scales
- More reliable feedback
- Avoids numerical biases

2. Deep Kernel Learning
Combines neural networks with Gaussian Processes:
Why this works:
- NN learns non-linear feature representations
- GP provides uncertainty quantification
- Best of both worlds.

3. Tie Support
Express that two options are equally good.
When to use ties:
- Options genuinely similar
- Uncertainty about which is better
- Save time on difficult decisions

4. Confidence Weighting
Express certainty about each comparison.
Effect on training:
- High confidence → larger gradient, more influence
- Low confidence → smaller gradient, less influence
- Model focuses on reliable comparisons

## Install Prerequisites
```python
!pip install git+https://github.com/yongtaoliu/DeepKernelPairwiseGP.git
```

### Example Workflow
```python
import numpy as np
import torch
from model import fit_dkpg, predict_utility
from acq import acq_ucb, get_user_preference

# Generate some data
n_points = 100
input_dim = 256  # e.g., 16x16 image patches
X = np.random.randn(n_points, input_dim)

# Initial training set
train_indices = np.array([0, 1, 2, 3, 4])
X_train = X[train_indices]

# Initial comparisons: [winner_idx, loser_idx]
train_comp = torch.tensor([
    [0, 1],  # Point 0 > Point 1
    [2, 3],  # Point 2 > Point 3
    [0, 4],  # Point 0 > Point 4
], dtype=torch.long)

# Train model
mll, pref_model, dkl_model = fit_dkpg(
    X_train=X_train,
    train_comp=train_comp,
    feature_dim=16,
    num_epochs=1000
)

# Predict on all points
mean, var = predict_utility(dkl_model, X)
print(f"Best point: {mean.argmax()}")
```

### Example with Ties and Confidence
```python
# Comparisons with tie support: [idx1, idx2, type]
# type: 0 = idx1>idx2, 1 = idx2>idx1, 2 = equal
train_comp = torch.tensor([
    [0, 1, 0],  # Point 0 > Point 1
    [2, 3, 2],  # Point 2 ≈ Point 3 (equal/tie)
    [0, 4, 0],  # Point 0 > Point 4
], dtype=torch.long)

# Confidence weights for each comparison
confidence = torch.tensor([1.0, 1.0, 0.5], dtype=torch.float64)

# Train with tie support and confidence weighting
mll, pref_model, dkl_model = fit_dkpg(
    X_train=X_train,
    train_comp=train_comp,
    confidence_weights=confidence,
    allow_ties=True,      # ← Enable tie support
    tolerance=0.15,       # ← Utilities within 0.15 considered equal
    feature_dim=16,
    num_epochs=1000
)

# Statistics
n_strict = ((train_comp[:, 2] == 0) | (train_comp[:, 2] == 1)).sum().item()
n_ties = (train_comp[:, 2] == 2).sum().item()
print(f"Comparisons: {len(train_comp)} total ({n_strict} strict, {n_ties} ties)")
```
### Active Learning Loop
```python
from acq import acq_ucb, acquire_preference

# Track previous comparisons
previous_comparisons = set()

for iteration in range(10):
    # Train model
    mll, pref_model, dkl_model = fit_dkpg(
        X_train, train_comp,
        confidence_weights=confidence,
        allow_ties=True,
        tolerance=0.15
    )
    
    # Select next comparison using acquisition function
    next_pair = acq_ucb(
        dkl_model=dkl_model,
        X_pool=X,
        previous_comparisons=previous_comparisons,
        beta=2.0
    )
    
    # Add to training set if needed
    for idx in next_pair:
        if idx not in train_indices:
            train_indices = np.append(train_indices, idx)
    X_train = X[train_indices]
    
    # Collect preference with tie and confidence support
    new_comp, new_conf = acquire_preference(
        # ... visualization parameters ...
        mode='human',  # or 'simulated'
        allow_ties=True,
        confidence_factors=[0.5, 0.75, 1.0]
    )
    
    # Update comparisons
    train_comp = torch.cat([train_comp, new_comp], dim=0)
    confidence = torch.cat([confidence, new_conf], dim=0)
    
    # Track this pair
    previous_comparisons.add((next_pair[0], next_pair[1]))
    
    print(f"Iteration {iteration+1}: {len(train_comp)} comparisons")
```

## Paper

TBD
```
