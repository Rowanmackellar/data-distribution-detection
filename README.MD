## Data Distribution Detection
An accentuation of an emergence in AI failures revolving around the constantly changing world of data. But even with countless effort towards helping to track the path of shifting data; the many sophistacated methods aren't even now, an absolute response to this troubling issue.

This Repository implements and evaluates four failure detection paradigms under two distribution shifts (Gaussian blur and Gaussian noise).
## Setup

1. Create a virtual environment and install dependencies:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Train the model:

   ```bash
   python train.py
   ```

3. Evaluate the detectors:

   ```bash
   python evaluate.py
   ```

4. Inspect individual failures:

   ```bash
   python inspect_failures.py
   ```

## Detection Methods Implemented

- **Confidence-based:** Softmax distribution entropy (single forward pass).
- **Monte Carlo Dropout:** Predictive entropy averaged over 15 stochastic forward passes.
- **Distance-based:** Mean k-NN distance from clean training embeddings.
- **OOD detection:** Unsupervised One-Class SVM trained only on clean (in-distribution) features.

Each method scores how likely a prediction is to be wrong. `evaluate.py` turns that score into an AUROC against actual model errors, for both the blur and noise shifts.

## Results

AUROC for detecting actual model errors, per method, per shift (0.5 = random, 1.0 = perfect):

| Shift | Confidence | Distance | MC Dropout | OOD (SVM) |
|-------|:----------:|:--------:|:----------:|:---------:|
| Blur  | 0.652      | 0.581    | 0.652      | 0.434     |
| Noise | 0.629      | 0.550    | 0.630      | 0.489     |

## Additional Failure Inspection

The core focus of this repository is highlighting *confidently wrong* predictions — cases where the model assigns high softmax confidence to the wrong class under distribution shift, the inspect failures call allows you you to have a display of multiple areas where the model failed.

