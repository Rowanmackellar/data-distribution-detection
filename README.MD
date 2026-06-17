## Data Distribution && Model Failure Detection

This repository implements and evaluates over four different failure detection paradigms to predict when a ResNet-18 model will fail under distribution shifts(blur and noise) on the CIFAR-10 dataset

## SETUP:

1: create a virtual environment; 
    ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt  # Or list torch, scikit-learn, tensorboard.

2: train the model;

    python train.py

3: evaluate the detectors;

    python evaluate.py

4: inspect failures;

    python inspect_failures

## DETECTION METHODS IMPLEMENTED:
Confidence-Based: Softmax distribution entropy (Single forward pass).
Monte Carlo Dropout: Predictive entropy averaged over 15 stochastic passes.
Distance-Based: Mean k-NN distance from clean training embeddings.
OOD Anomaly Detection: Unsupervised One-Class SVM trained on clean data features.

## 4. Additional Failure Inspection
in the 4th step we highlight the core focus of this repository-
Here are examples of images where our ResNet-18 model was highly confident, but completely wrong due to distribution shifts:

![Confident Failures Gallery](plots/confident_failures.png)

