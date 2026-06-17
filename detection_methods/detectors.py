import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression

# CONFIDENCE BASED
# Runs a single forward pass, no extra overhead.

@torch.no_grad()
def evaluate_confidence(model, loader, device):
    """
    Returns per-sample entropy of the softmax distribution.
    High entropy = spread-out probabilities = uncertain prediction.
    """
    model.eval()
    all_entropy, all_preds, all_labels = [], [], []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        probs = F.softmax(outputs, dim=1).cpu().numpy()

        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)

        all_entropy.extend(entropy)
        all_preds.extend(np.argmax(probs, axis=1))
        all_labels.extend(labels.numpy())

    return np.array(all_entropy), np.array(all_preds), np.array(all_labels)
    
# MONTE CARLO DROPOUT
# Runs N stochastic forward passes with dropout active, averages the softmax
# probabilities, then computes entropy over the mean distribution.

def evaluate_mc_dropout(model, loader, device, num_samples=15):
    """
    Returns per-sample predictive entropy averaged over MC samples.
    Requires model.forward() to accept mc_dropout=True.
    """
    model.eval()
    all_entropy, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            mc_probs = []

            for _ in range(num_samples):
                outputs = model(imgs, mc_dropout=True)
                mc_probs.append(F.softmax(outputs, dim=1).cpu().numpy())

            # Shape: (num_samples, batch, classes) -> average over samples
            expected_probs = np.mean(np.stack(mc_probs, axis=0), axis=0)
            entropy = -np.sum(expected_probs * np.log(expected_probs + 1e-10), axis=1)

            all_entropy.extend(entropy)
            all_preds.extend(np.argmax(expected_probs, axis=1))
            all_labels.extend(labels.numpy())

    return np.array(all_entropy), np.array(all_preds), np.array(all_labels)

# DISTANCE-BASED
# Extracts embeddings from the model's penultimate layer, fits a k-NN index on
# clean training data, then scores test samples by their distance to the k
# nearest training neighbors; Far from training data = likely OOD = likely fail.

@torch.no_grad()
def extract_embeddings_and_logits(model, loader, device):
    """Helper — extract embeddings AND classification predictions simultaneously."""
    model.eval()
    all_embeddings, all_preds, all_labels = [], [], []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        
        # Modify your ResNetCIFAR model to return both features and logits
        # or captures via hook, assuming both work...
        outputs = model(imgs) 
        embeddings = model.get_embeddings(imgs).cpu().numpy()
        
        all_embeddings.append(embeddings)
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.concatenate(all_embeddings, axis=0), np.array(all_preds), np.array(all_labels)


def fit_knn(train_embeddings, k=10):
    """Fit a k-NN index on clean training embeddings."""
    knn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
    knn.fit(train_embeddings)
    return knn


def evaluate_distance(model, train_loader, test_loader, device, k=10):
    """
    Returns mean k-NN distance to training set for each test sample.
    High distance = far from training distribution = likely to fail.
    """
    # Extract everything in a single, unified pass
    train_embeddings, _, _ = extract_embeddings_and_logits(model, train_loader, device)
    test_embeddings, test_preds, test_labels = extract_embeddings_and_logits(model, test_loader, device)

    # Compute k-NN distances
    knn = fit_knn(train_embeddings, k=k)
    distances, _ = knn.kneighbors(test_embeddings)
    mean_distances = distances.mean(axis=1)

    return mean_distances, test_preds, test_labels

#  OOD DETECTOR
# Trains a logistic regression binary classifier on embeddings:
#   - Clean training samples = label 0 (in-distribution)
#   - Shifted test samples   = label 1 (out-of-distribution)
# The classifier's predicted probability of being OOD is the failure score.

from sklearn.svm import OneClassSVM

def train_ood_detector(model, clean_loader, device):
    """
    Trains an unsupervised One-Class SVM on CLEAN training data only.
    No shifted data is leaked during training.
    """
    clean_embeddings, _, _ = extract_embeddings_and_logits(model, clean_loader, device)
    
    # Subsample if the training set is huge to speed up fitting
    if len(clean_embeddings) > 10000:
        indices = np.random.choice(len(clean_embeddings), 10000, replace=False)
        clean_embeddings = clean_embeddings[indices]

    # gamma='scale' or 'auto' handles high-dimensional space
    clf = OneClassSVM(nu=0.05, kernel="rbf", gamma="scale")
    clf.fit(clean_embeddings)
    return clf

def evaluate_ood(model, ood_clf, test_loader, device):
    """
    Scores each test sample based on its distance from the clean distribution.
    Invert score because lower decision function values mean MORE out-of-distribution.
    """
    # Extract everything in a single, unified pass
    test_embeddings, test_preds, test_labels = extract_embeddings_and_logits(model, test_loader, device)

    # 2. Compute OOD anomaly scores
    # score_samples returns the log density; inverting it means higher score = higher OOD likelihood.
    ood_scores = -ood_clf.score_samples(test_embeddings)

    return ood_scores, test_preds, test_labels
