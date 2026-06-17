import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from train import get_device
from src.models import ResNetCIFAR
from src.datasets import get_clean_dataloaders, get_shifted_dataloader
from detection_methods.detectors import (
    evaluate_confidence,
    evaluate_distance,
    evaluate_mc_dropout,
    train_ood_detector,
    evaluate_ood,
)
# Import your newly updated plotting functions
from plots import plot_calibration, plot_acc_vs_confidence

SHIFT_TYPES = ["blur", "noise"]

@torch.no_grad()
def get_probabilities_and_targets(model, loader, device):
    """Helper to extract raw softmax probabilities and true targets for calibration plots."""
    model.eval()
    all_probs, all_targets = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        probs = F.softmax(outputs, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_targets.extend(labels.numpy())
    return np.concatenate(all_probs, axis=0), np.array(all_targets)


def run_evaluation(model, train_loader, ood_clf, shift_type, device):
    """Runs failure detection evaluations over a specific distribution shift."""
    shifted_loader = get_shifted_dataloader(shift_type=shift_type)
    results = {}

    # ------------------------------------------------------------------
    # 1. CONFIDENCE — single forward pass, softmax entropy
    # ------------------------------------------------------------------
    scores, preds, targets = evaluate_confidence(model, shifted_loader, device)
    failures = (preds != targets).astype(int)
    results["confidence"] = roc_auc_score(failures, scores)

    # ------------------------------------------------------------------
    # 2. DISTANCE — k-NN distance from clean training embeddings
    # ------------------------------------------------------------------
    scores, preds, targets = evaluate_distance(model, train_loader, shifted_loader, device)
    failures = (preds != targets).astype(int)
    results["distance"] = roc_auc_score(failures, scores)

    # ------------------------------------------------------------------
    # 3. MONTE CARLO DROPOUT — entropy over 15 stochastic passes
    # ------------------------------------------------------------------
    scores, preds, targets = evaluate_mc_dropout(model, shifted_loader, device)
    failures = (preds != targets).astype(int)
    results["mc_dropout"] = roc_auc_score(failures, scores)

    # ------------------------------------------------------------------
    # 4. OOD DETECTOR — One-Class SVM trained on clean data only
    # ------------------------------------------------------------------
    scores, preds, targets = evaluate_ood(model, ood_clf, shifted_loader, device)
    failures = (preds != targets).astype(int)
    results["ood"] = roc_auc_score(failures, scores)

    # ------------------------------------------------------------------
    # STEP 4 VISUALIZATIONS — Generate Calibration plots per shift
    # ------------------------------------------------------------------
    print(f"Generating calibration charts for shift: {shift_type}...")
    shift_probs, shift_targets = get_probabilities_and_targets(model, shifted_loader, device)
    plot_calibration(shift_probs, shift_targets, save_path=f"plots/calibration_{shift_type}.png", dataset_name=f"Shift: {shift_type}")
    plot_acc_vs_confidence(shift_probs, shift_targets, save_path=f"plots/acc_vs_conf_{shift_type}.png", dataset_name=f"Shift: {shift_type}")

    return results


def print_results(all_results):
    detectors = ["confidence", "distance", "mc_dropout", "ood"]
    col_w = 14

    header = f"{'Shift':<12}" + "".join(f"{d:>{col_w}}" for d in detectors)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for shift_type, results in all_results.items():
        row = f"{shift_type:<12}" + "".join(f"{results[d]:>{col_w}.3f}" for d in detectors)
        print(row)

    print("=" * len(header) + "\n")


if __name__ == "__main__":
    device = get_device()

    model = ResNetCIFAR().to(device)
    model.load_state_dict(torch.load("models/checkpoints/resnet18_best.pt", map_location=device))
    model.eval()

    print("Loading clean data...")
    train_loader, clean_test_loader = get_clean_dataloaders()

    # ------------------------------------------------------------------
    # BASELINE VISUALIZATION — Generate Calibration plots for Clean Data
    # ------------------------------------------------------------------
    print("Generating clean baseline calibration charts...")
    clean_probs, clean_targets = get_probabilities_and_targets(model, clean_test_loader, device)
    plot_calibration(clean_probs, clean_targets, save_path="plots/calibration_clean.png", dataset_name="Clean Baseline")
    plot_acc_vs_confidence(clean_probs, clean_targets, save_path="plots/acc_vs_conf_clean.png", dataset_name="Clean Baseline")

    print("Training unsupervised OOD detector (One-Class SVM)...")
    ood_clf = train_ood_detector(model, train_loader, device)

    all_results = {}
    for shift_type in SHIFT_TYPES:
        print(f"Evaluating shift: {shift_type}...")
        all_results[shift_type] = run_evaluation(model, train_loader, ood_clf, shift_type, device)

    print_results(all_results)