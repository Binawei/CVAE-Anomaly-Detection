import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
from pytorch_msssim import ssim

import config
from model import CVAE
from dataset import get_test_loaders


def reconstruction_error(model, loader):
    model.eval()
    errors = []
    with torch.no_grad():
        for x in tqdm(loader, leave=False):
            x = x.to(config.DEVICE)
            recon, _, _ = model(x)
            for i in range(x.size(0)):
                mse = F.mse_loss(recon[i:i + 1], x[i:i + 1], reduction="sum").item()
                ssim_loss = (1 - ssim(recon[i:i + 1], x[i:i + 1], data_range=1.0, size_average=True)).item()
                errors.append(mse + ssim_loss)
    return errors


def _get_pairs(model, loader, n):
    originals, reconstructions = [], []
    with torch.no_grad():
        for x in loader:
            x = x.to(config.DEVICE)
            recon, _, _ = model(x)
            originals.append(x.cpu())
            reconstructions.append(recon.cpu())
            if sum(t.size(0) for t in originals) >= n:
                break
    return torch.cat(originals)[:n], torch.cat(reconstructions)[:n]


def visualize_reconstructions(model, normal_loader=None, abnormal_loader=None, n=8):
    model.eval()
    assert normal_loader is not None or abnormal_loader is not None, \
        "At least one of normal_loader or abnormal_loader must be provided"
    sections = []
    if normal_loader is not None:
        sections.append(("Normal", *_get_pairs(model, normal_loader, n)))
    if abnormal_loader is not None:
        sections.append(("Abnormal", *_get_pairs(model, abnormal_loader, n)))

    total_rows = sum(orig.size(0) for _, orig, _ in sections)
    fig, axes = plt.subplots(total_rows, 2, figsize=(6, total_rows * 2))
    if total_rows == 1:
        axes = [axes]

    row = 0
    for s_idx, (label, originals, reconstructions) in enumerate(sections):
        if s_idx > 0:
            # add a text separator by leaving a blank row with a label
            axes[row - 1][0].set_xlabel(f"── {label} ──", fontsize=9, labelpad=10)

        for i in range(originals.size(0)):
            axes[row][0].imshow(originals[i, 0], cmap="gray")
            axes[row][0].axis("off")
            axes[row][1].imshow(reconstructions[i, 0], cmap="gray")
            axes[row][1].axis("off")
            row += 1

    axes[0][0].set_title("Original", fontsize=10)
    axes[0][1].set_title("Reconstructed", fontsize=10)

    # section labels on the left
    row = 0
    for label, originals, _ in sections:
        axes[row][0].set_ylabel(label, fontsize=10, rotation=90, labelpad=10)
        row += originals.size(0)

    plt.suptitle("Original vs Reconstructed", y=1.01)
    plt.tight_layout()
    vis_path = os.path.join(config.CHECKPOINT_DIR, "reconstructions.png")
    plt.savefig(vis_path, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved reconstructions to {vis_path}")


def evaluate():
    config.setup_reproducibility()
    config.empty_cache()
    print(f"Using device: {config.DEVICE}")

    model = CVAE(config.LATENT_DIM).to(config.DEVICE)
    model.load_state_dict(torch.load(os.path.join(config.CHECKPOINT_DIR, "best_model.pth"),
                                     map_location=config.DEVICE))

    normal_loader, abnormal_loader = get_test_loaders()

    print("Visualising reconstructions...")
    # visualize_reconstructions(model, abnormal_loader=abnormal_loader)
    # visualize_reconstructions(model, normal_loader=normal_loader)
    visualize_reconstructions(model, normal_loader=normal_loader, abnormal_loader=abnormal_loader)

    print("Computing reconstruction errors...")
    normal_errors = reconstruction_error(model, normal_loader)
    abnormal_errors = reconstruction_error(model, abnormal_loader)

    # --- Print means ---
    print(f"\nMean reconstruction error:")
    print(f"  Normal   : {np.mean(normal_errors):.4f} ± {np.std(normal_errors):.4f}")
    print(f"  Abnormal : {np.mean(abnormal_errors):.4f} ± {np.std(abnormal_errors):.4f}")

    # --- CSV ---
    records = [{"file": "normal", "label": 0, "error": e} for e in normal_errors]
    records += [{"file": "abnormal", "label": 1, "error": e} for e in abnormal_errors]
    df = pd.DataFrame(records)
    csv_path = os.path.join(config.CHECKPOINT_DIR, "reconstruction_errors.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")

    # --- Histogram ---
    plt.figure(figsize=(8, 5))
    plt.hist(normal_errors, bins=30, alpha=0.6, label="Normal", color="blue")
    plt.hist(abnormal_errors, bins=30, alpha=0.6, label="Abnormal", color="red")
    plt.xlabel("Reconstruction Error (MSE + SSIM)")
    plt.ylabel("Count")
    plt.title("Reconstruction Error Distribution")
    plt.legend()
    plt.tight_layout()
    hist_path = os.path.join(config.CHECKPOINT_DIR, "error_histogram.png")
    plt.savefig(hist_path)
    plt.show()
    plt.close()
    print(f"Saved histogram to {hist_path}")

    # --- ROC + AUC ---
    labels = [0] * len(normal_errors) + [1] * len(abnormal_errors)
    scores = normal_errors + abnormal_errors
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="darkorange", label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    roc_path = os.path.join(config.CHECKPOINT_DIR, "roc_curve.png")
    plt.savefig(roc_path)
    plt.show()
    plt.close()
    print(f"Saved ROC curve to {roc_path}")
    print(f"\nAUC: {roc_auc:.3f}")


if __name__ == "__main__":
    evaluate()
