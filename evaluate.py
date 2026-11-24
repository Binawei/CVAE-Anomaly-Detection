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
from dataset import get_test_loaders, get_test_loaders_all_normal, get_test_loaders_with_meta


def _tissue_mask(x, threshold=0.05):
    """Binary mask of pixels above background threshold. x: (B,1,H,W) tensor."""
    return (x > threshold).float()


def _masked_error(orig, recon, mask):
    """
    Two scores per image:
      - masked_mean : mean absolute error inside tissue region only
      - topk_mean   : mean of top 10% highest-error pixels inside tissue
    orig, recon, mask: (1,H,W) cpu tensors
    """
    err = torch.abs(orig - recon)  # (1,H,W)
    tissue_pixels = err[mask.bool()]  # 1D tensor of errors inside tissue
    if tissue_pixels.numel() == 0:
        return 0.0, 0.0
    masked_mean = tissue_pixels.mean().item()
    k = max(1, int(0.10 * tissue_pixels.numel()))
    topk_mean = tissue_pixels.topk(k).values.mean().item()
    return masked_mean, topk_mean


def reconstruction_error(model, loader):
    """Returns three parallel lists: global_errors, masked_errors, topk_errors."""
    model.eval()
    global_errors, masked_errors, topk_errors = [], [], []
    with torch.no_grad():
        for x in tqdm(loader, leave=False):
            x = x.to(config.DEVICE)
            recon, _, _ = model(x)
            x_cpu = x.cpu()
            recon_cpu = recon.cpu()
            mask = _tissue_mask(x_cpu)
            for i in range(x_cpu.size(0)):
                # original global score (kept for reference)
                mse = F.mse_loss(recon_cpu[i:i + 1], x_cpu[i:i + 1], reduction="sum").item()
                ssim_val = (1 - ssim(recon_cpu[i:i + 1], x_cpu[i:i + 1], data_range=1.0, size_average=True)).item()
                global_errors.append(mse + ssim_val)
                # new scores
                m, t = _masked_error(x_cpu[i], recon_cpu[i], mask[i])
                masked_errors.append(m)
                topk_errors.append(t)
    return global_errors, masked_errors, topk_errors


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


def plot_error_heatmaps(model, loader, label="Abnormal", max_patients=5, max_slices_per_patient=6):
    """For each patient, plot all their slices as: original | reconstructed | error heatmap."""
    model.eval()

    # Collect all images grouped by patient
    patient_data = {}  # patient_id -> list of (image_tensor, recon_tensor, heatmap)
    with torch.no_grad():
        for imgs, patient_ids, _ in tqdm(loader, desc=f"Heatmap [{label}]", leave=False):
            imgs = imgs.to(config.DEVICE)
            recon, _, _ = model(imgs)
            imgs_cpu = imgs.cpu()
            recon_cpu = recon.cpu()
            for i in range(imgs_cpu.size(0)):
                pid = patient_ids[i]
                heatmap = torch.abs(imgs_cpu[i] - recon_cpu[i])  # (1, H, W)
                if pid not in patient_data:
                    patient_data[pid] = []
                patient_data[pid].append((imgs_cpu[i, 0], recon_cpu[i, 0], heatmap[0]))

    patients = list(patient_data.keys())[:max_patients]
    for pid in patients:
        slices = patient_data[pid][:max_slices_per_patient]
        n = len(slices)
        fig, axes = plt.subplots(n, 3, figsize=(9, n * 3))
        if n == 1:
            axes = [axes]
        for row, (orig, recon, hmap) in enumerate(slices):
            axes[row][0].imshow(orig.numpy(), cmap="gray")
            axes[row][0].axis("off")
            axes[row][1].imshow(recon.numpy(), cmap="gray")
            axes[row][1].axis("off")
            im = axes[row][2].imshow(hmap.numpy(), cmap="hot", vmin=0, vmax=1)
            axes[row][2].axis("off")
            plt.colorbar(im, ax=axes[row][2], fraction=0.046, pad=0.04)
        axes[0][0].set_title("Original", fontsize=10)
        axes[0][1].set_title("Reconstructed", fontsize=10)
        axes[0][2].set_title("Error Heatmap", fontsize=10)
        safe_pid = pid.replace("/", "_").replace(" ", "_")
        plt.suptitle(f"{label} — {pid}", fontsize=11, y=1.01)
        plt.tight_layout()
        out_path = os.path.join(config.CHECKPOINT_DIR, f"heatmap_{label.lower()}_{safe_pid}.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")


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

    normal_loader, abnormal_loader = get_test_loaders_all_normal()
    print(f"  Using ALL normal images for evaluation (not just test split)")

    print("Visualising reconstructions...")
    visualize_reconstructions(model, normal_loader=normal_loader, abnormal_loader=abnormal_loader)

    print("Plotting error heatmaps...")
    normal_meta_loader, abnormal_meta_loader = get_test_loaders_with_meta()
    plot_error_heatmaps(model, abnormal_meta_loader, label="Abnormal", max_patients=5, max_slices_per_patient=6)
    plot_error_heatmaps(model, normal_meta_loader, label="Normal", max_patients=3, max_slices_per_patient=4)

    print("Computing reconstruction errors...")
    n_global, n_masked, n_topk = reconstruction_error(model, normal_loader)
    a_global, a_masked, a_topk = reconstruction_error(model, abnormal_loader)

    # --- Print means ---
    print(f"\n{'Metric':<20} {'Normal':>20} {'Abnormal':>20}")
    print("-" * 62)
    for name, n_scores, a_scores in [
        ("Global (MSE+SSIM)", n_global, a_global),
        ("Masked mean", n_masked, a_masked),
        ("Top-10% mean", n_topk, a_topk),
    ]:
        print(f"{name:<20} {np.mean(n_scores):>10.4f}±{np.std(n_scores):<8.4f} "
              f"{np.mean(a_scores):>10.4f}±{np.std(a_scores):<8.4f}")

    # --- CSV ---
    records = [{"label": 0, "global": g, "masked": m, "topk": t}
               for g, m, t in zip(n_global, n_masked, n_topk)]
    records += [{"label": 1, "global": g, "masked": m, "topk": t}
                for g, m, t in zip(a_global, a_masked, a_topk)]
    df = pd.DataFrame(records)
    csv_path = os.path.join(config.CHECKPOINT_DIR, "reconstruction_errors.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV to {csv_path}")

    labels = [0] * len(n_global) + [1] * len(a_global)

    # --- Histogram + ROC for each scoring method ---
    for name, n_scores, a_scores, fname in [
        ("Global (MSE+SSIM)", n_global, a_global, "global"),
        ("Masked Mean", n_masked, a_masked, "masked"),
        ("Top-10% Mean", n_topk, a_topk, "topk"),
    ]:
        scores = n_scores + a_scores

        plt.figure(figsize=(8, 5))
        plt.hist(n_scores, bins=30, alpha=0.6, label="Normal", color="blue")
        plt.hist(a_scores, bins=30, alpha=0.6, label="Abnormal", color="red")
        plt.xlabel(f"Reconstruction Error ({name})")
        plt.ylabel("Count")
        plt.title(f"Error Distribution — {name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(config.CHECKPOINT_DIR, f"histogram_{fname}.png"))
        plt.show()
        plt.close()

        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color="darkorange", label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve — {name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(config.CHECKPOINT_DIR, f"roc_{fname}.png"))
        plt.show()
        plt.close()
        print(f"  {name:<20} AUC = {roc_auc:.3f}")


if __name__ == "__main__":
    evaluate()
