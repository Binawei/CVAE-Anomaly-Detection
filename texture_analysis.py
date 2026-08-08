import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import stats
from skimage.filters import sobel
from skimage.measure import shannon_entropy
from tqdm import tqdm

import config


def load_images(img_dir):
    paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")])
    images = []
    for p in tqdm(paths, desc=f"Loading {os.path.basename(img_dir)}", leave=False):
        img = np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0
        images.append(img)
    return images


def tissue_mask(img, threshold=0.05):
    return img > threshold


def compute_metrics(images):
    variances, gradients, entropies = [], [], []
    for img in images:
        mask = tissue_mask(img)
        tissue = img[mask]
        if tissue.size == 0:
            continue
        # Variance — spread of pixel intensities inside tissue
        variances.append(float(np.var(tissue)))
        # Gradient magnitude — edge/texture density inside tissue
        grad = sobel(img)
        gradients.append(float(np.mean(grad[mask])))
        # Shannon entropy — information content of full image
        entropies.append(float(shannon_entropy(img)))
    return variances, gradients, entropies


def plot_distributions(normal_vals, abnormal_vals, metric_name, fname):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    axes[0].hist(normal_vals, bins=30, alpha=0.6, color="blue", label="Normal")
    axes[0].hist(abnormal_vals, bins=30, alpha=0.6, color="red", label="Abnormal")
    axes[0].set_xlabel(metric_name)
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"{metric_name} Distribution")
    axes[0].legend()

    # Boxplot
    axes[1].boxplot([normal_vals, abnormal_vals], labels=["Normal", "Abnormal"],
                    patch_artist=True,
                    boxprops=dict(facecolor="lightblue"),
                    medianprops=dict(color="black", linewidth=2))
    axes[1].set_ylabel(metric_name)
    axes[1].set_title(f"{metric_name} Boxplot")

    plt.tight_layout()
    out_path = os.path.join(config.CHECKPOINT_DIR, f"texture_{fname}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


def analyse():
    print("Loading images...")
    normal_images = load_images(config.NORMAL_DIR)
    abnormal_images = load_images(config.ABNORMAL_DIR)
    print(f"  Normal: {len(normal_images)} images")
    print(f"  Abnormal: {len(abnormal_images)} images")

    print("\nComputing texture metrics...")
    n_var, n_grad, n_ent = compute_metrics(normal_images)
    a_var, a_grad, a_ent = compute_metrics(abnormal_images)

    metrics = [
        ("Variance", n_var, a_var, "variance"),
        ("Gradient Magnitude", n_grad, a_grad, "gradient"),
        ("Shannon Entropy", n_ent, a_ent, "entropy"),
    ]

    print(f"\n{'Metric':<22} {'Normal Mean':>14} {'Abnormal Mean':>14} {'p-value':>10} {'Conclusion':>20}")
    print("-" * 82)
    for name, n_vals, a_vals, fname in metrics:
        t_stat, p_val = stats.mannwhitneyu(n_vals, a_vals, alternative="two-sided")
        direction = "Normal > Abnormal" if np.mean(n_vals) > np.mean(a_vals) else "Abnormal > Normal"
        significant = "significant" if p_val < 0.05 else "not significant"
        print(f"{name:<22} {np.mean(n_vals):>14.6f} {np.mean(a_vals):>14.6f} "
              f"{p_val:>10.4f}  {direction} ({significant})")
        plot_distributions(n_vals, a_vals, name, fname)

    print("\nDone. All plots saved to checkpoints/")


if __name__ == "__main__":
    analyse()
