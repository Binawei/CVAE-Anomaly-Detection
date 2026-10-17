"""
check_alignment.py
Verify annotation aligns with the original image for a specific slice.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import zoom
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import config

# ── Pick case and slice to check ──────────────────────────────────────────────
CASE_NAME = "CC21460186Day of ETSDMODA2"
SLICE_NUM = 0

# ── Load original PNG image ───────────────────────────────────────────────────
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
img_path = os.path.join(parent_dir, 'png_data', 'abnormal',
                        f"{CASE_NAME}_s{SLICE_NUM}.png")
print(f"Loading image: {img_path}")
img = np.array(Image.open(img_path).convert('L'))

# ── Load annotation slice ─────────────────────────────────────────────────────
ann_path = config.get_annotation_path(CASE_NAME)
print(f"Loading annotation: {ann_path}")
nii = nib.load(ann_path)
data = nii.get_fdata()
if data.ndim > 3:
    data = data[:, :, :, 0]
print(f"Annotation volume shape: {data.shape}")
print(f"Number of slices: {data.shape[2]}")
for s in range(data.shape[2]):
    print(f"  Slice {s}: {int((data[:, :, s] > 0.5).sum())} annotated pixels")

ann_raw = data[:, :, SLICE_NUM].squeeze()
ann_raw = (ann_raw > 0.5).astype(np.float32)
zoom_factors = (256 / ann_raw.shape[0], 256 / ann_raw.shape[1])
ann_resized = zoom(ann_raw, zoom_factors, order=0)

# ── Rotate and flip image to match 3D Slicer orientation exactly ─────────────
img_slicer = np.fliplr(np.rot90(img, k=1))

# ── Try all 8 transforms (also flipped to match Slicer) ──────────────────────
transforms = {
    'no transform':       np.fliplr(ann_resized.copy()),
    'fliplr':             np.fliplr(np.fliplr(ann_resized)),
    'flipud':             np.fliplr(np.flipud(ann_resized)),
    'fliplr + flipud':    np.fliplr(np.flipud(np.fliplr(ann_resized))),
    'rot90 k=1':          np.fliplr(np.rot90(ann_resized, k=1)),
    'rot90 k=3':          np.fliplr(np.rot90(ann_resized, k=3)),
    'rot90 k=1 + fliplr': np.fliplr(np.fliplr(np.rot90(ann_resized, k=1))),
    'rot90 k=3 + fliplr': np.fliplr(np.fliplr(np.rot90(ann_resized, k=3))),
}

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, (name, ann_t) in enumerate(transforms.items()):
    axes[i].imshow(img_slicer, cmap='gray', alpha=0.7)
    axes[i].imshow(ann_t, cmap='hot', alpha=0.5, vmin=0, vmax=1)
    axes[i].set_title(name, fontsize=9)
    axes[i].axis('off')

plt.suptitle(f'{CASE_NAME} — slice {SLICE_NUM} — find correct transform',
             fontsize=11)
plt.tight_layout()
plt.savefig('alignment_check.png', dpi=150, bbox_inches='tight')
print("\nSaved: alignment_check.png")
print("Look at each panel and identify which one correctly overlays the annotation on the image.")
