import os
import nibabel as nib
import numpy as np
from PIL import Image

import config

DATA_DIR = './data'
OUT_NORMAL = './png_data/normal'
OUT_ABNORMAL = './png_data/abnormal'

def process():
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".nii.gz") or f.endswith(".nii")]
    normal, abnormal, skipped = 0, 0, 0

    for fname in files:
        is_normal = "norm" in fname.lower()
        out_dir = OUT_NORMAL if is_normal else OUT_ABNORMAL
        stem = fname.replace(".nii.gz", "").replace(".nii", "")
        fpath = os.path.join(DATA_DIR, fname)

        try:
            vol = nib.load(fpath).get_fdata()
        except Exception:
            try:
                tmp = fpath.replace(".nii.gz", ".nii")
                os.rename(fpath, tmp)
                vol = nib.load(tmp).get_fdata()
                os.rename(tmp, fpath)
            except Exception as e:
                print(f"Skipping {fname}: {e}")
                skipped += 1
                continue

        for i in range(vol.shape[2]):
            slc = np.ascontiguousarray(vol[:, :, i].astype(np.float32))
            slc = (slc - slc.min()) / (slc.max() - slc.min() + 1e-8)
            img = Image.fromarray((slc * 255).astype(np.uint8)).convert("L").resize(config.IMG_TUPLE, Image.BILINEAR)
            img.save(os.path.join(out_dir, f"{stem}_s{i}.png"))

        if is_normal:
            normal += 1
        else:
            abnormal += 1

    print(f"Done — normal: {normal} scans, abnormal: {abnormal} scans, skipped: {skipped}")
    print(f"Normal PNGs: {len(os.listdir(OUT_NORMAL))}, Abnormal PNGs: {len(os.listdir(OUT_ABNORMAL))}")


if __name__ == "__main__":
    process()
