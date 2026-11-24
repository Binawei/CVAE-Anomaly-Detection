import json, torch, sys, os

import numpy as np

sys.path.insert(0, '.')

import config

from data_loader import prepare_validation_data, to_tensor

from combine_maps import combine_maps_hard

from metrics import compute_all_metrics

val_data = prepare_validation_data()

val_tensors = [to_tensor(d) for d in val_data]

best_params = json.load(open('results/best_params.json'))

for d in val_tensors:
    # load precomputed image error map

    case_name = d['case_name']  # e.g. "CC21460186Day of ETSDMODA2_s0"

    base = case_name.rsplit('_s', 1)

    scan = base[0]

    slice_num = base[1]

    err_path = os.path.join(config.ERROR_MAPS_DIR, 'abnormal', 'image_errors', f"{scan}_s{slice_num}.npy")

    img_error = torch.from_numpy(

        np.fliplr(np.rot90(np.load(err_path).astype(np.float32), k=1)).copy()

    ).to(config.DEVICE)

    pred = combine_maps_hard(

        img_error, d['muscle'], d['endometrium'], d['cavity'],

        torch.tensor(best_params['threshold_img']),

        torch.tensor(best_params['threshold_muscle']),

        torch.tensor(best_params['threshold_endo']),

        torch.tensor(best_params['threshold_cavity']),

    )

    m = compute_all_metrics(pred, d['annotation'])

    print(
        f"{case_name}: Dice={m['dice']:.4f}, IoU={m['iou']:.4f}, Precision={m['precision']:.4f}, Recall={m['recall']:.4f}")

