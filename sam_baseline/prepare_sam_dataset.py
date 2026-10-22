import os
import argparse
import nibabel as nib
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True)
parser.add_argument("--out_dir", required=True)
args = parser.parse_args()

INPUT_DIR = args.input_dir
OUT_DIR = args.out_dir

images_in = os.path.join(INPUT_DIR, "imagesTr")
labels_in = os.path.join(INPUT_DIR, "labelsTr")
metadata_path = os.path.join(INPUT_DIR, "metadata.csv")

images_out = os.path.join(OUT_DIR, "images")
masks_out = os.path.join(OUT_DIR, "masks")

os.makedirs(images_out, exist_ok=True)
os.makedirs(masks_out, exist_ok=True)

metadata = None
if os.path.exists(metadata_path):
    metadata = pd.read_csv(metadata_path)

slice_metadata = []


def normalize_to_uint8(img):
    img = img.astype(np.float32)
    img = np.clip(img, -1000, 1000)
    img = (img + 1000) / 2000.0
    return (img * 255).astype(np.uint8)


for img_file in tqdm(sorted(os.listdir(images_in))):
    if not img_file.endswith(".nii.gz"):
        continue

    case_id = img_file.replace("_0000.nii.gz", "")
    lbl_file = f"{case_id}.nii.gz"

    img_vol = nib.load(os.path.join(images_in, img_file)).get_fdata()
    lbl_vol = nib.load(os.path.join(labels_in, lbl_file)).get_fdata()

    for z in range(img_vol.shape[-1]):
        img_slice = normalize_to_uint8(img_vol[:, :, z])
        mask_slice = lbl_vol[:, :, z].astype(np.uint8)

        mask_binary = (mask_slice > 0).astype(np.uint8) * 255

        if mask_binary.sum() == 0:
            continue

        img_rgb = cv2.cvtColor(img_slice, cv2.COLOR_GRAY2RGB)

        out_name = f"{case_id}_slice{z:03d}.png"

        cv2.imwrite(os.path.join(images_out, out_name), img_rgb)
        cv2.imwrite(os.path.join(masks_out, out_name), mask_binary)

        row = {
            "file": out_name,
            "case_id": case_id,
            "slice": z
        }

        if metadata is not None:
            case_meta = metadata[metadata["case_id"] == case_id]
            if len(case_meta) > 0:
                row["class"] = case_meta.iloc[0]["class"]
                row["variations"] = case_meta.iloc[0]["variations"]
                row["noise_sigma"] = case_meta.iloc[0]["noise_sigma"]
                row["blur_sigma"] = case_meta.iloc[0]["blur_sigma"]
                row["intensity_scale"] = case_meta.iloc[0]["intensity_scale"]

        slice_metadata.append(row)

pd.DataFrame(slice_metadata).to_csv(
    os.path.join(OUT_DIR, "slice_metadata.csv"),
    index=False
)

print(f"Done. Created {OUT_DIR}/images and {OUT_DIR}/masks")
print(f"Saved metadata: {OUT_DIR}/slice_metadata.csv")
