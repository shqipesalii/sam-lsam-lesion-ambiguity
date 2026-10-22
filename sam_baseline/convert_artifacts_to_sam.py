import os
import cv2
import nibabel as nib
import numpy as np

INPUT_ROOT = "Artifact_Datasets"
OUTPUT_ROOT = "Artifact_SAM_Datasets"

IMG_SUBDIR = "imagesTr"
LBL_SUBDIR = "labelsTr"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

def normalize_to_uint8(img):
    img = img.astype(np.float32)
    img = np.clip(img, -1000, 1000)
    img = (img + 1000) / 2000.0
    img = (img * 255).astype(np.uint8)
    return img

datasets = sorted([
    d for d in os.listdir(INPUT_ROOT)
    if os.path.isdir(os.path.join(INPUT_ROOT, d))
])

for dataset_name in datasets:
    in_img_dir = os.path.join(INPUT_ROOT, dataset_name, IMG_SUBDIR)
    in_lbl_dir = os.path.join(INPUT_ROOT, dataset_name, LBL_SUBDIR)

    out_img_dir = os.path.join(OUTPUT_ROOT, dataset_name, "images")
    out_msk_dir = os.path.join(OUTPUT_ROOT, dataset_name, "masks")

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_msk_dir, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(in_img_dir)
        if f.endswith(".nii.gz")
    ])

    saved = 0

    for img_file in image_files:
        case_id = img_file.replace("_0000.nii.gz", "")
        lbl_file = f"{case_id}.nii.gz"

        img_path = os.path.join(in_img_dir, img_file)
        lbl_path = os.path.join(in_lbl_dir, lbl_file)

        if not os.path.exists(lbl_path):
            print(f"Missing label for {case_id}")
            continue

        img_vol = nib.load(img_path).get_fdata()
        lbl_vol = nib.load(lbl_path).get_fdata()

        for z in range(img_vol.shape[-1]):
            img_slice = img_vol[:, :, z]
            lbl_slice = lbl_vol[:, :, z]

            mask = (lbl_slice > 0).astype(np.uint8)

            if mask.sum() == 0:
                continue

            img_uint8 = normalize_to_uint8(img_slice)
            img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)

            out_name = f"{case_id}_slice{z:03d}.png"

            cv2.imwrite(
                os.path.join(out_img_dir, out_name),
                img_rgb
            )

            cv2.imwrite(
                os.path.join(out_msk_dir, out_name),
                mask * 255
            )

            saved += 1

    print(f"{dataset_name}: saved {saved} SAM slices")

print("\nDone. Converted all artifact datasets to SAM format.")
