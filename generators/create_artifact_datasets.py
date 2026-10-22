import os
import shutil
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter


INPUT_DIR = "PhantomShapes_V2_NoEmpty"
OUTPUT_ROOT = "Artifact_Datasets"

IMG_SUBDIR = "imagesTr"
LBL_SUBDIR = "labelsTr"

N_RUNS = 10


ARTIFACT_LEVELS = {
    "moderate": {
        "alpha": 1.10,
        "beta": 10,
        "noise_sigma": 15,
        "blur_sigma": 0.3
    },
    "severe": {
        "alpha": 1.30,
        "beta": 25,
        "noise_sigma": 35,
        "blur_sigma": 1.0
    }
}


def apply_artifact(img, alpha, beta, noise_sigma, blur_sigma, seed):
    rng = np.random.default_rng(seed)

    img = img.astype(np.float32)

    # Brightness / contrast
    img = alpha * img + beta

    # Gaussian noise
    noise = rng.normal(0, noise_sigma, img.shape)
    img = img + noise

    # Blur
    img = gaussian_filter(img, sigma=blur_sigma)

    # Keep CT-like intensity range
    img = np.clip(img, -1000, 1000)

    return img.astype(np.int16)


def copy_metadata_files(src_dir, dst_dir):
    for fname in ["dataset.json", "seg_info.json", "metadata.csv"]:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, dst)


def create_artifact_dataset(level_name, run_id, settings):
    input_img_dir = os.path.join(INPUT_DIR, IMG_SUBDIR)
    input_lbl_dir = os.path.join(INPUT_DIR, LBL_SUBDIR)

    output_dir = os.path.join(
        OUTPUT_ROOT,
        f"V2_{level_name}_run{run_id:02d}"
    )

    output_img_dir = os.path.join(output_dir, IMG_SUBDIR)
    output_lbl_dir = os.path.join(output_dir, LBL_SUBDIR)

    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_lbl_dir, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(input_img_dir)
        if f.endswith(".nii.gz")
    ])

    for i, img_file in enumerate(image_files):
        img_path = os.path.join(input_img_dir, img_file)

        img_nii = nib.load(img_path)
        img = img_nii.get_fdata()

        seed = run_id * 10000 + i

        artifact_img = apply_artifact(
            img,
            alpha=settings["alpha"],
            beta=settings["beta"],
            noise_sigma=settings["noise_sigma"],
            blur_sigma=settings["blur_sigma"],
            seed=seed
        )

        out_img_path = os.path.join(output_img_dir, img_file)

        nib.save(
            nib.Nifti1Image(
                artifact_img,
                img_nii.affine,
                img_nii.header
            ),
            out_img_path
        )

    # Copy labels unchanged
    for lbl_file in os.listdir(input_lbl_dir):
        if lbl_file.endswith(".nii.gz"):
            shutil.copy(
                os.path.join(input_lbl_dir, lbl_file),
                os.path.join(output_lbl_dir, lbl_file)
            )

    copy_metadata_files(INPUT_DIR, output_dir)

    print(f"Created: {output_dir}")


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for level_name, settings in ARTIFACT_LEVELS.items():
        for run_id in range(1, N_RUNS + 1):
            create_artifact_dataset(level_name, run_id, settings)

    print("\nDone. Artifact datasets created.")


if __name__ == "__main__":
    main()

