import os
import json
import csv
import argparse
import numpy as np
import nibabel as nib

from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from skimage.draw import disk, polygon
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--out_dir", type=str, required=True)
args = parser.parse_args()

np.random.seed(args.seed)

OUT_DIR = args.out_dir
IMG_SUBDIR = "imagesTr"
LBL_SUBDIR = "labelsTr"

N_SLICES = 5
IMG_DIM = 256
NUM_VOLUMES = 150
NUM_EMPTY = 30
NUM_WITH_LESIONS = NUM_VOLUMES - NUM_EMPTY

os.makedirs(os.path.join(OUT_DIR, IMG_SUBDIR), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, LBL_SUBDIR), exist_ok=True)

classes = {
    "background": 0,
    "circle": 1,
    "square": 2,
    "triangle": 3
}

metadata_rows = []


def create_base_phantom():
    phantom = shepp_logan_phantom()
    phantom = resize(phantom, (IMG_DIM, IMG_DIM), anti_aliasing=True)
    phantom = (phantom * 2000) - 1000
    return phantom.astype(np.float32)


def random_center():
    return np.random.randint(60, 196), np.random.randint(60, 196)


def create_circle_mask():
    mask = np.zeros((IMG_DIM, IMG_DIM), dtype=np.uint8)
    cy, cx = random_center()
    radius = np.random.randint(12, 28)
    rr, cc = disk((cy, cx), radius, shape=mask.shape)
    mask[rr, cc] = 1
    return mask


def create_square_mask():
    mask = np.zeros((IMG_DIM, IMG_DIM), dtype=np.uint8)
    cy, cx = random_center()
    size = np.random.randint(25, 45)
    y1 = max(0, cy - size // 2)
    y2 = min(IMG_DIM, cy + size // 2)
    x1 = max(0, cx - size // 2)
    x2 = min(IMG_DIM, cx + size // 2)
    mask[y1:y2, x1:x2] = 1
    return mask


def create_triangle_mask():
    mask = np.zeros((IMG_DIM, IMG_DIM), dtype=np.uint8)
    cy, cx = random_center()
    size = np.random.randint(30, 55)

    points = np.array([
        [cy - size // 2, cx],
        [cy + size // 2, cx - size // 2],
        [cy + size // 2, cx + size // 2]
    ])

    rr, cc = polygon(points[:, 0], points[:, 1], shape=mask.shape)
    mask[rr, cc] = 1
    return mask


def make_irregular(mask):
    random_field = np.random.rand(IMG_DIM, IMG_DIM)
    random_field = gaussian_filter(random_field, sigma=np.random.uniform(3, 8))
    random_field = random_field > np.percentile(random_field, 55)

    irregular = mask.copy().astype(bool)

    if np.random.rand() > 0.5:
        irregular = binary_dilation(irregular, iterations=np.random.randint(1, 3))
    else:
        irregular = binary_erosion(irregular, iterations=1)

    irregular = np.logical_and(irregular, random_field | mask.astype(bool))
    return irregular.astype(np.uint8)


def move_to_edge(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return mask

    shift_y = np.random.choice([-50, -35, 35, 50])
    shift_x = np.random.choice([-50, -35, 35, 50])

    new_mask = np.zeros_like(mask)
    new_y = np.clip(ys + shift_y, 0, IMG_DIM - 1)
    new_x = np.clip(xs + shift_x, 0, IMG_DIM - 1)
    new_mask[new_y, new_x] = 1

    return new_mask


def create_shape_mask(shape_name):
    if shape_name == "circle":
        return create_circle_mask()
    elif shape_name == "square":
        return create_square_mask()
    elif shape_name == "triangle":
        return create_triangle_mask()
    else:
        return np.zeros((IMG_DIM, IMG_DIM), dtype=np.uint8)


def add_noise_and_blur(img, noise_sigma, blur_sigma, intensity_scale):
    img = img * intensity_scale
    img = img + np.random.normal(0, noise_sigma, img.shape)

    if blur_sigma > 0:
        img = gaussian_filter(img, sigma=blur_sigma)

    img = np.clip(img, -1000, 1000)
    return img.astype(np.float32)


def generate_case(case_idx, empty=False):
    case_id = f"Case{case_idx:03d}"

    image_volume = np.zeros((IMG_DIM, IMG_DIM, N_SLICES), dtype=np.float32)
    label_volume = np.zeros((IMG_DIM, IMG_DIM, N_SLICES), dtype=np.uint8)

    base = create_base_phantom()

    if empty:
        for z in range(N_SLICES):
            noise_sigma = np.random.choice([8, 12, 18, 25, 35])
            blur_sigma = np.random.choice([0.3, 0.5, 0.8, 1.1])
            intensity_scale = np.random.uniform(0.85, 1.15)
            image_volume[:, :, z] = add_noise_and_blur(base, noise_sigma, blur_sigma, intensity_scale)

        lesion_class = "empty"
        variations = "empty"
        noise_sigma = 0
        blur_sigma = 0
        intensity_scale = 1.0

    else:
        lesion_class = np.random.choice(["circle", "square", "triangle"])
        class_label = classes[lesion_class]

        variations_list = []

        use_irregular = np.random.rand() < 0.65
        use_multiple = np.random.rand() < 0.35
        use_low_contrast = np.random.rand() < 0.45
        use_partial_edge = np.random.rand() < 0.35

        if use_irregular:
            variations_list.append("irregular")
        if use_multiple:
            variations_list.append("multiple_lesions")
        if use_low_contrast:
            variations_list.append("low_contrast")
        if use_partial_edge:
            variations_list.append("partial_edge")

        noise_sigma = int(np.random.choice([8, 12, 18, 25, 35]))
        blur_sigma = float(np.random.choice([0.3, 0.5, 0.8, 1.1]))
        intensity_scale = float(np.round(np.random.uniform(0.85, 1.15), 3))

        if noise_sigma >= 25:
            variations_list.append("strong_noise")

        if len(variations_list) == 0:
            variations_list.append("standard")

        num_lesions = np.random.randint(2, 4) if use_multiple else 1

        final_mask = np.zeros((IMG_DIM, IMG_DIM), dtype=np.uint8)

        for _ in range(num_lesions):
            mask = create_shape_mask(lesion_class)

            if use_irregular:
                mask = make_irregular(mask)

            if use_partial_edge:
                mask = move_to_edge(mask)

            final_mask = np.maximum(final_mask, mask)

        if use_low_contrast:
            lesion_intensity = np.random.randint(80, 250)
        else:
            lesion_intensity = np.random.randint(350, 600)

        for z in range(N_SLICES):
            img = base.copy()
            lbl = np.zeros((IMG_DIM, IMG_DIM), dtype=np.uint8)

            slice_mask = final_mask.copy()

            if z == 0 or z == N_SLICES - 1:
                if np.random.rand() < 0.5:
                    slice_mask = binary_erosion(slice_mask, iterations=1).astype(np.uint8)

            img[slice_mask > 0] = lesion_intensity
            lbl[slice_mask > 0] = class_label

            image_volume[:, :, z] = add_noise_and_blur(
                img,
                noise_sigma,
                blur_sigma,
                intensity_scale
            )
            label_volume[:, :, z] = lbl

        variations = ";".join(variations_list)

    img_nii = nib.Nifti1Image(image_volume, affine=np.eye(4))
    lbl_nii = nib.Nifti1Image(label_volume, affine=np.eye(4))

    nib.save(img_nii, os.path.join(OUT_DIR, IMG_SUBDIR, f"{case_id}_0000.nii.gz"))
    nib.save(lbl_nii, os.path.join(OUT_DIR, LBL_SUBDIR, f"{case_id}.nii.gz"))

    metadata_rows.append({
        "case_id": case_id,
        "class": lesion_class,
        "variations": variations,
        "noise_sigma": noise_sigma,
        "blur_sigma": blur_sigma,
        "intensity_scale": intensity_scale
    })


for i in range(1, NUM_VOLUMES + 1):
    empty = i <= NUM_EMPTY
    generate_case(i, empty=empty)

dataset_json = {
    "channel_names": {
        "0": "CT"
    },
    "labels": classes,
    "numTraining": NUM_VOLUMES,
    "file_ending": ".nii.gz"
}

with open(os.path.join(OUT_DIR, "dataset.json"), "w") as f:
    json.dump(dataset_json, f, indent=4)

with open(os.path.join(OUT_DIR, "metadata.csv"), "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "case_id",
            "class",
            "variations",
            "noise_sigma",
            "blur_sigma",
            "intensity_scale"
        ]
    )
    writer.writeheader()
    writer.writerows(metadata_rows)

print(f"Dataset created: {OUT_DIR}")
print(f"Seed: {args.seed}")
