"""
Generate a harder synthetic 3D Shepp–Logan phantom dataset for SAM / nnU-Net experiments.

New variations:
1. Irregular lesion boundaries
2. Multiple lesions per volume
3. Different noise levels
4. Different contrast levels
5. Partial / edge lesions
"""

import os
import json
import csv
import numpy as np
import nibabel as nib

from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from skimage.draw import disk, polygon
from scipy.ndimage import (
    binary_fill_holes,
    binary_erosion,
    gaussian_filter,
    binary_dilation,
    binary_erosion as ndi_binary_erosion
)


OUT_DIR = "PhantomShapes_V2"
IMG_SUBDIR = "imagesTr"
LBL_SUBDIR = "labelsTr"

N_SLICES = 5
IMG_DIM = 256
NUM_VOLUMES = 150
NUM_EMPTY = 30
NUM_WITH_LESIONS = NUM_VOLUMES - NUM_EMPTY

CLASSES = ["circle", "square", "triangle"]
LABEL_MAP = {
    "background": 0,
    "circle": 1,
    "square": 2,
    "triangle": 3
}

np.random.seed(42)


images_dir = os.path.join(OUT_DIR, IMG_SUBDIR)
labels_dir = os.path.join(OUT_DIR, LBL_SUBDIR)

os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)


phantom = shepp_logan_phantom()
phantom = resize(
    phantom,
    (IMG_DIM, IMG_DIM),
    mode="reflect",
    anti_aliasing=True
)

phantom = np.rot90(phantom, k=1)
phantom_hu = (phantom * 2000 - 1000).astype(np.int16)

mask_phantom = phantom_hu > -500
mask_phantom = binary_fill_holes(mask_phantom)

# Safe zone
EROSION_MARGIN = 25
safe_zone = binary_erosion(mask_phantom, iterations=EROSION_MARGIN)
coords = np.argwhere(safe_zone)

safe_min_y, safe_min_x = coords.min(axis=0)
safe_max_y, safe_max_x = coords.max(axis=0)


def create_shape_mask(shape, center_y, center_x, size):
    mask = np.zeros((IMG_DIM, IMG_DIM), dtype=bool)

    if shape == "circle":
        rr, cc = disk((center_y, center_x), size, shape=mask.shape)
        mask[rr, cc] = True

    elif shape == "square":
        half = size // 2
        y1 = max(0, center_y - half)
        y2 = min(IMG_DIM, center_y + half)
        x1 = max(0, center_x - half)
        x2 = min(IMG_DIM, center_x + half)
        mask[y1:y2, x1:x2] = True

    elif shape == "triangle":
        h = int(size * 1.3)
        b = size
        verts = np.array([
            [center_y - h // 2, center_x],
            [center_y + h // 2, center_x - b // 2],
            [center_y + h // 2, center_x + b // 2]
        ])
        verts = np.clip(verts, 0, IMG_DIM - 1)
        rr, cc = polygon(verts[:, 0], verts[:, 1], shape=mask.shape)
        mask[rr, cc] = True

    return mask


def make_irregular(mask):
    """
    Make lesion boundary less perfect.
    """
    random_field = np.random.normal(0, 1, mask.shape)
    random_field = gaussian_filter(random_field, sigma=np.random.uniform(3, 7))

    dilated = binary_dilation(mask, iterations=np.random.randint(1, 4))
    eroded = ndi_binary_erosion(mask, iterations=np.random.randint(1, 3))

    irregular = np.where(random_field > 0, dilated, eroded)
    irregular = binary_fill_holes(irregular)

    return irregular.astype(bool)


def choose_position(partial=False):
    """
    Choose lesion position.
    If partial=True, place closer to phantom edge.
    """
    if partial:
        # Less safe margin: allows partial/edge lesions
        y = np.random.randint(safe_min_y, safe_max_y)
        x = np.random.randint(safe_min_x, safe_max_x)

        # push lesion toward one side
        direction = np.random.choice(["top", "bottom", "left", "right"])

        if direction == "top":
            y = safe_min_y + np.random.randint(0, 20)
        elif direction == "bottom":
            y = safe_max_y - np.random.randint(0, 20)
        elif direction == "left":
            x = safe_min_x + np.random.randint(0, 20)
        else:
            x = safe_max_x - np.random.randint(0, 20)

    else:
        y = np.random.randint(safe_min_y + 35, safe_max_y - 35)
        x = np.random.randint(safe_min_x + 35, safe_max_x - 35)

    return y, x



volume_types = ["empty"] * NUM_EMPTY

per_class = NUM_WITH_LESIONS // len(CLASSES)
remaining = NUM_WITH_LESIONS % len(CLASSES)

for i, cls in enumerate(CLASSES):
    count = per_class + (1 if i < remaining else 0)
    volume_types += [cls] * count

np.random.shuffle(volume_types)

dataset_entries = []
metadata_rows = []
class_counts = {k: 0 for k in ["empty"] + CLASSES}


for idx in range(1, NUM_VOLUMES + 1):

    main_class = volume_types[idx - 1]
    class_counts[main_class] += 1

    img_base = phantom_hu.copy().astype(np.float32)
    lbl_base = np.zeros((IMG_DIM, IMG_DIM), dtype=np.uint8)

    variation_flags = []

    if main_class != "empty":

        # Variation probabilities
        use_irregular = np.random.rand() < 0.55
        use_multiple = np.random.rand() < 0.30
        use_low_contrast = np.random.rand() < 0.45
        use_partial = np.random.rand() < 0.25

        if use_irregular:
            variation_flags.append("irregular")
        if use_multiple:
            variation_flags.append("multiple_lesions")
        if use_low_contrast:
            variation_flags.append("low_contrast")
        if use_partial:
            variation_flags.append("partial_edge")

        n_lesions = np.random.randint(2, 4) if use_multiple else 1

        for lesion_id in range(n_lesions):

            size = np.random.randint(18, 65)
            y0, x0 = choose_position(partial=use_partial)

            lesion_mask = create_shape_mask(main_class, y0, x0, size)

            if use_irregular:
                lesion_mask = make_irregular(lesion_mask)

            # Keep lesion inside phantom region
            lesion_mask = lesion_mask & mask_phantom

            # Different contrast levels
            if use_low_contrast:
                les_hu = np.random.choice([80, 120, 160, 200, 250])
            else:
                les_hu = np.random.choice([350, 400, 450, 500, 550, 600])

            # Add soft intensity variation inside lesion
            lesion_texture = np.random.normal(0, 25, img_base.shape)
            lesion_texture = gaussian_filter(lesion_texture, sigma=2)

            img_base[lesion_mask] = les_hu + lesion_texture[lesion_mask]
            lbl_base[lesion_mask] = LABEL_MAP[main_class]

    else:
        variation_flags.append("empty")

   
    img_slices = []

    # Random global difficulty per volume
    noise_sigma = np.random.choice([8, 12, 18, 25, 35])
    blur_sigma = np.random.choice([0.3, 0.5, 0.8, 1.1])
    intensity_scale = np.random.uniform(0.85, 1.15)

    if noise_sigma >= 25:
        variation_flags.append("strong_noise")

    for _ in range(N_SLICES):
        sl = img_base.copy()

        # Gaussian noise
        sl += np.random.normal(0, noise_sigma, sl.shape)

        # Blur
        sl = gaussian_filter(sl, sigma=blur_sigma)

        # Intensity scaling
        sl *= intensity_scale

        # Clip to CT-like range
        sl = np.clip(sl, -1000, 1000)

        img_slices.append(sl.astype(np.int16))

    img_vol = np.stack(img_slices, axis=-1)
    lbl_vol = np.stack([lbl_base] * N_SLICES, axis=-1)

   
    case_id = f"Case{idx:03d}"
    img_name = f"{case_id}_0000.nii.gz"
    lbl_name = f"{case_id}.nii.gz"

    nib.save(
        nib.Nifti1Image(img_vol, np.eye(4)),
        os.path.join(images_dir, img_name)
    )

    nib.save(
        nib.Nifti1Image(lbl_vol, np.eye(4)),
        os.path.join(labels_dir, lbl_name)
    )

    dataset_entries.append({
        "image": f"./{IMG_SUBDIR}/{img_name}",
        "label": f"./{LBL_SUBDIR}/{lbl_name}"
    })

    metadata_rows.append({
        "case_id": case_id,
        "class": main_class,
        "variations": ";".join(variation_flags),
        "noise_sigma": noise_sigma,
        "blur_sigma": blur_sigma,
        "intensity_scale": round(float(intensity_scale), 3)
    })


with open(os.path.join(OUT_DIR, "dataset.json"), "w") as f:
    json.dump({
        "name": "PhantomShapes3D_V2",
        "description": "Harder synthetic 3D Shepp-Logan phantom dataset with irregular lesions, multiple lesions, low contrast, noise variation, and partial edge lesions.",
        "tensorImageSize": "3D",
        "modality": {"0": "CT"},
        "channel_names": {"0": "CT"},
        "labels": {
            "background": "0",
            "circle": "1",
            "square": "2",
            "triangle": "3"
        },
        "numTraining": NUM_VOLUMES,
        "numTest": 0,
        "training": dataset_entries,
        "test": []
    }, f, indent=2)


with open(os.path.join(OUT_DIR, "seg_info.json"), "w") as f:
    json.dump({
        "algorithm": "Ground truth",
        "seg_info": [
            {"label_name": "circle", "label_int": "1"},
            {"label_name": "square", "label_int": "2"},
            {"label_name": "triangle", "label_int": "3"}
        ]
    }, f, indent=2)


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

print("\nGenerated dataset:", OUT_DIR)
print("Class distribution:", class_counts)
print("Metadata saved to:", os.path.join(OUT_DIR, "metadata.csv"))
