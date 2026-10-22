import os
import re
import random
import shutil
import numpy as np
import cv2
from glob import glob

random.seed(42)  # reproducibility

PROJECT_DIR = "/home/shqipe/Desktop/sam_phantom_project"
BUSI_ROOT = f"{PROJECT_DIR}/BUSI_dataset/Dataset_BUSI_with_GT"

OUT_DIR = f"{PROJECT_DIR}/BUSI_subset"
OUT_IMAGES = f"{OUT_DIR}/images"
OUT_MASKS = f"{OUT_DIR}/masks"

os.makedirs(OUT_IMAGES, exist_ok=True)
os.makedirs(OUT_MASKS, exist_ok=True)

N_PER_CLASS = 25  # adjust if you want a bigger/smaller subset


def get_base_images(class_dir):
    """Returns list of base image filenames (excluding any _mask files)."""
    all_pngs = glob(os.path.join(class_dir, "*.png"))
    base_images = [
        f for f in all_pngs
        if "_mask" not in os.path.basename(f)
    ]
    return base_images


def merge_masks(base_image_path):
    """Finds and merges all mask files belonging to one base image into a single binary mask."""
    base_name = os.path.splitext(os.path.basename(base_image_path))[0]
    class_dir = os.path.dirname(base_image_path)

    # matches "name_mask.png", "name_mask_1.png", "name_mask_2.png", etc.
    mask_pattern = re.escape(base_name) + r"_mask(_\d+)?\.png"
    mask_files = [
        f for f in os.listdir(class_dir)
        if re.fullmatch(mask_pattern, f)
    ]

    if not mask_files:
        return None

    merged = None
    for mf in mask_files:
        mask_path = os.path.join(class_dir, mf)
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        m_bin = (m > 0).astype(np.uint8)
        merged = m_bin if merged is None else np.maximum(merged, m_bin)

    return merged


def process_class(class_name, n_select):
    class_dir = os.path.join(BUSI_ROOT, class_name)
    base_images = get_base_images(class_dir)

    random.shuffle(base_images)

    selected = []
    for img_path in base_images:
        if len(selected) >= n_select:
            break

        mask = merge_masks(img_path)
        if mask is None or mask.sum() == 0:
            continue  # skip images with no valid mask

        selected.append((img_path, mask))

    print(f"{class_name}: selected {len(selected)} / requested {n_select}")
    return selected


def save_subset(class_name, selected):
    for img_path, mask in selected:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        # sanitize filename (remove spaces/parentheses for safety)
        clean_name = re.sub(r"[^\w]", "_", base_name)
        out_fname = f"{class_name}_{clean_name}.png"

        image = cv2.imread(img_path)
        if image is None:
            continue

        cv2.imwrite(os.path.join(OUT_IMAGES, out_fname), image)

        mask_255 = (mask * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(OUT_MASKS, out_fname), mask_255)


def main():
    benign_selected = process_class("benign", N_PER_CLASS)
    malignant_selected = process_class("malignant", N_PER_CLASS)

    save_subset("benign", benign_selected)
    save_subset("malignant", malignant_selected)

    print("\nDone.")
    print(f"Total images saved: {len(benign_selected) + len(malignant_selected)}")
    print(f"Images dir: {OUT_IMAGES}")
    print(f"Masks dir: {OUT_MASKS}")


if __name__ == "__main__":
    main()
