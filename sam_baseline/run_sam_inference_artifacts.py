import os
import cv2
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamPredictor

# -------------------------
# Arguments
# -------------------------
parser = argparse.ArgumentParser()

parser.add_argument("--image_dir", required=True)
parser.add_argument("--mask_dir", required=True)
parser.add_argument("--pred_dir", required=True)
parser.add_argument("--vis_dir", default=None)

parser.add_argument(
    "--checkpoint",
    default="checkpoints/sam_vit_b_01ec64.pth"
)

parser.add_argument(
    "--model_type",
    default="vit_b"
)

args = parser.parse_args()

IMAGE_DIR = args.image_dir
MASK_DIR = args.mask_dir
PRED_DIR = args.pred_dir
VIS_DIR = args.vis_dir if args.vis_dir is not None else PRED_DIR + "_visuals"

os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

# -------------------------
# Load SAM
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[args.model_type](
    checkpoint=args.checkpoint
)

sam.to(device=device)
predictor = SamPredictor(sam)

# -------------------------
# Helper function
# -------------------------
def get_bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    return np.array([x_min, y_min, x_max, y_max])


# -------------------------
# Run inference
# -------------------------
files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.endswith(".png")
])

processed = 0
skipped = 0

for fname in files:
    image_path = os.path.join(IMAGE_DIR, fname)
    mask_path = os.path.join(MASK_DIR, fname)

    if not os.path.exists(mask_path):
        print(f"Missing mask: {fname}")
        skipped += 1
        continue

    image = cv2.imread(image_path)
    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or gt_mask is None:
        print(f"Could not read image or mask: {fname}")
        skipped += 1
        continue

    gt_binary = (gt_mask > 0).astype(np.uint8)

    bbox = get_bbox_from_mask(gt_binary)

    if bbox is None:
        skipped += 1
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(image_rgb)

    masks, scores, logits = predictor.predict(
        box=bbox,
        multimask_output=True
    )

    best_idx = np.argmax(scores)
    pred_mask = masks[best_idx].astype(np.uint8) * 255

    cv2.imwrite(
        os.path.join(PRED_DIR, fname),
        pred_mask
    )

    # Visualization
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gt_binary, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask, cmap="gray")
    plt.title("SAM Prediction")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            VIS_DIR,
            fname.replace(".png", "_result.png")
        )
    )
    plt.close()

    processed += 1

print("===================================")
print(f"Processed images: {processed}")
print(f"Skipped images: {skipped}")
print(f"Raw masks saved in: {PRED_DIR}")
print(f"Visual results saved in: {VIS_DIR}")
print("===================================")

