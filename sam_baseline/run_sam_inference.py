import os
import argparse
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor


parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", required=True)
parser.add_argument("--mask_dir", required=True)
parser.add_argument("--pred_dir", required=True)
parser.add_argument("--checkpoint", default="checkpoints/sam_vit_b_01ec64.pth")
parser.add_argument("--model_type", default="vit_b")
args = parser.parse_args()

IMAGE_DIR = args.image_dir
MASK_DIR = args.mask_dir
PRED_DIR = args.pred_dir
CHECKPOINT = args.checkpoint
MODEL_TYPE = args.model_type

os.makedirs(PRED_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT)
sam.to(device=device)
predictor = SamPredictor(sam)


def get_box_from_mask(mask):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    return np.array([x_min, y_min, x_max, y_max])


files = sorted(os.listdir(IMAGE_DIR))

for fname in files:
    image_path = os.path.join(IMAGE_DIR, fname)
    mask_path = os.path.join(MASK_DIR, fname)

    image = cv2.imread(image_path)

    if image is None:
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if gt_mask is None:
        continue

    box = get_box_from_mask(gt_mask)

    if box is None:
        continue

    predictor.set_image(image)

    masks, scores, logits = predictor.predict(
        box=box,
        multimask_output=True
    )

    best_idx = np.argmax(scores)
    pred_mask = masks[best_idx].astype(np.uint8) * 255

    cv2.imwrite(os.path.join(PRED_DIR, fname), pred_mask)

print(f"Done. Predictions saved in: {PRED_DIR}")
