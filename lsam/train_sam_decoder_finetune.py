"""
Generic Fine-Tune Control for M2 fix.

Identical setup to LSAM training EXCEPT:
- No LesionSpecializedAdapter (no specialized blocks)
- No BoundaryHead (no boundary loss)
- Only the mask decoder is unfrozen and trained

This isolates how much of the SAM→LSAM gain comes from
in-distribution fine-tuning alone vs. the specialized architecture.

Run this, evaluate with the same LSAM inference + evaluation scripts,
and add the result as a "SAM Fine-Tuned (decoder only)" row in Table 13.
"""

import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from segment_anything import sam_model_registry


PROJECT_DIR = "/home/shqipe/Desktop/sam_phantom_project"

IMAGE_DIR = f"{PROJECT_DIR}/SAM_10_runs/LSAM_train_dataset/images"
MASK_DIR  = f"{PROJECT_DIR}/SAM_10_runs/LSAM_train_dataset/masks"

SAM_CHECKPOINT = f"{PROJECT_DIR}/checkpoints/sam_vit_b_01ec64.pth"
OUT_DIR = f"{PROJECT_DIR}/lsam_project/checkpoints"
os.makedirs(OUT_DIR, exist_ok=True)


DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE   = 1024
MASK_SIZE  = 256
BATCH_SIZE = 1
EPOCHS     = 20
LR         = 1e-5



class LSAMDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.images   = sorted(glob(os.path.join(image_dir, "*.png")))
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.images)

    def get_box(self, mask):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        pad = 5
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(mask.shape[1] - 1, x2 + pad)
        y2 = min(mask.shape[0] - 1, y2 + pad)
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def __getitem__(self, idx):
        img_path  = self.images[idx]
        fname     = os.path.basename(img_path)
        mask_path = os.path.join(self.mask_dir, fname)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask  = (mask > 0).astype(np.float32)

        box = self.get_box(mask)
        if box is None:
            box = np.array([0, 0, 1, 1], dtype=np.float32)

        h, w = image.shape[:2]

        image_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        mask_resized  = cv2.resize(
            mask, (MASK_SIZE, MASK_SIZE),
            interpolation=cv2.INTER_NEAREST
        )

        scale_x = IMG_SIZE / w
        scale_y = IMG_SIZE / h
        box[0] *= scale_x;  box[2] *= scale_x
        box[1] *= scale_y;  box[3] *= scale_y

        image_tensor = torch.tensor(image_resized).permute(2, 0, 1).float()
        mask_tensor  = torch.tensor(mask_resized).unsqueeze(0).float()
        box_tensor   = torch.tensor(box).float()

        return image_tensor, mask_tensor, box_tensor


def dice_loss(pred, target, eps=1e-6):
    pred         = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    union        = pred.sum() + target.sum()
    return 1 - (2.0 * intersection + eps) / (union + eps)


def main():
    print("Device:", DEVICE)
    print("Control: decoder-only fine-tune (no adapter, no boundary loss)")

    dataset = LSAMDataset(IMAGE_DIR, MASK_DIR)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print("Training samples:", len(dataset))

    sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
    sam.to(DEVICE)

    # Freeze encoder and prompt encoder — same as LSAM
    for p in sam.image_encoder.parameters():
        p.requires_grad = False
    for p in sam.prompt_encoder.parameters():
        p.requires_grad = False

    # Only train the mask decoder — no adapter, no boundary head
    for p in sam.mask_decoder.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        sam.mask_decoder.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    bce       = nn.BCEWithLogitsLoss()
    best_loss = 9999

    for epoch in range(EPOCHS):
        sam.train()
        epoch_loss = 0.0

        for image, mask, box in tqdm(loader):
            image = image.to(DEVICE)
            mask  = mask.to(DEVICE)
            box   = box.to(DEVICE)

            image = sam.preprocess(image)

            with torch.no_grad():
                image_embedding = sam.image_encoder(image)
                sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                    points=None,
                    boxes=box,
                    masks=None
                )

            # No adapter — pass encoder output directly to decoder
            low_res_masks, _ = sam.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )

            # Segmentation loss only — no boundary loss
            loss = bce(low_res_masks, mask) + dice_loss(low_res_masks, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(loader)
        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(
                {
                    "sam_mask_decoder": sam.mask_decoder.state_dict(),
                    "epoch": epoch + 1,
                    "loss": best_loss
                },
                os.path.join(OUT_DIR, "sam_decoder_finetune_best.pth")
            )
            print("Saved best model.")

    print("Training finished. Best loss:", best_loss)
    print("Checkpoint: lsam_project/checkpoints/sam_decoder_finetune_best.pth")


if __name__ == "__main__":
    main()
